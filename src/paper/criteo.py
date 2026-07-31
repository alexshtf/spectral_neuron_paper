import fcntl
import json
import pickle
import shutil
import sys
import tempfile
from compression import zstd
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from hashlib import file_digest
from pathlib import Path
from typing import ClassVar, Literal, TextIO
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from paper.compression import ZSTD_LEVEL, open_dataset_file
from paper.shuffling import ShuffledEpochs


NUM_NUMERIC_FIELDS = 13
NUM_CATEGORICAL_FIELDS = 26
NUM_FIELDS = NUM_NUMERIC_FIELDS + NUM_CATEGORICAL_FIELDS

LABEL = "label"
NUMERIC_COLUMNS = tuple(f"I{i}" for i in range(1, NUM_NUMERIC_FIELDS + 1))
CATEGORICAL_COLUMNS = tuple(f"C{i}" for i in range(1, NUM_CATEGORICAL_FIELDS + 1))
COLUMNS = (LABEL, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)

CACHE_VERSION = 2
# Bump when the corresponding fitted or encoded artifact changes meaning.
PREPROCESSOR_CACHE_VERSION = 2
ENCODED_CACHE_VERSION = 4
ENCODED_CHUNK_SIZE = 2**18
FEATURE_ID_DTYPE = np.dtype(np.uint16)
NUMERICS_FILE = "numerics.dat"
CATEGORICALS_FILE = "categoricals.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"
MISSING_NUMERIC = np.iinfo(np.int32).min

type PreprocessingKind = Literal["bucket", "hybrid"]
type FeatureArrays = tuple[np.ndarray, np.ndarray | None]


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _bucket_numeric(values: np.ndarray) -> np.ndarray:
    """Winner-style categorical buckets for Criteo's integer fields."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        present = values != MISSING_NUMERIC
    else:
        present = ~np.isnan(values)
    numeric = values.astype(np.float64)
    small = present & (numeric <= 2)
    large = present & ~small

    buckets = np.zeros(values.shape, dtype=np.int32)
    buckets[small] = numeric[small].astype(np.int32)
    buckets[large] = np.floor(np.square(np.log(numeric[large]))).astype(np.int32) + 2
    return buckets


_HEX_VALUE = np.zeros(256, dtype=np.uint8)
_HEX_VALUE[ord("0") : ord("9") + 1] = np.arange(10)
_HEX_VALUE[ord("a") : ord("f") + 1] = np.arange(10, 16)
_HEX_VALUE[ord("A") : ord("F") + 1] = np.arange(10, 16)


def _parse_hex(values: pd.Series) -> np.ndarray:
    encoded = values.fillna("").to_numpy(dtype="S8").view(np.uint8).reshape(-1, 8)
    present = encoded[:, 0] != 0
    parsed = np.zeros(len(encoded), dtype=np.uint32)
    for column in encoded.T:
        parsed = (parsed << 4) | _HEX_VALUE[column]
    parsed[present & (parsed == 0)] = 1
    parsed[~present] = 0
    return parsed


def _parse_numerics(chunk: pd.DataFrame) -> np.ndarray:
    raw = chunk.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    present = ~np.isnan(raw)
    values = np.full(raw.shape, MISSING_NUMERIC, dtype=np.int32)
    values[present] = raw[present].astype(np.int32)
    return values


def prepare_corpus(
    raw_path: Path,
    cache_dir: Path,
    *,
    chunk_size: int = 1_000_000,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> "CriteoCorpus":
    """Convert the headerless challenge TSV to compact memory-mapped arrays."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / METADATA_FILE
    with _exclusive_lock(cache_dir / ".corpus.lock"):
        if metadata_path.exists():
            corpus = CriteoCorpus.open(cache_dir)
            if progress:
                tqdm.write(
                    f"Criteo corpus: {corpus.rows:,} rows (cached)",
                    file=sys.stderr if progress_file is None else progress_file,
                )
            return corpus

        with tempfile.TemporaryDirectory(prefix=".corpus-", dir=cache_dir) as tmp:
            temporary = Path(tmp)
            numeric_tmp = temporary / NUMERICS_FILE
            categorical_tmp = temporary / CATEGORICALS_FILE
            label_tmp = temporary / LABELS_FILE
            rows = 0
            dtypes = {
                LABEL: np.uint8,
                **dict.fromkeys(CATEGORICAL_COLUMNS, "string"),
            }

            with (
                open_dataset_file(raw_path) as raw_file,
                numeric_tmp.open("wb") as numeric_file,
                categorical_tmp.open("wb") as categorical_file,
                label_tmp.open("wb") as label_file,
            ):
                chunks = pd.read_csv(
                    raw_file,
                    sep="\t",
                    header=None,
                    names=COLUMNS,
                    dtype=dtypes,
                    chunksize=chunk_size,
                )
                progress_bar = tqdm(
                    desc="Preparing Criteo corpus",
                    unit="rows",
                    unit_scale=True,
                    disable=not progress,
                    file=progress_file,
                    dynamic_ncols=True,
                )
                with progress_bar:
                    for chunk in chunks:
                        categoricals = np.empty(
                            (len(chunk), NUM_CATEGORICAL_FIELDS), dtype=np.uint32
                        )
                        for field, column in enumerate(CATEGORICAL_COLUMNS):
                            categoricals[:, field] = _parse_hex(chunk[column])

                        _parse_numerics(chunk).tofile(numeric_file)
                        categoricals.tofile(categorical_file)
                        chunk[LABEL].to_numpy(dtype=np.uint8).tofile(label_file)
                        rows += len(chunk)
                        progress_bar.update(len(chunk))

            numeric_tmp.replace(cache_dir / NUMERICS_FILE)
            categorical_tmp.replace(cache_dir / CATEGORICALS_FILE)
            label_tmp.replace(cache_dir / LABELS_FILE)
            metadata_tmp = temporary / METADATA_FILE
            metadata_tmp.write_text(
                json.dumps({"version": CACHE_VERSION, "rows": rows})
            )
            metadata_tmp.replace(metadata_path)
        return CriteoCorpus.open(cache_dir)


@dataclass(frozen=True)
class CriteoCorpus:
    cache_dir: Path
    rows: int

    @classmethod
    def open(cls, cache_dir: Path) -> "CriteoCorpus":
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        version = metadata.get("version")
        if version != CACHE_VERSION:
            raise ValueError(
                f"{cache_dir} uses Criteo cache version {version or 1}; "
                f"expected {CACHE_VERSION}"
            )
        return cls(cache_dir=cache_dir, rows=metadata["rows"])

    @property
    def train_stop(self) -> int:
        return self.rows * 8 // 10

    @property
    def val_stop(self) -> int:
        return self.rows * 9 // 10

    def _memmap(
        self,
        filename: str,
        dtype: type[np.generic],
        *shape: int,
    ) -> np.memmap:
        return np.memmap(
            self.cache_dir / filename,
            mode="r",
            dtype=dtype,
            shape=(self.rows, *shape),
        )

    def numerics(self) -> np.memmap:
        return self._memmap(NUMERICS_FILE, np.int32, NUM_NUMERIC_FIELDS)

    def categoricals(self) -> np.memmap:
        return self._memmap(CATEGORICALS_FILE, np.uint32, NUM_CATEGORICAL_FIELDS)

    def labels(self) -> np.memmap:
        return self._memmap(LABELS_FILE, np.uint8)

    def shuffled_epochs(self, seed: int) -> ShuffledEpochs:
        return ShuffledEpochs(self.cache_dir, self.train_stop, seed)


def _categorical_ids(
    values: np.ndarray,
    vocabulary: np.ndarray,
    *,
    offset: int,
) -> np.ndarray:
    encoded = np.full(values.shape, offset + 1, dtype=np.int32)
    positions = np.searchsorted(vocabulary, values)
    known = positions < len(vocabulary)
    known[known] = vocabulary[positions[known]] == values[known]
    encoded[known] = offset + 2 + positions[known]
    encoded[values == 0] = offset
    return encoded


def _categorical_sizes(
    vocabularies: tuple[np.ndarray, ...],
) -> np.ndarray:
    return np.fromiter(
        (len(vocabulary) + 2 for vocabulary in vocabularies),
        dtype=np.int64,
        count=NUM_CATEGORICAL_FIELDS,
    )


def _field_offsets(sizes: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes[:-1]))).astype(
        np.int32
    )


def _check_inputs(numerics: np.ndarray, categoricals: np.ndarray) -> None:
    if numerics.shape[-1:] != (NUM_NUMERIC_FIELDS,):
        raise ValueError(
            f"expected numerical shape (..., {NUM_NUMERIC_FIELDS}); "
            f"got {numerics.shape}"
        )
    expected = numerics.shape[:-1] + (NUM_CATEGORICAL_FIELDS,)
    if categoricals.shape != expected:
        raise ValueError(
            f"expected categorical shape {expected}; got {categoricals.shape}"
        )


@dataclass(frozen=True)
class BucketPreprocessor:
    numeric_minimums: np.ndarray
    numeric_maximums: np.ndarray
    categorical_vocabularies: tuple[np.ndarray, ...]
    kind: ClassVar[PreprocessingKind] = "bucket"

    @property
    def field_sizes(self) -> np.ndarray:
        numeric = self.numeric_maximums.astype(np.int64) - self.numeric_minimums + 3
        return np.concatenate(
            (numeric, _categorical_sizes(self.categorical_vocabularies))
        )

    @property
    def num_features(self) -> int:
        return int(self.field_sizes.sum())

    @property
    def field_offsets(self) -> np.ndarray:
        return _field_offsets(self.field_sizes)

    def encode(
        self,
        numerics: np.ndarray,
        categoricals: np.ndarray,
    ) -> FeatureArrays:
        _check_inputs(numerics, categoricals)
        shape = numerics.shape[:-1] + (NUM_FIELDS,)
        feature_ids = np.empty(shape, dtype=np.int32)
        buckets = _bucket_numeric(numerics)
        offsets = self.field_offsets

        for field, (offset, minimum, maximum) in enumerate(
            zip(
                offsets[:NUM_NUMERIC_FIELDS],
                self.numeric_minimums,
                self.numeric_maximums,
                strict=True,
            )
        ):
            raw = numerics[..., field]
            values = buckets[..., field]
            ids = feature_ids[..., field]
            ids.fill(offset + 1)
            present = raw != MISSING_NUMERIC
            known = present & (minimum <= values) & (values <= maximum)
            ids[known] = offset + 2 + values[known] - minimum
            ids[~present] = offset

        for category, (offset, vocabulary) in enumerate(
            zip(
                offsets[NUM_NUMERIC_FIELDS:],
                self.categorical_vocabularies,
                strict=True,
            )
        ):
            feature_ids[..., NUM_NUMERIC_FIELDS + category] = _categorical_ids(
                categoricals[..., category],
                vocabulary,
                offset=int(offset),
            )

        return feature_ids, None


@dataclass(frozen=True)
class HybridPreprocessor:
    categorical_vocabularies: tuple[np.ndarray, ...]
    negative_values: tuple[np.ndarray, ...]
    positive_mean: np.ndarray
    positive_scale: np.ndarray
    kind: ClassVar[PreprocessingKind] = "hybrid"

    @property
    def field_sizes(self) -> np.ndarray:
        numeric = np.fromiter(
            (len(values) + 4 for values in self.negative_values),
            dtype=np.int64,
            count=NUM_NUMERIC_FIELDS,
        )
        return np.concatenate(
            (numeric, _categorical_sizes(self.categorical_vocabularies))
        )

    @property
    def num_numeric_features(self) -> int:
        return sum(len(values) + 4 for values in self.negative_values)

    @property
    def num_features(self) -> int:
        return int(self.field_sizes.sum())

    @property
    def field_offsets(self) -> np.ndarray:
        return _field_offsets(self.field_sizes)

    def encode(
        self,
        numerics: np.ndarray,
        categoricals: np.ndarray,
    ) -> FeatureArrays:
        _check_inputs(numerics, categoricals)
        shape = numerics.shape[:-1] + (NUM_FIELDS,)
        feature_ids = np.empty(shape, dtype=np.int32)
        feature_values = np.ones(shape, dtype=np.float32)
        offsets = self.field_offsets

        for field, (offset, negatives) in enumerate(
            zip(
                offsets[:NUM_NUMERIC_FIELDS],
                self.negative_values,
                strict=True,
            )
        ):
            raw = numerics[..., field]
            ids = feature_ids[..., field]
            values = feature_values[..., field]
            positive_id = int(offset + 3 + len(negatives))

            ids.fill(positive_id)
            positive = raw > 0
            values[positive] = (
                (np.log1p(raw[positive]) - self.positive_mean[field])
                / self.positive_scale[field]
            ).astype(np.float32)

            missing = raw == MISSING_NUMERIC
            zero = raw == 0
            negative = (raw < 0) & ~missing
            ids[missing] = offset
            ids[zero] = offset + 1
            ids[negative] = offset + 2

            positions = np.searchsorted(negatives, raw)
            valid = positions < len(negatives)
            known = negative & valid
            known[valid] &= negatives[positions[valid]] == raw[valid]
            ids[known] = offset + 3 + positions[known]

        for category, (offset, vocabulary) in enumerate(
            zip(
                offsets[NUM_NUMERIC_FIELDS:],
                self.categorical_vocabularies,
                strict=True,
            )
        ):
            field = NUM_NUMERIC_FIELDS + category
            feature_ids[..., field] = _categorical_ids(
                categoricals[..., category],
                vocabulary,
                offset=int(offset),
            )

        return feature_ids, feature_values


type FittedPreprocessor = BucketPreprocessor | HybridPreprocessor


def _preprocessor_path(
    corpus: CriteoCorpus,
    kind: PreprocessingKind,
    *,
    sample_size: int,
    sample_seed: int,
    min_count: int,
) -> Path:
    return corpus.cache_dir / (
        f"preprocessor-v{PREPROCESSOR_CACHE_VERSION}_{kind}_"
        f"sample{sample_size}_seed{sample_seed}_min{min_count}.pkl.zstd"
    )


def _save_preprocessor(preprocessor: FittedPreprocessor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with _exclusive_lock(path.with_name(f".{path.name}.lock")):
        if path.exists():
            return
        try:
            with zstd.open(temporary, "wb", level=ZSTD_LEVEL) as file:
                pickle.dump(preprocessor, file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _compress_preprocessor(source: Path, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with _exclusive_lock(path.with_name(f".{path.name}.lock")):
        if path.exists():
            return
        try:
            with (
                source.open("rb") as source_file,
                zstd.open(temporary, "wb", level=ZSTD_LEVEL) as compressed_file,
            ):
                shutil.copyfileobj(source_file, compressed_file)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_preprocessor(path: Path) -> FittedPreprocessor:
    # These files are trusted, local artifacts created by this module.
    with zstd.open(path, "rb") as file:
        preprocessor = pickle.load(file)
    if not isinstance(preprocessor, BucketPreprocessor | HybridPreprocessor):
        raise TypeError(f"{path} does not contain a fitted Criteo preprocessor")
    return preprocessor


def fit_preprocessors(
    corpus: CriteoCorpus,
    kinds: Iterable[PreprocessingKind],
    *,
    sample_size: int,
    sample_seed: int,
    min_count: int,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> dict[PreprocessingKind, Path]:
    kinds = tuple(dict.fromkeys(kinds))
    if not kinds:
        return {}
    if not set(kinds) <= {"bucket", "hybrid"}:
        raise ValueError(f"unknown preprocessing kind in {kinds}")
    if not 0 < sample_size <= corpus.train_stop:
        raise ValueError(
            f"sample_size must be in [1, {corpus.train_stop}]; got {sample_size}"
        )
    paths = {
        kind: _preprocessor_path(
            corpus,
            kind,
            sample_size=sample_size,
            sample_seed=sample_seed,
            min_count=min_count,
        )
        for kind in kinds
    }
    for path in paths.values():
        legacy = path.with_suffix("")
        if not path.exists() and legacy.exists():
            _compress_preprocessor(legacy, path)
    missing = [kind for kind, path in paths.items() if not path.exists()]
    if progress:
        output = sys.stderr if progress_file is None else progress_file
        for kind in kinds:
            if kind not in missing:
                tqdm.write(
                    f"{kind.title()} preprocessor: {sample_size:,} rows (cached)",
                    file=output,
                )
    if not missing:
        return paths

    rows = np.random.default_rng(sample_seed).choice(
        corpus.train_stop,
        size=sample_size,
        replace=False,
        shuffle=False,
    )
    rows.sort()
    categoricals = corpus.categoricals()
    fields = tqdm(
        range(NUM_CATEGORICAL_FIELDS),
        desc=f"Fitting categorical vocabulary on {sample_size:,} rows",
        unit="field",
        disable=not progress,
        file=progress_file,
        dynamic_ncols=True,
    )
    categorical_vocabularies = []
    for field in fields:
        values, counts = np.unique(categoricals[rows, field], return_counts=True)
        categorical_vocabularies.append(values[(values != 0) & (counts >= min_count)])
    vocabularies = tuple(categorical_vocabularies)

    numeric_minimums = np.empty(NUM_NUMERIC_FIELDS, dtype=np.int32)
    numeric_maximums = np.empty(NUM_NUMERIC_FIELDS, dtype=np.int32)
    negative_values = []
    positive_mean = np.empty(NUM_NUMERIC_FIELDS)
    positive_scale = np.empty(NUM_NUMERIC_FIELDS)
    numerics = corpus.numerics()
    fields = tqdm(
        range(NUM_NUMERIC_FIELDS),
        desc=f"Fitting numeric preprocessing on {sample_size:,} rows",
        unit="field",
        disable=not progress,
        file=progress_file,
        dynamic_ncols=True,
    )
    for field in fields:
        values = numerics[rows, field]
        if "bucket" in missing:
            present = values != MISSING_NUMERIC
            buckets = _bucket_numeric(values[present])
            numeric_minimums[field] = buckets.min()
            numeric_maximums[field] = buckets.max()
        if "hybrid" in missing:
            negative_values.append(
                np.unique(values[(values < 0) & (values != MISSING_NUMERIC)])
            )
            logged = np.log1p(values[values > 0].astype(np.float64))
            positive_mean[field] = logged.mean() if len(logged) else 0.0
            scale = logged.std() if len(logged) else 1.0
            positive_scale[field] = scale if scale > 0 else 1.0

    if "bucket" in missing:
        _save_preprocessor(
            BucketPreprocessor(
                numeric_minimums,
                numeric_maximums,
                vocabularies,
            ),
            paths["bucket"],
        )

    if "hybrid" in missing:
        _save_preprocessor(
            HybridPreprocessor(
                vocabularies,
                tuple(negative_values),
                positive_mean,
                positive_scale,
            ),
            paths["hybrid"],
        )

    return paths


type BinaryBatch = tuple[tuple[torch.Tensor, ...], torch.Tensor]
type EncodedArrays = tuple[np.ndarray, np.ndarray | None, np.ndarray]


def load_encoded(path: Path) -> EncodedArrays:
    values_path = path / "feature_values.npy"
    return (
        np.load(path / "feature_ids.npy", mmap_mode="c", allow_pickle=False),
        (
            np.load(values_path, mmap_mode="c", allow_pickle=False)
            if values_path.exists()
            else None
        ),
        np.load(path / "labels.npy", mmap_mode="c", allow_pickle=False),
    )


@dataclass(frozen=True)
class EncodedData:
    num_features: int
    field_offsets: tuple[int, ...]
    train: Path
    holdout: Path
    validation_rows: int


def _valid_encoded_split(
    path: Path,
    *,
    row_count: int,
    kind: PreprocessingKind,
) -> bool:
    if not (path / "complete").exists():
        return False
    try:
        feature_ids = np.load(
            path / "feature_ids.npy", mmap_mode="r", allow_pickle=False
        )
        labels = np.load(path / "labels.npy", mmap_mode="r", allow_pickle=False)
        values_path = path / "feature_values.npy"
        feature_values = (
            np.load(values_path, mmap_mode="r", allow_pickle=False)
            if values_path.exists()
            else None
        )
    except EOFError, OSError, ValueError:
        return False

    shape = (row_count, NUM_FIELDS)
    valid_values = feature_values is None
    if kind == "hybrid":
        valid_values = (
            feature_values is not None
            and feature_values.dtype == np.float32
            and feature_values.shape == shape
        )
    return (
        feature_ids.dtype == FEATURE_ID_DTYPE
        and feature_ids.shape == shape
        and labels.dtype == np.uint8
        and labels.shape == (row_count,)
        and valid_values
    )


def _write_encoded_split(
    corpus: CriteoCorpus,
    preprocessor: FittedPreprocessor,
    field_offsets: np.ndarray,
    rows: range,
    path: Path,
    *,
    description: str,
    chunk_size: int,
    progress: bool,
    progress_file: TextIO | None,
) -> None:
    row_count = len(rows)
    shape = (row_count, NUM_FIELDS)
    feature_ids = np.lib.format.open_memmap(
        path / "feature_ids.npy", mode="w+", dtype=FEATURE_ID_DTYPE, shape=shape
    )
    feature_values = (
        np.lib.format.open_memmap(
            path / "feature_values.npy",
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        if preprocessor.kind == "hybrid"
        else None
    )
    encoded_labels = np.lib.format.open_memmap(
        path / "labels.npy", mode="w+", dtype=np.uint8, shape=(row_count,)
    )
    numerics = corpus.numerics()
    categoricals = corpus.categoricals()
    labels = corpus.labels()
    progress_bar = tqdm(
        total=row_count,
        desc=description,
        unit="rows",
        unit_scale=True,
        disable=not progress,
        file=progress_file,
        dynamic_ncols=True,
    )
    with progress_bar:
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            source = rows[start:stop]
            batch_ids, batch_values = preprocessor.encode(
                np.asarray(numerics[source.start : source.stop]),
                np.asarray(categoricals[source.start : source.stop]),
            )

            feature_ids[start:stop] = batch_ids - field_offsets
            if batch_values is not None:
                feature_values[start:stop] = batch_values
            encoded_labels[start:stop] = labels[source.start : source.stop]
            progress_bar.update(stop - start)

    for output in (feature_ids, feature_values, encoded_labels):
        if output is not None:
            output.flush()
    (path / "complete").touch()


def _prepare_encoded_split(
    corpus: CriteoCorpus,
    preprocessor: FittedPreprocessor,
    field_offsets: np.ndarray,
    rows: range,
    path: Path,
    *,
    description: str,
    chunk_size: int,
    progress: bool,
    progress_file: TextIO | None,
) -> Path:
    row_count = len(rows)

    def validity(candidate: Path) -> bool:
        return _valid_encoded_split(
            candidate,
            row_count=row_count,
            kind=preprocessor.kind,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(path.with_name(f".{path.name}.lock")):
        if validity(path):
            if progress:
                tqdm.write(
                    f"{description}: {row_count:,} rows (cached)",
                    file=sys.stderr if progress_file is None else progress_file,
                )
            return path

        if path.exists():
            shutil.rmtree(path)

        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            _write_encoded_split(
                corpus,
                preprocessor,
                field_offsets,
                rows,
                temporary,
                description=description,
                chunk_size=chunk_size,
                progress=progress,
                progress_file=progress_file,
            )
            temporary.replace(path)
            return path
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def prepare_encoded_data(
    corpus: CriteoCorpus,
    preprocessor_path: Path,
    *,
    chunk_size: int = ENCODED_CHUNK_SIZE,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> EncodedData:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")

    preprocessor = load_preprocessor(preprocessor_path)
    if preprocessor.num_features > np.iinfo(np.int32).max:
        raise ValueError("encoded feature ids do not fit in int32")
    field_offsets = preprocessor.field_offsets
    field_sizes = np.diff(np.append(field_offsets, preprocessor.num_features))
    if np.any(field_sizes > np.iinfo(FEATURE_ID_DTYPE).max + 1):
        raise ValueError("per-field feature ids do not fit in uint16")

    with zstd.open(preprocessor_path, "rb") as file:
        digest = file_digest(file, "sha256").hexdigest()[:16]
    cache_key = preprocessor_path.name.removesuffix(".pkl.zstd")
    root = (
        corpus.cache_dir / f"encoded-v{ENCODED_CACHE_VERSION}" / f"{cache_key}-{digest}"
    )

    def prepare(name: str, rows: range) -> Path:
        return _prepare_encoded_split(
            corpus,
            preprocessor,
            field_offsets,
            rows,
            root / f"{name}_n{len(rows)}",
            description=f"Encoding {preprocessor.kind} {name}",
            chunk_size=chunk_size,
            progress=progress,
            progress_file=progress_file,
        )

    train = prepare("train", range(corpus.train_stop))
    holdout = prepare("holdout", range(corpus.train_stop, corpus.rows))

    return EncodedData(
        num_features=preprocessor.num_features,
        field_offsets=tuple(map(int, field_offsets)),
        train=train,
        holdout=holdout,
        validation_rows=corpus.val_stop - corpus.train_stop,
    )


@dataclass
class CriteoTask:
    data: EncodedData
    order: ShuffledEpochs
    batch_size: int
    _train_arrays: EncodedArrays = dataclass_field(init=False, repr=False)
    _holdout_arrays: EncodedArrays = dataclass_field(init=False, repr=False)
    _field_offsets: np.ndarray = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._train_arrays = load_encoded(self.data.train)
        self._holdout_arrays = load_encoded(self.data.holdout)
        self._field_offsets = np.asarray(self.data.field_offsets, dtype=np.int32)
        if self.order.size != len(self._train_arrays[0]):
            raise ValueError("shuffle size must match the encoded training split")
        if not 0 < self.data.validation_rows < len(self._holdout_arrays[0]):
            raise ValueError("validation and test data must not be empty")

    def train_batches(self, max_examples: int) -> Iterator[BinaryBatch]:
        for rows in self.order.batches(max_examples, self.batch_size):
            yield self._batch(self._train_arrays, rows)

    def val_batches(self) -> Iterator[BinaryBatch]:
        yield from self._batches(self._holdout_arrays, 0, self.data.validation_rows)

    def test_batches(self) -> Iterator[BinaryBatch]:
        yield from self._batches(
            self._holdout_arrays,
            self.data.validation_rows,
            len(self._holdout_arrays[0]),
        )

    def _batches(
        self,
        arrays: EncodedArrays,
        start: int,
        stop: int,
    ) -> Iterator[BinaryBatch]:
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            yield self._batch(arrays, slice(batch_start, batch_stop))

    def _batch(
        self,
        arrays: EncodedArrays,
        rows: slice | np.ndarray,
    ) -> BinaryBatch:
        feature_ids, feature_values, labels = arrays
        batch_values = (
            None if feature_values is None else np.asarray(feature_values[rows])
        )
        batch_ids_array = np.asarray(feature_ids[rows], dtype=np.int32)
        batch_ids_array += self._field_offsets
        batch_ids = torch.from_numpy(batch_ids_array)
        model_inputs = (
            (batch_ids,)
            if batch_values is None
            else (batch_ids, torch.from_numpy(batch_values))
        )
        return model_inputs, torch.from_numpy(
            np.asarray(labels[rows], dtype=np.float32)
        )
