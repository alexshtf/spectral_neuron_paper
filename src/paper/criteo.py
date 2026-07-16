import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


NUM_NUMERIC_FIELDS = 13
NUM_CATEGORICAL_FIELDS = 26
NUM_FIELDS = NUM_NUMERIC_FIELDS + NUM_CATEGORICAL_FIELDS

LABEL = "label"
NUMERIC_COLUMNS = tuple(f"I{i}" for i in range(1, NUM_NUMERIC_FIELDS + 1))
CATEGORICAL_COLUMNS = tuple(
    f"C{i}" for i in range(1, NUM_CATEGORICAL_FIELDS + 1)
)
COLUMNS = (LABEL, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)

CACHE_VERSION = 2
# Bump when fitted preprocessing or the encoded representation changes.
ENCODED_CACHE_VERSION = 1
ENCODED_CHUNK_SIZE = 2**18
NUMERICS_FILE = "numerics.dat"
CATEGORICALS_FILE = "categoricals.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"
MISSING_NUMERIC = np.iinfo(np.int32).min

type PreprocessingKind = Literal["bucket", "hybrid"]
type FeatureArrays = tuple[np.ndarray, np.ndarray | None]


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

    tokens = np.zeros(values.shape, dtype=np.uint32)
    integers = numeric[small].astype(np.int64)
    zigzag = (integers << 1) ^ (integers >> 63)
    tokens[small] = zigzag.astype(np.uint32) + 1
    tokens[large] = (
        np.floor(np.square(np.log(numeric[large]))).astype(np.uint32)
        | np.uint32(1 << 31)
    )
    return tokens


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
    if metadata_path.exists():
        corpus = CriteoCorpus.open(cache_dir)
        if progress:
            tqdm.write(
                f"Criteo corpus: {corpus.rows:,} rows (cached)",
                file=sys.stderr if progress_file is None else progress_file,
            )
        return corpus

    numeric_tmp = cache_dir / f".{NUMERICS_FILE}.tmp"
    categorical_tmp = cache_dir / f".{CATEGORICALS_FILE}.tmp"
    label_tmp = cache_dir / f".{LABELS_FILE}.tmp"
    rows = 0
    dtypes = {LABEL: np.uint8, **dict.fromkeys(CATEGORICAL_COLUMNS, "string")}

    with (
        numeric_tmp.open("wb") as numeric_file,
        categorical_tmp.open("wb") as categorical_file,
        label_tmp.open("wb") as label_file,
    ):
        chunks = pd.read_csv(
            raw_path,
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

    os.replace(numeric_tmp, cache_dir / NUMERICS_FILE)
    os.replace(categorical_tmp, cache_dir / CATEGORICALS_FILE)
    os.replace(label_tmp, cache_dir / LABELS_FILE)
    metadata_tmp = cache_dir / f".{METADATA_FILE}.tmp"
    metadata_tmp.write_text(json.dumps({"version": CACHE_VERSION, "rows": rows}))
    os.replace(metadata_tmp, metadata_path)
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

    def numerics(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / NUMERICS_FILE,
            mode="r",
            dtype=np.int32,
            shape=(self.rows, NUM_NUMERIC_FIELDS),
        )

    def categoricals(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / CATEGORICALS_FILE,
            mode="r",
            dtype=np.uint32,
            shape=(self.rows, NUM_CATEGORICAL_FIELDS),
        )

    def labels(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / LABELS_FILE,
            mode="r",
            dtype=np.uint8,
            shape=(self.rows,),
        )

    def order_path(self, seed: int) -> Path:
        path = self.cache_dir / f"train_order_seed{seed}.npy"
        if path.exists():
            return path

        order = np.random.default_rng(seed).permutation(self.train_stop)
        dtype = np.uint32 if self.train_stop < 2**32 else np.uint64
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("wb") as file:
            np.save(file, order.astype(dtype, copy=False))
        os.replace(tmp, path)
        return path


def _mix32(values: np.ndarray) -> np.ndarray:
    mixed = values.astype(np.uint64)
    mixed ^= mixed >> 16
    mixed *= np.uint64(0x7FEB352D)
    mixed ^= mixed >> 15
    mixed *= np.uint64(0x846CA68B)
    mixed ^= mixed >> 16
    return mixed


def _hashed_ids(values: np.ndarray, offset: int, buckets: int) -> np.ndarray:
    return (_mix32(values) % (buckets - 2) + offset + 2).astype(np.int32)


def _categorical_ids(
    values: np.ndarray,
    frequent: np.ndarray,
    *,
    offset: int,
    buckets: int,
) -> np.ndarray:
    encoded = _hashed_ids(values, offset, buckets)
    encoded[values == 0] = offset

    positions = np.searchsorted(frequent, values)
    known = np.zeros(values.shape, dtype=bool)
    valid = positions < len(frequent)
    known[valid] = frequent[positions[valid]] == values[valid]
    encoded[(values != 0) & ~known] = offset + 1
    return encoded


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
    buckets_per_field: int
    frequent_categories: tuple[np.ndarray, ...]
    kind: PreprocessingKind = field(default="bucket", init=False)

    @property
    def num_features(self) -> int:
        return NUM_FIELDS * self.buckets_per_field

    def encode(
        self,
        numerics: np.ndarray,
        categoricals: np.ndarray,
    ) -> FeatureArrays:
        _check_inputs(numerics, categoricals)
        tokens = np.concatenate((_bucket_numeric(numerics), categoricals), axis=-1)
        feature_ids = np.empty(tokens.shape, dtype=np.int32)

        for field in range(NUM_NUMERIC_FIELDS):
            values = tokens[..., field]
            offset = field * self.buckets_per_field
            feature_ids[..., field] = _hashed_ids(
                values, offset, self.buckets_per_field
            )
            feature_ids[..., field][values == 0] = offset

        for category, frequent in enumerate(self.frequent_categories):
            field = NUM_NUMERIC_FIELDS + category
            feature_ids[..., field] = _categorical_ids(
                tokens[..., field],
                frequent,
                offset=field * self.buckets_per_field,
                buckets=self.buckets_per_field,
            )

        return feature_ids, None


@dataclass(frozen=True)
class HybridPreprocessor:
    buckets_per_categorical_field: int
    frequent_categories: tuple[np.ndarray, ...]
    negative_values: tuple[np.ndarray, ...]
    positive_mean: np.ndarray
    positive_scale: np.ndarray
    kind: PreprocessingKind = field(default="hybrid", init=False)

    @property
    def numeric_offsets(self) -> np.ndarray:
        sizes = np.fromiter(
            (len(values) + 4 for values in self.negative_values),
            dtype=np.int64,
            count=NUM_NUMERIC_FIELDS,
        )
        return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes[:-1])))

    @property
    def num_numeric_features(self) -> int:
        return sum(len(values) + 4 for values in self.negative_values)

    @property
    def num_features(self) -> int:
        return (
            self.num_numeric_features
            + NUM_CATEGORICAL_FIELDS * self.buckets_per_categorical_field
        )

    def encode(
        self,
        numerics: np.ndarray,
        categoricals: np.ndarray,
    ) -> FeatureArrays:
        _check_inputs(numerics, categoricals)
        shape = numerics.shape[:-1] + (NUM_FIELDS,)
        feature_ids = np.empty(shape, dtype=np.int32)
        feature_values = np.ones(shape, dtype=np.float32)

        for field, (offset, negatives) in enumerate(
            zip(self.numeric_offsets, self.negative_values)
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

        categorical_offset = self.num_numeric_features
        for category, frequent in enumerate(self.frequent_categories):
            field = NUM_NUMERIC_FIELDS + category
            feature_ids[..., field] = _categorical_ids(
                categoricals[..., category],
                frequent,
                offset=(
                    categorical_offset
                    + category * self.buckets_per_categorical_field
                ),
                buckets=self.buckets_per_categorical_field,
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
    buckets_per_field: int,
) -> Path:
    return corpus.cache_dir / (
        f"preprocessor_{kind}_sample{sample_size}_seed{sample_seed}_"
        f"min{min_count}_b{buckets_per_field}.npz"
    )


def _save_preprocessor(preprocessor: FittedPreprocessor, path: Path) -> None:
    arrays = {
        "kind": np.asarray(preprocessor.kind),
        "buckets_per_field": np.asarray(
            preprocessor.buckets_per_field
            if isinstance(preprocessor, BucketPreprocessor)
            else preprocessor.buckets_per_categorical_field
        ),
        **{
            f"category_{field}": values
            for field, values in enumerate(preprocessor.frequent_categories)
        },
    }
    if isinstance(preprocessor, HybridPreprocessor):
        arrays |= {
            "positive_mean": preprocessor.positive_mean,
            "positive_scale": preprocessor.positive_scale,
            **{
                f"negative_{field}": values
                for field, values in enumerate(preprocessor.negative_values)
            },
        }

    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(tmp, path)


def load_preprocessor(path: Path) -> FittedPreprocessor:
    with np.load(path) as data:
        kind = data["kind"].item()
        buckets = int(data["buckets_per_field"])
        frequent = tuple(
            data[f"category_{field}"] for field in range(NUM_CATEGORICAL_FIELDS)
        )
        if kind == "bucket":
            return BucketPreprocessor(buckets, frequent)
        if kind == "hybrid":
            return HybridPreprocessor(
                buckets,
                frequent,
                tuple(
                    data[f"negative_{field}"]
                    for field in range(NUM_NUMERIC_FIELDS)
                ),
                data["positive_mean"],
                data["positive_scale"],
            )
    raise ValueError(f"unknown preprocessor kind {kind!r}")


def fit_preprocessors(
    corpus: CriteoCorpus,
    kinds: Iterable[PreprocessingKind],
    *,
    sample_size: int,
    sample_seed: int,
    min_count: int,
    buckets_per_field: int,
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
    if buckets_per_field < 3:
        raise ValueError(
            "buckets_per_field must leave room for missing and rare values"
        )

    paths = {
        kind: _preprocessor_path(
            corpus,
            kind,
            sample_size=sample_size,
            sample_seed=sample_seed,
            min_count=min_count,
            buckets_per_field=buckets_per_field,
        )
        for kind in kinds
    }
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
    )
    frequent = []
    for field in fields:
        values, counts = np.unique(categoricals[rows, field], return_counts=True)
        frequent.append(values[(values != 0) & (counts >= min_count)])
    frequent_categories = tuple(frequent)

    if "bucket" in missing:
        _save_preprocessor(
            BucketPreprocessor(buckets_per_field, frequent_categories),
            paths["bucket"],
        )

    if "hybrid" in missing:
        numerics = corpus.numerics()
        negative_values = []
        positive_mean = np.empty(NUM_NUMERIC_FIELDS)
        positive_scale = np.empty(NUM_NUMERIC_FIELDS)
        fields = tqdm(
            range(NUM_NUMERIC_FIELDS),
            desc=f"Fitting hybrid numerics on {sample_size:,} rows",
            unit="field",
            disable=not progress,
            file=progress_file,
        )
        for field in fields:
            values = numerics[rows, field]
            negative_values.append(
                np.unique(values[(values < 0) & (values != MISSING_NUMERIC)])
            )
            logged = np.log1p(values[values > 0].astype(np.float64))
            positive_mean[field] = logged.mean() if len(logged) else 0.0
            scale = logged.std() if len(logged) else 1.0
            positive_scale[field] = scale if scale > 0 else 1.0

        _save_preprocessor(
            HybridPreprocessor(
                buckets_per_field,
                frequent_categories,
                tuple(negative_values),
                positive_mean,
                positive_scale,
            ),
            paths["hybrid"],
        )

    return paths


type TensorBatch = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]


@dataclass(frozen=True)
class EncodedSplit:
    path: Path
    rows: int
    has_values: bool

    def arrays(self) -> FeatureArrays:
        feature_ids = np.load(
            self.path / "feature_ids.npy", mmap_mode="c", allow_pickle=False
        )
        feature_values = (
            np.load(
                self.path / "feature_values.npy",
                mmap_mode="c",
                allow_pickle=False,
            )
            if self.has_values
            else None
        )
        return feature_ids, feature_values


@dataclass(frozen=True)
class EncodedData:
    num_features: int
    train: dict[int, EncodedSplit]
    holdout: EncodedSplit


def _source_rows(
    rows: range | np.ndarray,
    start: int,
    stop: int,
) -> tuple[slice | np.ndarray, np.ndarray | None]:
    chunk = rows[start:stop]
    if isinstance(chunk, range):
        return slice(chunk.start, chunk.stop), None

    chunk = np.asarray(chunk)
    sorter = np.argsort(chunk)
    return chunk[sorter], np.argsort(sorter)


def _prepare_encoded_split(
    corpus: CriteoCorpus,
    preprocessor: FittedPreprocessor,
    rows: range | np.ndarray,
    path: Path,
    *,
    description: str,
    chunk_size: int,
    progress: bool,
    progress_file: TextIO | None,
) -> EncodedSplit:
    split = EncodedSplit(path, len(rows), preprocessor.kind == "hybrid")
    if (path / "complete").exists():
        if progress:
            tqdm.write(
                f"{description}: {split.rows:,} rows (cached)",
                file=sys.stderr if progress_file is None else progress_file,
            )
        return split

    path.mkdir(parents=True, exist_ok=True)
    shapes = {
        "feature_ids": (np.int32, (split.rows, NUM_FIELDS)),
    }
    if split.has_values:
        shapes["feature_values"] = (np.float32, (split.rows, NUM_FIELDS))

    token = os.getpid()
    temporary = {
        name: path / f".{name}.{token}.tmp" for name in shapes
    }
    outputs = {
        name: np.lib.format.open_memmap(
            temporary[name], mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in shapes.items()
    }
    numerics = corpus.numerics()
    categoricals = corpus.categoricals()
    progress_bar = tqdm(
        total=split.rows,
        desc=description,
        unit="rows",
        unit_scale=True,
        disable=not progress,
        file=progress_file,
    )
    with progress_bar:
        for start in range(0, split.rows, chunk_size):
            stop = min(start + chunk_size, split.rows)
            source, inverse = _source_rows(rows, start, stop)
            feature_ids, feature_values = preprocessor.encode(
                np.asarray(numerics[source]),
                np.asarray(categoricals[source]),
            )
            if inverse is not None:
                feature_ids = feature_ids[inverse]
                feature_values = (
                    None if feature_values is None else feature_values[inverse]
                )

            outputs["feature_ids"][start:stop] = feature_ids
            if feature_values is not None:
                outputs["feature_values"][start:stop] = feature_values
            progress_bar.update(stop - start)

    for output in outputs.values():
        output.flush()
    del outputs
    for name in shapes:
        os.replace(temporary[name], path / f"{name}.npy")
    (path / "complete").touch()
    return split


def prepare_encoded_data(
    corpus: CriteoCorpus,
    preprocessor_path: Path,
    train_size: int,
    data_seeds: Iterable[int],
    *,
    chunk_size: int = ENCODED_CHUNK_SIZE,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> EncodedData:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    if not 0 < train_size <= corpus.train_stop:
        raise ValueError(
            f"train_size must be in [1, {corpus.train_stop}]; got {train_size}"
        )

    preprocessor = load_preprocessor(preprocessor_path)
    if preprocessor.num_features > np.iinfo(np.int32).max:
        raise ValueError("encoded feature ids do not fit in int32")

    with preprocessor_path.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()[:16]
    root = (
        corpus.cache_dir
        / f"encoded-v{ENCODED_CACHE_VERSION}"
        / f"{preprocessor_path.stem}-{digest}"
    )

    def prepare(name: str, rows: range | np.ndarray) -> EncodedSplit:
        return _prepare_encoded_split(
            corpus,
            preprocessor,
            rows,
            root / f"{name}_n{len(rows)}",
            description=f"Encoding {preprocessor.kind} {name}",
            chunk_size=chunk_size,
            progress=progress,
            progress_file=progress_file,
        )

    holdout = prepare("holdout", range(corpus.train_stop, corpus.rows))
    train = {}
    for seed in sorted(set(data_seeds)):
        order = np.load(corpus.order_path(seed), mmap_mode="r")
        train_rows = order[:train_size]
        order_digest = sha256(train_rows).hexdigest()[:16]
        train[seed] = prepare(
            f"train_seed{seed}_{order_digest}", train_rows
        )

    return EncodedData(preprocessor.num_features, train, holdout)


@dataclass
class CriteoTask:
    corpus: CriteoCorpus
    train: EncodedSplit
    holdout: EncodedSplit
    train_rows: np.ndarray
    batch_size: int
    _train_arrays: FeatureArrays = field(init=False, repr=False)
    _holdout_arrays: FeatureArrays = field(init=False, repr=False)
    _labels: np.memmap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.train_rows) != self.train.rows:
            raise ValueError("training order and encoded cache differ in length")
        self._train_arrays = self.train.arrays()
        self._holdout_arrays = self.holdout.arrays()
        self._labels = self.corpus.labels()

    def train_batches(self, start: int, stop: int) -> Iterator[TensorBatch]:
        if not 0 <= start <= stop <= self.train.rows:
            raise ValueError(
                f"expected 0 <= start <= stop <= {self.train.rows}; "
                f"got start={start}, stop={stop}"
            )
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            rows = slice(batch_start, batch_stop)
            source_rows = np.asarray(self.train_rows[rows])
            order = np.argsort(source_rows)
            yield self._batch(
                self._train_arrays,
                rows,
                source_rows[order],
                order,
            )

    def val_batches(self) -> Iterator[TensorBatch]:
        validation_rows = self.corpus.val_stop - self.corpus.train_stop
        yield from self._sequential_batches(0, validation_rows)

    def test_batches(self) -> Iterator[TensorBatch]:
        validation_rows = self.corpus.val_stop - self.corpus.train_stop
        yield from self._sequential_batches(validation_rows, self.holdout.rows)

    def _sequential_batches(self, start: int, stop: int) -> Iterator[TensorBatch]:
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            rows = slice(batch_start, batch_stop)
            labels = slice(
                self.corpus.train_stop + batch_start,
                self.corpus.train_stop + batch_stop,
            )
            yield self._batch(self._holdout_arrays, rows, labels)

    def _batch(
        self,
        arrays: FeatureArrays,
        rows: slice,
        label_rows: slice | np.ndarray,
        order: np.ndarray | None = None,
    ) -> TensorBatch:
        feature_ids, feature_values = arrays
        batch_ids = np.asarray(feature_ids[rows])
        batch_values = (
            None if feature_values is None else np.asarray(feature_values[rows])
        )
        if order is not None:
            batch_ids = batch_ids[order]
            batch_values = None if batch_values is None else batch_values[order]
        batch_labels = np.asarray(self._labels[label_rows], dtype=np.float32)

        return (
            torch.from_numpy(batch_ids),
            None if batch_values is None else torch.from_numpy(batch_values),
            torch.from_numpy(batch_labels),
        )
