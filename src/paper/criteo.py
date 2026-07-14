import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TextIO

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
NUMERICS_FILE = "numerics.dat"
CATEGORICALS_FILE = "categoricals.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"
MISSING_NUMERIC = np.iinfo(np.int32).min

type PreprocessingKind = Literal["bucket", "hybrid"]


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
    return (_mix32(values) % (buckets - 2) + offset + 2).astype(np.int64)


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


class CriteoPreprocessor(Protocol):
    kind: PreprocessingKind

    @property
    def num_features(self) -> int: ...

    def encode(
        self,
        numerics: np.ndarray,
        categoricals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


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
    ) -> tuple[np.ndarray, np.ndarray]:
        _check_inputs(numerics, categoricals)
        tokens = np.concatenate((_bucket_numeric(numerics), categoricals), axis=-1)
        feature_ids = np.empty(tokens.shape, dtype=np.int64)

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

        return feature_ids, np.ones(feature_ids.shape, dtype=np.float32)


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
    ) -> tuple[np.ndarray, np.ndarray]:
        _check_inputs(numerics, categoricals)
        shape = numerics.shape[:-1] + (NUM_FIELDS,)
        feature_ids = np.empty(shape, dtype=np.int64)
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


TensorBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class CriteoTask:
    corpus: CriteoCorpus
    preprocessor: CriteoPreprocessor
    train_rows: np.ndarray
    batch_size: int
    _numerics: np.memmap = field(init=False, repr=False)
    _categoricals: np.memmap = field(init=False, repr=False)
    _labels: np.memmap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._numerics = self.corpus.numerics()
        self._categoricals = self.corpus.categoricals()
        self._labels = self.corpus.labels()

    def train_batches(self, start: int, stop: int) -> Iterator[TensorBatch]:
        if not 0 <= start <= stop <= len(self.train_rows):
            raise ValueError(
                f"expected 0 <= start <= stop <= {len(self.train_rows)}; "
                f"got start={start}, stop={stop}"
            )
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            rows = np.sort(self.train_rows[batch_start:batch_stop])
            yield self._batch(rows)

    def val_batches(self) -> Iterator[TensorBatch]:
        yield from self._sequential_batches(
            self.corpus.train_stop, self.corpus.val_stop
        )

    def test_batches(self) -> Iterator[TensorBatch]:
        yield from self._sequential_batches(self.corpus.val_stop, self.corpus.rows)

    def _sequential_batches(self, start: int, stop: int) -> Iterator[TensorBatch]:
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            yield self._batch(slice(batch_start, batch_stop))

    def _batch(self, rows: np.ndarray | slice) -> TensorBatch:
        feature_ids, feature_values = self.preprocessor.encode(
            np.asarray(self._numerics[rows]),
            np.asarray(self._categoricals[rows]),
        )
        labels = np.asarray(self._labels[rows], dtype=np.float32)
        return (
            torch.from_numpy(feature_ids),
            torch.from_numpy(feature_values),
            torch.from_numpy(labels),
        )
