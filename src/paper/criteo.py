import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

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

TOKENS_FILE = "tokens.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"


def _bucket_numeric(values: np.ndarray) -> np.ndarray:
    """Winner-style categorical buckets for Criteo's integer fields."""
    values = np.asarray(values, dtype=np.float64)
    present = ~np.isnan(values)
    small = present & (values <= 2)
    large = present & ~small

    tokens = np.zeros(values.shape, dtype=np.uint32)
    integers = values[small].astype(np.int64)
    zigzag = (integers << 1) ^ (integers >> 63)
    tokens[small] = zigzag.astype(np.uint32) + 1
    tokens[large] = (
        np.floor(np.square(np.log(values[large]))).astype(np.uint32)
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

    token_tmp = cache_dir / f".{TOKENS_FILE}.tmp"
    label_tmp = cache_dir / f".{LABELS_FILE}.tmp"
    rows = 0
    dtypes = {LABEL: np.uint8, **dict.fromkeys(CATEGORICAL_COLUMNS, "string")}

    with token_tmp.open("wb") as token_file, label_tmp.open("wb") as label_file:
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
                tokens = np.empty((len(chunk), NUM_FIELDS), dtype=np.uint32)
                numeric = chunk.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
                tokens[:, :NUM_NUMERIC_FIELDS] = _bucket_numeric(numeric)
                for index, column in enumerate(
                    CATEGORICAL_COLUMNS, NUM_NUMERIC_FIELDS
                ):
                    tokens[:, index] = _parse_hex(chunk[column])

                tokens.tofile(token_file)
                chunk[LABEL].to_numpy(dtype=np.uint8).tofile(label_file)
                rows += len(chunk)
                progress_bar.update(len(chunk))

    os.replace(token_tmp, cache_dir / TOKENS_FILE)
    os.replace(label_tmp, cache_dir / LABELS_FILE)
    metadata_tmp = cache_dir / f".{METADATA_FILE}.tmp"
    metadata_tmp.write_text(json.dumps({"rows": rows, "fields": NUM_FIELDS}))
    os.replace(metadata_tmp, metadata_path)
    return CriteoCorpus.open(cache_dir)


@dataclass(frozen=True)
class CriteoCorpus:
    cache_dir: Path
    rows: int

    @classmethod
    def open(cls, cache_dir: Path) -> "CriteoCorpus":
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        return cls(cache_dir=cache_dir, rows=metadata["rows"])

    @property
    def train_stop(self) -> int:
        return self.rows * 8 // 10

    @property
    def val_stop(self) -> int:
        return self.rows * 9 // 10

    def tokens(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / TOKENS_FILE,
            mode="r",
            dtype=np.uint32,
            shape=(self.rows, NUM_FIELDS),
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


@dataclass(frozen=True)
class CriteoPreprocessor:
    buckets_per_field: int
    frequent_categories: tuple[np.ndarray, ...]

    @property
    def num_features(self) -> int:
        return NUM_FIELDS * self.buckets_per_field

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.shape[-1:] != (NUM_FIELDS,):
            raise ValueError(
                f"expected token shape (..., {NUM_FIELDS}); got {tokens.shape}"
            )

        encoded = np.empty(tokens.shape, dtype=np.int64)
        for field in range(NUM_FIELDS):
            values = tokens[..., field]
            offset = field * self.buckets_per_field
            encoded[..., field] = (
                _mix32(values) % (self.buckets_per_field - 2) + offset + 2
            )
            encoded[..., field][values == 0] = offset

            if field >= NUM_NUMERIC_FIELDS:
                frequent = self.frequent_categories[field - NUM_NUMERIC_FIELDS]
                positions = np.searchsorted(frequent, values)
                known = np.zeros(values.shape, dtype=bool)
                valid = positions < len(frequent)
                known[valid] = frequent[positions[valid]] == values[valid]
                encoded[..., field][(values != 0) & ~known] = offset + 1
        return encoded

    def save(self, path: Path) -> None:
        arrays = {
            f"field_{field}": values
            for field, values in enumerate(
                self.frequent_categories, NUM_NUMERIC_FIELDS
            )
        }
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("wb") as file:
            np.savez(file, buckets_per_field=self.buckets_per_field, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CriteoPreprocessor":
        with np.load(path) as data:
            return cls(
                buckets_per_field=int(data["buckets_per_field"]),
                frequent_categories=tuple(
                    data[f"field_{field}"]
                    for field in range(NUM_NUMERIC_FIELDS, NUM_FIELDS)
                ),
            )


def fit_preprocessor(
    corpus: CriteoCorpus,
    *,
    sample_size: int,
    sample_seed: int,
    min_count: int,
    buckets_per_field: int,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> Path:
    if not 0 < sample_size <= corpus.train_stop:
        raise ValueError(
            f"sample_size must be in [1, {corpus.train_stop}]; got {sample_size}"
        )
    if buckets_per_field < 3:
        raise ValueError(
            "buckets_per_field must leave room for missing and rare values"
        )

    path = corpus.cache_dir / (
        f"preprocessor_sample{sample_size}_seed{sample_seed}_"
        f"min{min_count}_b{buckets_per_field}.npz"
    )
    if path.exists():
        if progress:
            tqdm.write(
                f"Preprocessor sample={sample_size:,}, seed={sample_seed} (cached)",
                file=sys.stderr if progress_file is None else progress_file,
            )
        return path

    rows = np.random.default_rng(sample_seed).choice(
        corpus.train_stop,
        size=sample_size,
        replace=False,
        shuffle=False,
    )
    rows.sort()
    tokens = corpus.tokens()
    frequent = []
    fields = tqdm(
        range(NUM_NUMERIC_FIELDS, NUM_FIELDS),
        desc=f"Fitting preprocessor on {sample_size:,} rows",
        unit="field",
        disable=not progress,
        file=progress_file,
    )
    for field in fields:
        values, counts = np.unique(tokens[rows, field], return_counts=True)
        frequent.append(values[(values != 0) & (counts >= min_count)])

    CriteoPreprocessor(buckets_per_field, tuple(frequent)).save(path)
    return path


TensorBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class CriteoTask:
    corpus: CriteoCorpus
    preprocessor: CriteoPreprocessor
    train_rows: np.ndarray
    batch_size: int
    _tokens: np.memmap = field(init=False, repr=False)
    _labels: np.memmap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = self.corpus.tokens()
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
        feature_ids = self.preprocessor.encode(np.asarray(self._tokens[rows]))
        labels = np.asarray(self._labels[rows], dtype=np.float32)
        return torch.from_numpy(feature_ids), torch.from_numpy(labels)
