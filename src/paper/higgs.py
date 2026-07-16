import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from operator import index
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


NUM_FEATURES = 28
CACHE_VERSION = 1
FEATURES_FILE = "features.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"

type BinaryBatch = tuple[tuple[torch.Tensor, ...], torch.Tensor]


@dataclass(frozen=True)
class HiggsLayout:
    rows: int
    train_stop: int
    val_stop: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in asdict(self).values()):
            raise TypeError("HIGGS layout boundaries must be integers")
        if not 0 < self.train_stop < self.val_stop < self.rows:
            raise ValueError(
                "expected 0 < train_stop < val_stop < rows; "
                f"got {self.train_stop}, {self.val_stop}, {self.rows}"
            )
        if self.train_stop > 2**32:
            raise ValueError("training row indices must fit in uint32")


OFFICIAL_LAYOUT = HiggsLayout(11_000_000, 10_000_000, 10_500_000)


def _temporary_path(directory: Path, filename: str) -> Path:
    return directory / f".{filename}.{uuid4().hex}.tmp"


def _close_memmap(array: np.memmap | None, *, flush: bool = False) -> None:
    if array is None or array._mmap.closed:
        return
    if flush:
        array.flush()
    # NumPy exposes no public close method for memmap objects.
    array._mmap.close()


def _combine_moments(
    count: int,
    mean: np.ndarray,
    squared_deviations: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    batch_count = len(values)
    if batch_count == 0:
        return count, mean, squared_deviations

    batch_mean = values.mean(axis=0)
    centered = values - batch_mean
    batch_squared_deviations = np.einsum("ij,ij->j", centered, centered)
    if count == 0:
        return batch_count, batch_mean, batch_squared_deviations

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * (batch_count / total)
    squared_deviations = (
        squared_deviations
        + batch_squared_deviations
        + np.square(delta) * (count * batch_count / total)
    )
    return total, mean, squared_deviations


def prepare_corpus(
    raw_path: Path,
    cache_dir: Path,
    *,
    layout: HiggsLayout = OFFICIAL_LAYOUT,
    chunk_size: int = 250_000,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> "HiggsCorpus":
    """Convert the HIGGS CSV into validated memory-mapped arrays."""
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir)
    try:
        chunk_size = index(chunk_size)
    except TypeError as error:
        raise TypeError("chunk_size must be an integer") from error
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    source_size = raw_path.stat().st_size
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / METADATA_FILE
    if metadata_path.exists():
        try:
            corpus = HiggsCorpus.open(
                cache_dir,
                layout=layout,
                source_size=source_size,
            )
        except (KeyError, OSError, TypeError, ValueError):
            metadata_path.unlink(missing_ok=True)
        else:
            if progress:
                tqdm.write(
                    f"HIGGS corpus: {corpus.rows:,} rows (cached)",
                    file=sys.stderr if progress_file is None else progress_file,
                )
            return corpus

    feature_tmp = _temporary_path(cache_dir, FEATURES_FILE)
    label_tmp = _temporary_path(cache_dir, LABELS_FILE)
    metadata_tmp = _temporary_path(cache_dir, METADATA_FILE)
    temporary_paths = (feature_tmp, label_tmp, metadata_tmp)

    features: np.memmap | None = None
    labels: np.memmap | None = None
    rows = 0
    count = 0
    mean = np.zeros(NUM_FEATURES, dtype=np.float64)
    squared_deviations = np.zeros(NUM_FEATURES, dtype=np.float64)

    try:
        features = np.memmap(
            feature_tmp,
            mode="w+",
            dtype=np.float32,
            shape=(layout.rows, NUM_FEATURES),
        )
        labels = np.memmap(
            label_tmp,
            mode="w+",
            dtype=np.uint8,
            shape=(layout.rows,),
        )
        chunks = pd.read_csv(
            raw_path,
            header=None,
            dtype=np.float32,
            chunksize=chunk_size,
        )
        progress_bar = tqdm(
            total=layout.rows,
            desc="Preparing HIGGS corpus",
            unit="rows",
            unit_scale=True,
            disable=not progress,
            file=progress_file,
        )
        with progress_bar:
            for chunk in chunks:
                if chunk.shape[1] != NUM_FEATURES + 1:
                    raise ValueError(
                        f"expected {NUM_FEATURES + 1} columns; got {chunk.shape[1]}"
                    )
                stop = rows + len(chunk)
                if stop > layout.rows:
                    raise ValueError(
                        f"expected {layout.rows:,} rows; input has more"
                    )

                values = chunk.to_numpy(dtype=np.float32, copy=False)
                batch_labels = values[:, 0]
                batch_features = values[:, 1:]
                if not np.isfinite(batch_features).all():
                    raise ValueError("HIGGS features must all be finite")
                if not ((batch_labels == 0.0) | (batch_labels == 1.0)).all():
                    raise ValueError("HIGGS labels must be binary")

                features[rows:stop] = batch_features
                labels[rows:stop] = batch_labels.astype(np.uint8, copy=False)
                training_rows = max(0, min(stop, layout.train_stop) - rows)
                if training_rows > 0:
                    count, mean, squared_deviations = _combine_moments(
                        count,
                        mean,
                        squared_deviations,
                        batch_features[:training_rows],
                    )
                rows = stop
                progress_bar.update(len(chunk))

        if rows != layout.rows:
            raise ValueError(f"expected {layout.rows:,} rows; got {rows:,}")
        if raw_path.stat().st_size != source_size:
            raise ValueError("HIGGS source changed while it was being read")

        variance = np.maximum(squared_deviations / count, 0.0)
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0
        _close_memmap(features, flush=True)
        _close_memmap(labels, flush=True)
        features = None
        labels = None

        feature_tmp.replace(cache_dir / FEATURES_FILE)
        label_tmp.replace(cache_dir / LABELS_FILE)
        metadata = {
            "version": CACHE_VERSION,
            "layout": asdict(layout),
            "num_features": NUM_FEATURES,
            "feature_dtype": np.dtype(np.float32).name,
            "label_dtype": np.dtype(np.uint8).name,
            "source_size": source_size,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
        }
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True))
        metadata_tmp.replace(metadata_path)
    except BaseException:
        _close_memmap(features)
        _close_memmap(labels)
        features = None
        labels = None
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    return HiggsCorpus.open(
        cache_dir,
        layout=layout,
        source_size=source_size,
    )


@dataclass(frozen=True)
class HiggsCorpus:
    cache_dir: Path
    layout: HiggsLayout
    source_size: int
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]

    @classmethod
    def open(
        cls,
        cache_dir: Path,
        *,
        layout: HiggsLayout | None = None,
        source_size: int | None = None,
    ) -> "HiggsCorpus":
        cache_dir = Path(cache_dir)
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        if not isinstance(metadata, dict):
            raise ValueError("HIGGS cache metadata must be a JSON object")
        version = metadata.get("version")
        if version != CACHE_VERSION:
            raise ValueError(
                f"{cache_dir} uses HIGGS cache version {version}; "
                f"expected {CACHE_VERSION}"
            )

        cached_layout = HiggsLayout(**metadata["layout"])
        if layout is not None and cached_layout != layout:
            raise ValueError(
                f"cached HIGGS layout {cached_layout} does not match {layout}"
            )
        if metadata.get("num_features") != NUM_FEATURES:
            raise ValueError(f"expected {NUM_FEATURES} cached HIGGS features")
        if metadata.get("feature_dtype") != np.dtype(np.float32).name:
            raise ValueError("cached HIGGS features must use float32")
        if metadata.get("label_dtype") != np.dtype(np.uint8).name:
            raise ValueError("cached HIGGS labels must use uint8")

        cached_source_size = metadata["source_size"]
        if type(cached_source_size) is not int or cached_source_size < 0:
            raise ValueError("invalid cached HIGGS source size")
        if source_size is not None and cached_source_size != source_size:
            raise ValueError("cached HIGGS source size does not match the input")

        expected_sizes = {
            FEATURES_FILE: cached_layout.rows
            * NUM_FEATURES
            * np.dtype(np.float32).itemsize,
            LABELS_FILE: cached_layout.rows * np.dtype(np.uint8).itemsize,
        }
        for filename, expected in expected_sizes.items():
            actual = (cache_dir / filename).stat().st_size
            if actual != expected:
                raise ValueError(
                    f"expected {filename} to contain {expected} bytes; got {actual}"
                )

        feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(metadata["feature_scale"], dtype=np.float64)
        if feature_mean.shape != (NUM_FEATURES,) or not np.isfinite(
            feature_mean
        ).all():
            raise ValueError("invalid cached HIGGS feature means")
        if (
            feature_scale.shape != (NUM_FEATURES,)
            or not np.isfinite(feature_scale).all()
            or np.any(feature_scale <= 0.0)
        ):
            raise ValueError("invalid cached HIGGS feature scales")

        return cls(
            cache_dir=cache_dir,
            layout=cached_layout,
            source_size=cached_source_size,
            feature_mean=tuple(map(float, feature_mean)),
            feature_scale=tuple(map(float, feature_scale)),
        )

    @property
    def rows(self) -> int:
        return self.layout.rows

    @property
    def train_stop(self) -> int:
        return self.layout.train_stop

    @property
    def val_stop(self) -> int:
        return self.layout.val_stop

    def features(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / FEATURES_FILE,
            mode="r",
            dtype=np.float32,
            shape=(self.rows, NUM_FEATURES),
        )

    def labels(self) -> np.memmap:
        return np.memmap(
            self.cache_dir / LABELS_FILE,
            mode="r",
            dtype=np.uint8,
            shape=(self.rows,),
        )

    def order_path(self, seed: int) -> Path:
        try:
            seed = index(seed)
        except TypeError as error:
            raise TypeError("data seed must be an integer") from error
        if seed < 0:
            raise ValueError(f"data seed must be nonnegative; got {seed}")

        path = self.cache_dir / f"train_order_seed{seed}.npy"
        if path.exists() and self._valid_order(path):
            return path

        order = np.arange(self.train_stop, dtype=np.uint32)
        np.random.default_rng(seed).shuffle(order)
        temporary = _temporary_path(self.cache_dir, path.name)
        try:
            with temporary.open("wb") as file:
                np.save(file, order, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _valid_order(self, path: Path) -> bool:
        try:
            order = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return False
        return order.dtype == np.uint32 and order.shape == (self.train_stop,)


@dataclass
class HiggsTask:
    corpus: HiggsCorpus
    data_seed: int
    batch_size: int
    _order_path: Path = field(init=False, repr=False)
    _mean: np.ndarray = field(init=False, repr=False)
    _scale: np.ndarray = field(init=False, repr=False)
    _features: np.memmap | None = field(init=False, default=None, repr=False)
    _labels: np.memmap | None = field(init=False, default=None, repr=False)
    _order: np.memmap | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self.batch_size = index(self.batch_size)
        except TypeError as error:
            raise TypeError("batch_size must be an integer") from error
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {self.batch_size}")
        self._order_path = self.corpus.order_path(self.data_seed)
        self._mean = np.asarray(self.corpus.feature_mean, dtype=np.float32)
        self._scale = np.asarray(self.corpus.feature_scale, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = vars(self).copy()
        state.update(_features=None, _labels=None, _order=None)
        return state

    def train_batches(self, start: int, stop: int) -> Iterator[BinaryBatch]:
        if not 0 <= start <= stop <= self.corpus.train_stop:
            raise ValueError(
                "expected 0 <= start <= stop <= train_stop; "
                f"got {start}, {stop}, {self.corpus.train_stop}"
            )
        features, labels, order = self._arrays()
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            rows = np.array(order[batch_start:batch_stop], copy=True)
            rows.sort()
            yield self._batch(features, labels, rows)

    def val_batches(self) -> Iterator[BinaryBatch]:
        yield from self._sequential_batches(
            self.corpus.train_stop,
            self.corpus.val_stop,
        )

    def test_batches(self) -> Iterator[BinaryBatch]:
        yield from self._sequential_batches(
            self.corpus.val_stop,
            self.corpus.rows,
        )

    def _arrays(self) -> tuple[np.memmap, np.memmap, np.memmap]:
        if self._features is None:
            self._features = self.corpus.features()
            self._labels = self.corpus.labels()
            self._order = np.load(
                self._order_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if (
                self._order.dtype != np.uint32
                or self._order.shape != (self.corpus.train_stop,)
            ):
                raise ValueError(f"invalid HIGGS training order: {self._order_path}")
        assert (
            self._features is not None
            and self._labels is not None
            and self._order is not None
        )
        return self._features, self._labels, self._order

    def _sequential_batches(self, start: int, stop: int) -> Iterator[BinaryBatch]:
        features, labels, _ = self._arrays()
        for batch_start in range(start, stop, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, stop)
            yield self._batch(features, labels, slice(batch_start, batch_stop))

    def _batch(
        self,
        features: np.memmap,
        labels: np.memmap,
        rows: slice | np.ndarray,
    ) -> BinaryBatch:
        batch_features = np.array(features[rows], dtype=np.float32, copy=True)
        batch_features -= self._mean
        batch_features /= self._scale
        batch_labels = np.array(labels[rows], dtype=np.float32, copy=True)
        return ((torch.from_numpy(batch_features),), torch.from_numpy(batch_labels))
