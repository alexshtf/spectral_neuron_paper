import json
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from paper.compression import open_dataset_file
from paper.shuffling import ShuffledEpochs


NUM_FEATURES = 28
CACHE_VERSION = 1
FEATURES_FILE = "features.dat"
LABELS_FILE = "labels.dat"
METADATA_FILE = "metadata.json"

FEATURE_NAMES = (
    "lepton pT",
    "lepton eta",
    "lepton phi",
    "missing energy magnitude",
    "missing energy phi",
    "jet 1 pT",
    "jet 1 eta",
    "jet 1 phi",
    "jet 1 b-tag",
    "jet 2 pT",
    "jet 2 eta",
    "jet 2 phi",
    "jet 2 b-tag",
    "jet 3 pT",
    "jet 3 eta",
    "jet 3 phi",
    "jet 3 b-tag",
    "jet 4 pT",
    "jet 4 eta",
    "jet 4 phi",
    "jet 4 b-tag",
    "m(jj)",
    "m(jjj)",
    "m(lv)",
    "m(jlv)",
    "m(bb)",
    "m(wbb)",
    "m(wwbb)",
)

type BinaryBatch = tuple[tuple[torch.Tensor, ...], torch.Tensor]


def default_cache_dir(raw_path: Path) -> Path:
    raw_path = Path(raw_path)
    return raw_path.with_name(f".{raw_path.name}.cache-v{CACHE_VERSION}")


@dataclass(frozen=True)
class HiggsLayout:
    rows: int
    train_stop: int
    val_stop: int

    def __post_init__(self) -> None:
        if not 0 < self.train_stop < self.val_stop < self.rows:
            raise ValueError(
                "expected 0 < train_stop < val_stop < rows; "
                f"got {self.train_stop}, {self.val_stop}, {self.rows}"
            )
        if self.train_stop > 2**32:
            raise ValueError("training row indices must fit in uint32")


OFFICIAL_LAYOUT = HiggsLayout(11_000_000, 10_000_000, 10_500_000)


def _combine_moments(
    count: int,
    mean: np.ndarray,
    squared_deviations: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    batch_count = len(values)
    batch_mean = values.mean(axis=0)
    centered = values - batch_mean
    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * (batch_count / total)
    squared_deviations = (
        squared_deviations
        + np.einsum("ij,ij->j", centered, centered)
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
    """Convert the CSV to memory-mapped arrays, trusting an existing cache."""
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / METADATA_FILE
    if metadata_path.exists():
        corpus = HiggsCorpus.open(cache_dir, layout=layout)
        if progress:
            tqdm.write(
                f"HIGGS corpus: {corpus.rows:,} rows (cached)",
                file=sys.stderr if progress_file is None else progress_file,
            )
        return corpus

    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=cache_dir) as tmp:
        temporary = Path(tmp)
        feature_path = temporary / FEATURES_FILE
        label_path = temporary / LABELS_FILE
        rows = 0
        count = 0
        mean = np.zeros(NUM_FEATURES, dtype=np.float64)
        squared_deviations = np.zeros(NUM_FEATURES, dtype=np.float64)

        with (
            open_dataset_file(raw_path) as raw_file,
            feature_path.open("wb") as feature_file,
            label_path.open("wb") as label_file,
        ):
            chunks = pd.read_csv(
                raw_file,
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
                dynamic_ncols=True,
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

                    batch_features.tofile(feature_file)
                    batch_labels.astype(np.uint8, copy=False).tofile(label_file)
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

        variance = np.maximum(squared_deviations / count, 0.0)
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0

        feature_path.replace(cache_dir / FEATURES_FILE)
        label_path.replace(cache_dir / LABELS_FILE)
        metadata = {
            "version": CACHE_VERSION,
            "layout": asdict(layout),
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
        }
        metadata_tmp = temporary / METADATA_FILE
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True))
        metadata_tmp.replace(metadata_path)

    return HiggsCorpus.open(cache_dir, layout=layout)


@dataclass(frozen=True)
class HiggsCorpus:
    cache_dir: Path
    layout: HiggsLayout
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]

    @classmethod
    def open(
        cls,
        cache_dir: Path,
        *,
        layout: HiggsLayout | None = None,
    ) -> "HiggsCorpus":
        cache_dir = Path(cache_dir)
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
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

        return cls(
            cache_dir=cache_dir,
            layout=cached_layout,
            feature_mean=tuple(metadata["feature_mean"]),
            feature_scale=tuple(metadata["feature_scale"]),
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

    def shuffled_epochs(self, seed: int) -> ShuffledEpochs:
        return ShuffledEpochs(self.cache_dir, self.train_stop, seed)


@dataclass
class HiggsTask:
    corpus: HiggsCorpus
    data_seed: int
    batch_size: int
    _shuffled_epochs: ShuffledEpochs = field(init=False, repr=False)
    _mean: np.ndarray = field(init=False, repr=False)
    _scale: np.ndarray = field(init=False, repr=False)
    _features: np.memmap | None = field(init=False, default=None, repr=False)
    _labels: np.memmap | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._shuffled_epochs = self.corpus.shuffled_epochs(self.data_seed)
        self._mean = np.asarray(self.corpus.feature_mean, dtype=np.float32)
        self._scale = np.asarray(self.corpus.feature_scale, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = vars(self).copy()
        state.update(_features=None, _labels=None)
        return state

    def train_batches(self, max_examples: int) -> Iterator[BinaryBatch]:
        features, labels = self._arrays()
        for rows in self._shuffled_epochs.batches(max_examples, self.batch_size):
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

    def _arrays(self) -> tuple[np.memmap, np.memmap]:
        if self._features is None:
            self._features = self.corpus.features()
            self._labels = self.corpus.labels()
        assert self._features is not None and self._labels is not None
        return self._features, self._labels

    def _sequential_batches(self, start: int, stop: int) -> Iterator[BinaryBatch]:
        features, labels = self._arrays()
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
