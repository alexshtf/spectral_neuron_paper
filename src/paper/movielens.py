import json
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from operator import index
from pathlib import Path
from typing import Any, BinaryIO, TextIO
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


CACHE_VERSION = 1
NUM_FIELDS = 2
RATING_CENTER = 2.75
FEATURE_IDS_FILE = "feature_ids.npy"
RATINGS_FILE = "ratings.npy"
METADATA_FILE = "metadata.json"

type RatingBatch = tuple[tuple[torch.Tensor, ...], torch.Tensor]


def _temporary_path(directory: Path, filename: str) -> Path:
    return directory / f".{filename}.{uuid4().hex}.tmp"


def _source_size(path: Path) -> int:
    ratings = path / "ratings.csv" if path.is_dir() else path
    if not ratings.is_file():
        raise FileNotFoundError(ratings)
    return ratings.stat().st_size


@contextmanager
def _open_ratings(path: Path) -> Iterator[Path | BinaryIO]:
    if path.is_dir():
        yield path / "ratings.csv"
    elif path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            member = next(
                name
                for name in archive.namelist()
                if Path(name).name == "ratings.csv"
            )
            with archive.open(member) as ratings:
                yield ratings
    else:
        yield path


def _split_by_user(feature_ids: np.ndarray, seed: int) -> np.ndarray:
    """Assign exact per-user 80/10/10 splits in random within-user order."""
    users = feature_ids[:, 0]
    grouped = None
    grouped_users = users
    if not np.all(users[:-1] <= users[1:]):
        grouped = np.argsort(users, kind="stable").astype(np.uint32)
        grouped_users = users[grouped]
    boundaries = np.flatnonzero(grouped_users[1:] != grouped_users[:-1]) + 1
    boundaries = np.concatenate(([0], boundaries, [len(users)]))

    rng = np.random.default_rng(seed)
    split = np.empty(len(users), dtype=np.uint8)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = (
            np.arange(start, stop, dtype=np.uint32)
            if grouped is None
            else np.array(grouped[start:stop], copy=True)
        )
        rng.shuffle(rows)
        train_stop = max(1, len(rows) * 8 // 10)
        val_stop = max(train_stop, len(rows) * 9 // 10)
        split[rows[:train_stop]] = 0
        split[rows[train_stop:val_stop]] = 1
        split[rows[val_stop:]] = 2
    return split


def prepare_corpus(
    raw_path: Path,
    cache_dir: Path,
    *,
    split_seed: int = 0,
    chunk_size: int = 1_000_000,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> "MovieLensCorpus":
    """Encode and split MovieLens ratings into compact memory-mapped arrays."""
    raw_path, cache_dir = Path(raw_path), Path(cache_dir)
    split_seed, chunk_size = index(split_seed), index(chunk_size)
    if split_seed < 0:
        raise ValueError(f"split_seed must be nonnegative; got {split_seed}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")

    source_size = _source_size(raw_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / METADATA_FILE
    if metadata_path.exists():
        corpus = MovieLensCorpus.open(
            cache_dir,
            split_seed=split_seed,
            source_size=source_size,
        )
        if progress:
            tqdm.write(
                f"MovieLens corpus: {corpus.rows:,} ratings (cached)",
                file=sys.stderr if progress_file is None else progress_file,
            )
        return corpus

    raw_user_path = _temporary_path(cache_dir, "users.dat")
    raw_movie_path = _temporary_path(cache_dir, "movies.dat")
    raw_rating_path = _temporary_path(cache_dir, "ratings.dat")
    encoded_path = _temporary_path(cache_dir, "encoded.npy")
    split_path = _temporary_path(cache_dir, "split.dat")
    feature_path = _temporary_path(cache_dir, FEATURE_IDS_FILE)
    rating_path = _temporary_path(cache_dir, RATINGS_FILE)
    metadata_tmp = _temporary_path(cache_dir, METADATA_FILE)
    temporary_paths = (
        raw_user_path,
        raw_movie_path,
        raw_rating_path,
        encoded_path,
        split_path,
        feature_path,
        rating_path,
        metadata_tmp,
    )

    rows = 0
    user_values: set[int] = set()
    movie_values: set[int] = set()
    try:
        with (
            raw_user_path.open("wb") as user_file,
            raw_movie_path.open("wb") as movie_file,
            raw_rating_path.open("wb") as rating_file,
            _open_ratings(raw_path) as source,
        ):
            chunks = pd.read_csv(
                source,
                usecols=("userId", "movieId", "rating"),
                dtype={
                    "userId": np.uint32,
                    "movieId": np.uint32,
                    "rating": np.float32,
                },
                chunksize=chunk_size,
            )
            progress_bar = tqdm(
                desc="Reading MovieLens ratings",
                unit="rows",
                unit_scale=True,
                disable=not progress,
                file=progress_file,
            )
            with progress_bar:
                for chunk in chunks:
                    users = chunk["userId"].to_numpy(dtype=np.uint32)
                    movies = chunk["movieId"].to_numpy(dtype=np.uint32)
                    ratings = chunk["rating"].to_numpy(dtype=np.float32)
                    users.tofile(user_file)
                    movies.tofile(movie_file)
                    ratings.tofile(rating_file)
                    user_values.update(map(int, np.unique(users)))
                    movie_values.update(map(int, np.unique(movies)))
                    rows += len(chunk)
                    progress_bar.update(len(chunk))

        if rows == 0:
            raise ValueError("MovieLens ratings must not be empty")

        raw_users = np.memmap(
            raw_user_path, mode="r", dtype=np.uint32, shape=(rows,)
        )
        raw_movies = np.memmap(
            raw_movie_path, mode="r", dtype=np.uint32, shape=(rows,)
        )
        raw_ratings = np.memmap(
            raw_rating_path, mode="r", dtype=np.float32, shape=(rows,)
        )
        users = np.fromiter(sorted(user_values), dtype=np.uint32)
        movies = np.fromiter(sorted(movie_values), dtype=np.uint32)
        num_users, num_movies = len(users), len(movies)

        encoded = np.lib.format.open_memmap(
            encoded_path, mode="w+", dtype=np.int32, shape=(rows, NUM_FIELDS)
        )
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            encoded[start:stop, 0] = np.searchsorted(users, raw_users[start:stop])
            encoded[start:stop, 1] = num_users + np.searchsorted(
                movies, raw_movies[start:stop]
            )

        split = np.memmap(split_path, mode="w+", dtype=np.uint8, shape=(rows,))
        split[:] = _split_by_user(encoded, split_seed)
        training_movies = np.unique(encoded[split == 0, 1]) - num_users
        warm_movies = np.zeros(num_movies, dtype=bool)
        warm_movies[training_movies] = True
        holdout_movies = encoded[:, 1] - num_users
        cold_rows = np.flatnonzero((split != 0) & ~warm_movies[holdout_movies])
        _, first = np.unique(holdout_movies[cold_rows], return_index=True)
        split[cold_rows[first]] = 0

        counts = np.bincount(split, minlength=3)
        train_rows, val_rows, test_rows = map(int, counts)
        if not val_rows or not test_rows:
            raise ValueError("MovieLens validation and test splits must not be empty")

        feature_ids = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.int32, shape=(rows, NUM_FIELDS)
        )
        ratings = np.lib.format.open_memmap(
            rating_path, mode="w+", dtype=np.float32, shape=(rows,)
        )
        offsets = np.array([0, train_rows, train_rows + val_rows], dtype=np.int64)
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            for part in range(3):
                selected = split[start:stop] == part
                count = int(selected.sum())
                destination = slice(offsets[part], offsets[part] + count)
                feature_ids[destination] = encoded[start:stop][selected]
                ratings[destination] = raw_ratings[start:stop][selected]
                offsets[part] += count
        feature_ids.flush()
        ratings.flush()

        feature_path.replace(cache_dir / FEATURE_IDS_FILE)
        rating_path.replace(cache_dir / RATINGS_FILE)
        metadata = {
            "version": CACHE_VERSION,
            "source_size": source_size,
            "split_seed": split_seed,
            "rows": rows,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "num_users": num_users,
            "num_movies": num_movies,
            "rating_center": RATING_CENTER,
        }
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True))
        metadata_tmp.replace(metadata_path)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    return MovieLensCorpus.open(
        cache_dir,
        split_seed=split_seed,
        source_size=source_size,
    )


@dataclass(frozen=True)
class MovieLensCorpus:
    cache_dir: Path
    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    num_users: int
    num_movies: int
    rating_center: float
    split_seed: int
    source_size: int

    @classmethod
    def open(
        cls,
        cache_dir: Path,
        *,
        split_seed: int | None = None,
        source_size: int | None = None,
    ) -> "MovieLensCorpus":
        cache_dir = Path(cache_dir)
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        if metadata.get("version") != CACHE_VERSION:
            raise ValueError(f"expected MovieLens cache version {CACHE_VERSION}")
        if split_seed is not None and metadata["split_seed"] != split_seed:
            raise ValueError("cached MovieLens split seed does not match")
        if source_size is not None and metadata["source_size"] != source_size:
            raise ValueError("cached MovieLens source size does not match")
        return cls(
            cache_dir=cache_dir,
            **{
                key: metadata[key]
                for key in cls.__dataclass_fields__
                if key != "cache_dir"
            },
        )

    @property
    def num_features(self) -> int:
        return self.num_users + self.num_movies

    @property
    def val_stop(self) -> int:
        return self.train_rows + self.val_rows

    def feature_ids(self) -> np.memmap:
        return np.load(
            self.cache_dir / FEATURE_IDS_FILE, mmap_mode="r", allow_pickle=False
        )

    def ratings(self) -> np.memmap:
        return np.load(
            self.cache_dir / RATINGS_FILE, mmap_mode="r", allow_pickle=False
        )

    def order_path(self, seed: int) -> Path:
        seed = index(seed)
        if seed < 0:
            raise ValueError(f"data seed must be nonnegative; got {seed}")
        path = self.cache_dir / f"train_order_seed{seed}.npy"
        if path.exists():
            order = np.load(path, mmap_mode="r", allow_pickle=False)
            if order.dtype == np.uint32 and order.shape == (self.train_rows,):
                return path

        order = np.random.default_rng(seed).permutation(self.train_rows).astype(
            np.uint32, copy=False
        )
        temporary = _temporary_path(self.cache_dir, path.name)
        try:
            with temporary.open("wb") as file:
                np.save(file, order, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


@dataclass
class MovieLensTask:
    corpus: MovieLensCorpus
    data_seed: int
    batch_size: int
    _order_path: Path = field(init=False, repr=False)
    _feature_ids: np.memmap | None = field(init=False, default=None, repr=False)
    _ratings: np.memmap | None = field(init=False, default=None, repr=False)
    _order: np.memmap | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.batch_size = index(self.batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {self.batch_size}")
        self._order_path = self.corpus.order_path(self.data_seed)

    def __getstate__(self) -> dict[str, Any]:
        return vars(self) | {"_feature_ids": None, "_ratings": None, "_order": None}

    def train_batches(self, start: int, stop: int) -> Iterator[RatingBatch]:
        if not 0 <= start <= stop <= self.corpus.train_rows:
            raise ValueError(
                "expected 0 <= start <= stop <= train_rows; "
                f"got {start}, {stop}, {self.corpus.train_rows}"
            )
        feature_ids, ratings, order = self._arrays()
        for batch_start in range(start, stop, self.batch_size):
            rows = np.array(
                order[batch_start : min(batch_start + self.batch_size, stop)],
                copy=True,
            )
            rows.sort()
            yield self._batch(feature_ids, ratings, rows)

    def val_batches(self) -> Iterator[RatingBatch]:
        yield from self._sequential_batches(
            self.corpus.train_rows, self.corpus.val_stop
        )

    def test_batches(self) -> Iterator[RatingBatch]:
        yield from self._sequential_batches(self.corpus.val_stop, self.corpus.rows)

    def warm_coverage(
        self, checkpoints: Iterable[int]
    ) -> dict[int, tuple[float, float]]:
        """Fraction of holdout rows whose user and movie have appeared by a prefix."""
        checkpoints = tuple(map(index, checkpoints))
        if any(not 0 <= stop <= self.corpus.train_rows for stop in checkpoints):
            raise ValueError("coverage checkpoints must lie within the training set")

        feature_ids, _, order = self._arrays()
        holdouts = (
            feature_ids[self.corpus.train_rows : self.corpus.val_stop],
            feature_ids[self.corpus.val_stop :],
        )
        seen = np.zeros(self.corpus.num_features, dtype=bool)
        result: dict[int, tuple[float, float]] = {}
        start = 0
        for stop in sorted(set(checkpoints)):
            seen[feature_ids[order[start:stop]].ravel()] = True
            result[stop] = tuple(
                float(seen[rows].all(axis=1).mean()) for rows in holdouts
            )
            start = stop
        return {stop: result[stop] for stop in checkpoints}

    def _arrays(self) -> tuple[np.memmap, np.memmap, np.memmap]:
        if self._feature_ids is None:
            self._feature_ids = self.corpus.feature_ids()
            self._ratings = self.corpus.ratings()
            self._order = np.load(
                self._order_path, mmap_mode="r", allow_pickle=False
            )
        assert (
            self._feature_ids is not None
            and self._ratings is not None
            and self._order is not None
        )
        return self._feature_ids, self._ratings, self._order

    def _sequential_batches(self, start: int, stop: int) -> Iterator[RatingBatch]:
        feature_ids, ratings, _ = self._arrays()
        for batch_start in range(start, stop, self.batch_size):
            yield self._batch(
                feature_ids,
                ratings,
                slice(batch_start, min(batch_start + self.batch_size, stop)),
            )

    def _batch(
        self,
        feature_ids: np.memmap,
        ratings: np.memmap,
        rows: slice | np.ndarray,
    ) -> RatingBatch:
        batch_ids = np.array(feature_ids[rows], dtype=np.int32, copy=True)
        batch_ratings = np.array(ratings[rows], dtype=np.float32, copy=True)
        batch_ratings -= self.corpus.rating_center
        return ((torch.from_numpy(batch_ids),), torch.from_numpy(batch_ratings))
