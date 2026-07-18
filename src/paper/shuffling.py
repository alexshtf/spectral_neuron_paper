from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index
from pathlib import Path
from uuid import uuid4

import numpy as np


_CACHE_VERSION = 1
_ORDER_DTYPE = np.dtype(np.uint32)


def _integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _positive(name: str, value: int) -> int:
    value = _integer(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


@dataclass(frozen=True)
class ShuffledEpochs:
    cache_dir: Path
    size: int
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        size = _positive("size", self.size)
        if size > np.iinfo(_ORDER_DTYPE).max + 1:
            raise ValueError("size does not fit in uint32 row ids")
        object.__setattr__(self, "size", size)
        seed = _integer("seed", self.seed)
        if seed < 0:
            raise ValueError(f"seed must be nonnegative; got {seed}")
        object.__setattr__(self, "seed", seed)

    @property
    def _root(self) -> Path:
        return self.cache_dir / f"shuffle-v{_CACHE_VERSION}"

    def _path(self, pass_index: int) -> Path:
        return self._root / (f"n{self.size}_seed{self.seed}_pass{pass_index}.npy")

    def _valid(self, path: Path) -> bool:
        try:
            order = np.load(path, mmap_mode="r", allow_pickle=False)
        except (EOFError, OSError, ValueError):
            return False
        return order.dtype == _ORDER_DTYPE and order.shape == (self.size,)

    def _save(self, path: Path, order: np.ndarray) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as file:
                np.save(file, order, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def prepare(self, passes: int) -> tuple[Path, ...]:
        passes = _positive("passes", passes)
        paths = tuple(self._path(pass_index) for pass_index in range(passes))
        self._root.mkdir(parents=True, exist_ok=True)
        if all(self._valid(path) for path in paths):
            return paths

        rng = np.random.default_rng(self.seed)
        order = np.arange(self.size, dtype=_ORDER_DTYPE)
        for path in paths:
            rng.shuffle(order)
            if not self._valid(path):
                self._save(path, order)
        return paths

    def batches(self, stop: int, batch_size: int) -> Iterator[np.ndarray]:
        stop = _integer("stop", stop)
        if stop < 0:
            raise ValueError(f"stop must be nonnegative; got {stop}")
        batch_size = _positive("batch_size", batch_size)
        if stop == 0:
            return

        passes = (stop + self.size - 1) // self.size
        orders = tuple(
            np.load(path, mmap_mode="r", allow_pickle=False)
            for path in self.prepare(passes)
        )
        for batch_start in range(0, stop, batch_size):
            batch_stop = min(batch_start + batch_size, stop)
            rows = np.empty(batch_stop - batch_start, dtype=_ORDER_DTYPE)
            cursor = batch_start
            output_start = 0
            while cursor < batch_stop:
                pass_index, pass_offset = divmod(cursor, self.size)
                output_stop = output_start + min(
                    batch_stop - cursor,
                    self.size - pass_offset,
                )
                rows[output_start:output_stop] = orders[pass_index][
                    pass_offset : pass_offset + output_stop - output_start
                ]
                cursor += output_stop - output_start
                output_start = output_stop
            rows.sort()
            yield rows


def resolve_train_sizes(
    requested: Iterable[int],
    *,
    train_pool_size: int,
    batch_size: int,
    passes: int | None = None,
) -> tuple[int, ...]:
    requested = tuple(_positive("requested train size", size) for size in requested)
    if not requested or requested != tuple(sorted(set(requested))):
        raise ValueError(
            "requested train sizes must be positive, unique, and increasing"
        )

    train_pool_size = _positive("train_pool_size", train_pool_size)
    batch_size = _positive("batch_size", batch_size)
    if passes is None:
        terminal = requested[-1]
        candidates = requested[:-1]
    else:
        passes = _positive("passes", passes)
        terminal = passes * train_pool_size
        candidates = (
            *(size for size in requested if size < terminal),
            *(epoch * train_pool_size for epoch in range(1, passes)),
        )

    rounded = {
        min(terminal, ((size + batch_size - 1) // batch_size) * batch_size)
        for size in candidates
    }
    return (*sorted(size for size in rounded if size < terminal), terminal)
