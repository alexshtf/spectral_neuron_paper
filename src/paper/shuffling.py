from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np


_CACHE_VERSION = 1
_ORDER_DTYPE = np.dtype(np.uint32)


@dataclass(frozen=True)
class ShuffledEpochs:
    cache_dir: Path
    size: int
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.size > np.iinfo(_ORDER_DTYPE).max + 1:
            raise ValueError("shuffle size does not fit in uint32 row ids")

    @property
    def _root(self) -> Path:
        return self.cache_dir / f"shuffle-v{_CACHE_VERSION}"

    def _path(self, pass_index: int) -> Path:
        return self._root / (f"n{self.size}_seed{self.seed}_pass{pass_index}.npy")

    def _save(self, path: Path, order: np.ndarray) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as file:
                np.save(file, order, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def prepare(self, passes: int) -> tuple[Path, ...]:
        paths = tuple(self._path(pass_index) for pass_index in range(passes))
        self._root.mkdir(parents=True, exist_ok=True)
        if all(path.exists() for path in paths):
            return paths

        rng = np.random.default_rng(self.seed)
        order = np.arange(self.size, dtype=_ORDER_DTYPE)
        for path in paths:
            rng.shuffle(order)
            if not path.exists():
                self._save(path, order)
        return paths

    def batches(self, stop: int, batch_size: int) -> Iterator[np.ndarray]:
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
            # Preserve random batch membership while reading memmaps in row order.
            rows.sort()
            yield rows
