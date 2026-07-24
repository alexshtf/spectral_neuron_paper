from collections.abc import Iterator
from compression import zstd
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


ZSTD_LEVEL = 3


@contextmanager
def open_dataset_file(path: Path) -> Iterator[BinaryIO]:
    path = Path(path)
    if path.suffix == ".zstd":
        with zstd.open(path, "rb") as file:
            yield file
    else:
        with path.open("rb") as file:
            yield file
