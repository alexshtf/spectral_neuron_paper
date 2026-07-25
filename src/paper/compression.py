from compression import zstd
from pathlib import Path
from typing import BinaryIO


ZSTD_LEVEL = 3


def open_dataset_file(path: Path) -> BinaryIO:
    path = Path(path)
    return zstd.open(path, "rb") if path.suffix == ".zstd" else path.open("rb")
