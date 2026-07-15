"""Shared experiment machinery."""

from collections.abc import Callable, Iterable, Sized
from typing import TextIO

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def run_many[T, R](
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    progress: bool,
    unit: str,
    desc: str | None = None,
    progress_file: TextIO | None = None,
) -> list[R]:
    if workers < 1:
        raise ValueError(f"workers must be positive; got {workers}")

    progress_options = {
        "desc": desc,
        "unit": unit,
        "total": len(items) if isinstance(items, Sized) else None,
        "disable": not progress,
        "file": progress_file,
    }
    if workers == 1:
        return [function(item) for item in tqdm(items, **progress_options)]

    return process_map(
        function,
        items,
        max_workers=workers,
        chunksize=1,
        **progress_options,
    )
