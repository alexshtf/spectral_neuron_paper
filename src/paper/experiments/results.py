from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd


type WriteMode = Literal["overwrite", "append"]

DEFAULT_RUNS_DIR = Path("notebooks") / "runs"
WRITE_MODES: tuple[WriteMode, ...] = ("overwrite", "append")


def summarize_quantiles(
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    metrics = tuple(metrics)
    aggregations: dict[str, tuple[str, str | Callable]] = {}
    for metric in metrics:
        aggregations[f"median_{metric}"] = (metric, "median")
        aggregations[f"q25_{metric}"] = (metric, lambda s: s.quantile(0.25))
        aggregations[f"q75_{metric}"] = (metric, lambda s: s.quantile(0.75))
    aggregations["n"] = (metrics[0], "size")

    return rows.groupby(list(group_columns)).agg(**aggregations).reset_index()


def write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    write_mode: WriteMode = "overwrite",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    append = write_mode == "append"
    has_content = path.exists() and path.stat().st_size > 0
    if append and has_content:
        existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
        new_columns = df.columns.tolist()
        if existing_columns != new_columns:
            raise ValueError(
                f"cannot append to {path}: CSV header {existing_columns} "
                f"does not match columns {new_columns}"
            )

    df.to_csv(
        path,
        mode="a" if append else "w",
        header=not append or not has_content,
        index=False,
    )
