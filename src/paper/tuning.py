import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd


RUN_COLUMNS = [
    "target_kind",
    "complexity",
    "target_seed",
    "noise_std",
    "model",
    "dim",
    "lr",
    "init_seed",
    "batch_size",
]

SELECTION_COLUMNS = [
    "target_kind",
    "complexity",
    "noise_std",
    "model",
    "dim",
    "batch_size",
    "train_size",
]


def _best_checkpoints_by_budget(
    raw: pd.DataFrame, train_sizes: Sequence[int]
) -> Iterator[pd.DataFrame]:
    groups = raw.groupby(RUN_COLUMNS, sort=True, dropna=False).ngroup().to_numpy()
    actual_train_sizes = raw["train_size"].to_numpy()
    val_rmse = raw["val_rmse"].to_numpy()
    steps = raw["step"].to_numpy()

    for train_size in train_sizes:
        eligible = np.flatnonzero(actual_train_sizes <= train_size)
        if not eligible.size:
            continue

        order = np.lexsort(
            (steps[eligible], val_rmse[eligible], groups[eligible])
        )
        ranked = eligible[order]
        ranked_groups = groups[ranked]
        first_in_run = np.r_[True, ranked_groups[1:] != ranked_groups[:-1]]
        selected = raw.iloc[ranked[first_in_run]].copy()
        selected["train_size"] = train_size
        yield selected


def best_checkpoints(
    raw: pd.DataFrame, train_sizes: list[int] | tuple[int, ...]
) -> pd.DataFrame:
    selected = list(_best_checkpoints_by_budget(raw, train_sizes))

    if not selected:
        return raw.head(0)

    return pd.concat(selected, ignore_index=True)


def best_lrs(
    tuning: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    validation_metric: str,
) -> pd.DataFrame:
    if tuning.empty:
        raise ValueError("tuning results must not be empty")

    curve_columns = list(curve_columns)
    candidate_columns = curve_columns + ["lr"]
    finite = np.isfinite(tuning[validation_metric])
    if not finite.all():
        warnings.warn(
            f"{(~finite).sum()} nonfinite {validation_metric} values reject "
            "their LR candidates",
            RuntimeWarning,
            stacklevel=2,
        )
    candidates = tuning.assign(_finite=finite)
    eligible = candidates.groupby(candidate_columns, dropna=False)[
        "_finite"
    ].transform("all")

    median_metric = f"median_{validation_metric}"
    scores = (
        tuning.loc[eligible]
        .groupby(candidate_columns, as_index=False)[validation_metric]
        .median()
        .rename(columns={validation_metric: median_metric})
    )
    missing = tuning[curve_columns].drop_duplicates().merge(
        scores[curve_columns].drop_duplicates(),
        on=curve_columns,
        how="left",
        indicator=True,
    )
    missing = missing.loc[missing["_merge"] == "left_only", curve_columns]
    if not missing.empty:
        raise ValueError(
            "no finite validation candidate for "
            + repr(missing.to_dict("records"))
        )

    best = (
        scores.sort_values(
            curve_columns + [median_metric, "lr"], kind="mergesort"
        )
        .groupby(curve_columns, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )
    return best[curve_columns + ["selected_lr", median_metric]]


def same_lrs(actual: pd.Series, expected: Sequence[float]) -> bool:
    values = np.sort(actual.unique())
    return len(values) == len(expected) and np.allclose(
        values, expected, rtol=1e-12, atol=0
    )


def select_lr(best: pd.DataFrame) -> pd.DataFrame:
    selected_lrs = best_lrs(
        best,
        curve_columns=SELECTION_COLUMNS,
        validation_metric="val_rmse",
    )

    selected = best.merge(
        selected_lrs,
        on=SELECTION_COLUMNS,
        how="inner",
    )
    selected = selected.loc[selected["lr"] == selected["selected_lr"]].copy()
    return selected.reset_index(drop=True)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    group_cols = SELECTION_COLUMNS + ["selected_lr"]
    summary = selected.groupby(group_cols)["test_rmse"].agg(
        median_test_rmse="median",
        q25_test_rmse=lambda s: s.quantile(0.25),
        q75_test_rmse=lambda s: s.quantile(0.75),
        n="size",
    )
    return summary.reset_index()


def _common_train_sizes(raw: pd.DataFrame) -> tuple[int, ...]:
    grids = raw.groupby(RUN_COLUMNS, sort=False, dropna=False)["train_size"].agg(
        lambda values: tuple(sorted(values.unique()))
    )
    if grids.empty:
        return ()
    if grids.nunique() != 1:
        raise ValueError("synthetic runs have inconsistent train_size checkpoints")
    return grids.iat[0]


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    summaries = (
        summarize_selected(select_lr(best))
        for best in _best_checkpoints_by_budget(raw, _common_train_sizes(raw))
    )
    return (
        pd.concat(summaries, ignore_index=True)
        .sort_values(SELECTION_COLUMNS + ["selected_lr"], kind="mergesort")
        .reset_index(drop=True)
    )
