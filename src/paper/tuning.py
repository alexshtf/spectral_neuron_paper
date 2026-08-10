import warnings
from collections.abc import Sequence

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


def _lowest_validation_rows(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    ordered = df.sort_values(
        group_cols + ["val_rmse", "step"], kind="mergesort"
    )
    return ordered.groupby(group_cols, as_index=False, sort=False).head(1)


def best_checkpoints(
    raw: pd.DataFrame, train_sizes: list[int] | tuple[int, ...]
) -> pd.DataFrame:
    selected = []

    for train_size in train_sizes:
        eligible = raw.loc[raw["train_size"] <= train_size].copy()
        if eligible.empty:
            continue
        eligible["train_size"] = train_size
        selected.append(
            _lowest_validation_rows(eligible, RUN_COLUMNS + ["train_size"])
        )

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
    return summarize_selected(
        select_lr(best_checkpoints(raw, _common_train_sizes(raw)))
    )
