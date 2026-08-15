import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


def select_learning_rates(
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


def select_rows_at_learning_rates(
    rows: pd.DataFrame,
    learning_rates: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
) -> pd.DataFrame:
    selected = rows.merge(learning_rates, on=list(curve_columns), how="inner")
    return selected.loc[
        selected["lr"] == selected["selected_lr"]
    ].reset_index(drop=True)


def same_learning_rates(actual: pd.Series, expected: Sequence[float]) -> bool:
    values = np.sort(actual.unique())
    return len(values) == len(expected) and np.allclose(
        values, expected, rtol=1e-12, atol=0
    )
