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
]

SELECTION_COLUMNS = [
    "target_kind",
    "complexity",
    "noise_std",
    "model",
    "dim",
    "budget",
]


def _lowest_validation_rows(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    ordered = df.sort_values(
        group_cols + ["val_rmse", "step"], kind="mergesort"
    )
    return ordered.groupby(group_cols, as_index=False, sort=False).head(1)


def best_checkpoints(
    raw: pd.DataFrame, budgets: list[int] | tuple[int, ...]
) -> pd.DataFrame:
    selected = []

    for budget in budgets:
        eligible = raw.loc[raw["step"] <= budget].copy()
        if eligible.empty:
            continue
        eligible["budget"] = budget
        selected.append(_lowest_validation_rows(eligible, RUN_COLUMNS + ["budget"]))

    if not selected:
        return raw.head(0).assign(budget=pd.Series(dtype=int))

    return pd.concat(selected, ignore_index=True)


def select_lr(best: pd.DataFrame) -> pd.DataFrame:
    lr_scores = (
        best.groupby(SELECTION_COLUMNS + ["lr"], as_index=False)["val_rmse"]
        .median()
        .rename(columns={"val_rmse": "median_val_rmse"})
    )
    best_lrs = (
        lr_scores.sort_values(
            SELECTION_COLUMNS + ["median_val_rmse", "lr"], kind="mergesort"
        )
        .groupby(SELECTION_COLUMNS, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )

    selected = best.merge(
        best_lrs[SELECTION_COLUMNS + ["selected_lr", "median_val_rmse"]],
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


def summarize_raw(
    raw: pd.DataFrame, budgets: list[int] | tuple[int, ...]
) -> pd.DataFrame:
    return summarize_selected(select_lr(best_checkpoints(raw, budgets)))
