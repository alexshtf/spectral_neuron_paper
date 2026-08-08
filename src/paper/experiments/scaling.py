import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


PROTOCOL = "repeated_shuffle"


@dataclass(frozen=True)
class SeedGrid:
    data_seeds: range = range(1)
    init_seeds: range = range(1)

    def __len__(self) -> int:
        return len(self.data_seeds) * len(self.init_seeds)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return product(self.data_seeds, self.init_seeds)


@dataclass(frozen=True)
class RunConfig[T]:
    data_seed: int
    model_spec: T
    lr: float
    init_seed: int


@dataclass(frozen=True)
class SelectedRun[T]:
    config: RunConfig[T]
    train_sizes: tuple[int, ...]


def tuning_configs[T](
    model_specs: tuple[T, ...],
    lrs: tuple[float, ...],
    seeds: SeedGrid,
) -> tuple[RunConfig[T], ...]:
    return tuple(
        RunConfig(data_seed, model_spec, lr, init_seed)
        for data_seed, model_spec, lr, init_seed in product(
            seeds.data_seeds,
            model_specs,
            lrs,
            seeds.init_seeds,
        )
    )


def best_lrs(
    tuning: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    validation_metric: str,
) -> pd.DataFrame:
    if tuning.empty:
        raise ValueError("tuning results must not be empty")

    curve_columns = list(curve_columns)
    median_metric = f"median_{validation_metric}"
    curves = tuning[curve_columns].drop_duplicates()
    finite_mask = np.isfinite(tuning[validation_metric])
    if not finite_mask.all():
        warnings.warn(
            f"{(~finite_mask).sum()} nonfinite {validation_metric} values ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    finite = tuning.loc[finite_mask]
    scores = (
        finite.groupby(curve_columns + ["lr"], as_index=False)[validation_metric]
        .median()
        .rename(columns={validation_metric: median_metric})
    )
    missing = curves.merge(
        scores[curve_columns].drop_duplicates(),
        on=curve_columns,
        how="left",
        indicator=True,
    )
    missing = missing.loc[missing["_merge"] == "left_only", curve_columns]
    if not missing.empty:
        raise ValueError(
            "no finite validation metric for " + repr(missing.to_dict("records"))
        )

    best = (
        scores.sort_values(
            curve_columns + [median_metric, "lr"],
            kind="mergesort",
        )
        .groupby(curve_columns, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )
    return best[curve_columns + ["selected_lr", median_metric]]


def selected_runs[T](
    tuning: pd.DataFrame,
    *,
    experiment_columns: Sequence[str],
    curve_columns: Sequence[str],
    validation_metric: str,
    evaluation_seeds: SeedGrid,
    make_model_spec: Callable[[str, int], T],
) -> tuple[SelectedRun[T], ...]:
    experiments = tuning[list(experiment_columns)].drop_duplicates()
    if len(experiments) != 1:
        raise ValueError("selected runs require exactly one experiment")

    train_sizes: dict[RunConfig[T], list[int]] = {}
    for row in best_lrs(
        tuning,
        curve_columns=curve_columns,
        validation_metric=validation_metric,
    ).itertuples(index=False):
        for data_seed, init_seed in evaluation_seeds:
            config = RunConfig(
                data_seed=data_seed,
                model_spec=make_model_spec(row.model, int(row.dim)),
                lr=row.selected_lr,
                init_seed=init_seed,
            )
            train_sizes.setdefault(config, []).append(int(row.train_size))

    return tuple(
        SelectedRun(config, tuple(sorted(set(sizes))))
        for config, sizes in train_sizes.items()
    )


def select_lr(
    raw: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    validation_metric: str,
) -> pd.DataFrame:
    curve_columns = list(curve_columns)
    best = best_lrs(
        raw.loc[raw["phase"] == "tuning"],
        curve_columns=curve_columns,
        validation_metric=validation_metric,
    )
    selected = raw.loc[raw["phase"] == "evaluation"].merge(
        best,
        on=curve_columns,
        how="inner",
    )
    return selected.loc[selected["lr"] == selected["selected_lr"]].reset_index(
        drop=True
    )


def summarize_scaling(
    raw: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    validation_metric: str,
    quantile_metrics: Sequence[str],
) -> pd.DataFrame:
    quantile_metrics = tuple(quantile_metrics)
    if not quantile_metrics:
        raise ValueError("at least one quantile metric is required")

    selected = select_lr(
        raw,
        curve_columns=curve_columns,
        validation_metric=validation_metric,
    )
    aggregations: dict[str, tuple[str, str | Callable]] = {}
    for metric in quantile_metrics:
        aggregations[f"median_{metric}"] = (metric, "median")
        aggregations[f"q25_{metric}"] = (metric, lambda s: s.quantile(0.25))
        aggregations[f"q75_{metric}"] = (metric, lambda s: s.quantile(0.75))
    aggregations["n"] = (quantile_metrics[0], "size")

    return (
        selected.groupby([*curve_columns, "selected_lr"])
        .agg(**aggregations)
        .reset_index()
    )
