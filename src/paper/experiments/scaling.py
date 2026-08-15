import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import TextIO

import pandas as pd
from tqdm import tqdm

from paper.experiments.runner import run_many
from paper.tuning import select_learning_rates, select_rows_at_learning_rates


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


def _report_timings(
    phase: str,
    evaluation_prefix: str,
    results: list[pd.DataFrame],
    progress_file: TextIO | None,
) -> None:
    train_seconds = sum(result["train_seconds"].iloc[-1] for result in results)
    evaluation_seconds = sum(
        result[f"{evaluation_prefix}_seconds"].iloc[-1] for result in results
    )
    evaluation = "validation" if evaluation_prefix == "val" else "test"
    tqdm.write(
        f"{phase} aggregate trajectory time: "
        f"training={timedelta(seconds=round(train_seconds))}, "
        f"{evaluation}={timedelta(seconds=round(evaluation_seconds))}",
        file=sys.stderr if progress_file is None else progress_file,
    )


def run_tuning_and_evaluation[T](
    configs: Sequence[RunConfig[T]],
    *,
    tune: Callable[[RunConfig[T]], pd.DataFrame],
    select_evaluation_runs: Callable[[pd.DataFrame], Sequence[SelectedRun[T]]],
    evaluate: Callable[[SelectedRun[T]], pd.DataFrame],
    workers: int,
    progress: bool,
    progress_file: TextIO | None,
) -> pd.DataFrame:
    tuning_results = run_many(
        tune,
        configs,
        workers=workers,
        desc="Tuning (train + validation)",
        unit="trajectory",
        progress=progress,
        progress_file=progress_file,
    )
    if progress:
        _report_timings("Tuning", "val", tuning_results, progress_file)
    tuning = pd.concat(tuning_results, ignore_index=True)

    evaluation_results = run_many(
        evaluate,
        select_evaluation_runs(tuning),
        workers=workers,
        desc="Evaluation (retrain + test)",
        unit="trajectory",
        progress=progress,
        progress_file=progress_file,
    )
    if progress:
        _report_timings("Evaluation", "test", evaluation_results, progress_file)

    evaluation = pd.concat(evaluation_results, ignore_index=True)
    return pd.concat((tuning, evaluation), ignore_index=True)


def select_evaluation_runs[T](
    tuning: pd.DataFrame,
    *,
    experiment_columns: Sequence[str],
    curve_columns: Sequence[str],
    validation_metric: str,
    evaluation_seeds: SeedGrid,
    model_columns: Sequence[str],
    model_specs: Mapping[tuple[object, ...], T],
) -> tuple[SelectedRun[T], ...]:
    experiments = tuning[list(experiment_columns)].drop_duplicates()
    if len(experiments) != 1:
        raise ValueError("selected runs require exactly one experiment")

    train_sizes: dict[RunConfig[T], list[int]] = {}
    for row in select_learning_rates(
        tuning,
        curve_columns=curve_columns,
        validation_metric=validation_metric,
    ).itertuples(index=False):
        model_key = tuple(getattr(row, column) for column in model_columns)
        for data_seed, init_seed in evaluation_seeds:
            config = RunConfig(
                data_seed=data_seed,
                model_spec=model_specs[model_key],
                lr=row.selected_lr,
                init_seed=init_seed,
            )
            train_sizes.setdefault(config, []).append(int(row.train_size))

    return tuple(
        SelectedRun(config, tuple(sorted(set(sizes))))
        for config, sizes in train_sizes.items()
    )


def select_evaluations(
    raw: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    validation_metric: str,
) -> pd.DataFrame:
    curve_columns = list(curve_columns)
    learning_rates = select_learning_rates(
        raw.loc[raw["phase"] == "tuning"],
        curve_columns=curve_columns,
        validation_metric=validation_metric,
    )
    return select_rows_at_learning_rates(
        raw.loc[raw["phase"] == "evaluation"],
        learning_rates,
        curve_columns=curve_columns,
    )


def summarize_evaluations(
    evaluations: pd.DataFrame,
    *,
    curve_columns: Sequence[str],
    quantile_metrics: Sequence[str],
) -> pd.DataFrame:
    quantile_metrics = tuple(quantile_metrics)
    aggregations: dict[str, tuple[str, str | Callable]] = {}
    for metric in quantile_metrics:
        aggregations[f"median_{metric}"] = (metric, "median")
        aggregations[f"q25_{metric}"] = (metric, lambda s: s.quantile(0.25))
        aggregations[f"q75_{metric}"] = (metric, lambda s: s.quantile(0.75))
    aggregations["n"] = (quantile_metrics[0], "size")

    return (
        evaluations.groupby([*curve_columns, "selected_lr"])
        .agg(**aggregations)
        .reset_index()
    )
