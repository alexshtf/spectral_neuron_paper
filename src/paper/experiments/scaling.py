import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import TextIO

import pandas as pd
from tqdm import tqdm

from paper.experiments.results import summarize_quantiles
from paper.experiments.runner import run_many
from paper.tuning import (
    select_learning_rates,
    select_rows_at_learning_rates,
)


PROTOCOL = "repeated_shuffle"


@dataclass(frozen=True)
class ScalingSchema:
    experiment_columns: tuple[str, ...]
    model_columns: tuple[str, ...]
    model_spec_columns: tuple[str, ...]
    validation_metric: str
    test_metrics: tuple[str, ...]

    @property
    def identity_columns(self) -> tuple[str, ...]:
        return (
            *self.experiment_columns,
            "phase",
            "train_size",
            "data_seed",
            *self.model_columns,
            "lr",
            "init_seed",
        )

    @property
    def curve_columns(self) -> tuple[str, ...]:
        return (*self.experiment_columns, *self.model_columns, "train_size")

    @property
    def raw_columns(self) -> tuple[str, ...]:
        return (
            *self.identity_columns,
            self.validation_metric,
            *self.test_metrics,
        )


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
    schema: ScalingSchema,
    evaluation_seeds: SeedGrid,
    model_specs: Mapping[tuple[object, ...], T],
) -> tuple[SelectedRun[T], ...]:
    experiments = tuning[list(schema.experiment_columns)].drop_duplicates()
    if len(experiments) != 1:
        raise ValueError("selected runs require exactly one experiment")

    train_sizes: dict[RunConfig[T], list[int]] = {}
    for row in select_learning_rates(
        tuning,
        curve_columns=schema.curve_columns,
        validation_metric=schema.validation_metric,
    ).itertuples(index=False):
        model_key = tuple(
            getattr(row, column) for column in schema.model_spec_columns
        )
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
    schema: ScalingSchema,
) -> pd.DataFrame:
    learning_rates = select_learning_rates(
        raw.loc[raw["phase"] == "tuning"],
        curve_columns=schema.curve_columns,
        validation_metric=schema.validation_metric,
    )
    return select_rows_at_learning_rates(
        raw.loc[raw["phase"] == "evaluation"],
        learning_rates,
        curve_columns=schema.curve_columns,
    )


def summarize_evaluations(
    evaluations: pd.DataFrame,
    *,
    schema: ScalingSchema,
) -> pd.DataFrame:
    return summarize_quantiles(
        evaluations,
        group_columns=(*schema.curve_columns, "selected_lr"),
        metrics=schema.test_metrics,
    )
