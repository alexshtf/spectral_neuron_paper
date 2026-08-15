import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import TextIO

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from paper.experiments.results import summarize_quantiles
from paper.experiments.runner import run_many
from paper.tasks import Task
from paper.training import (
    Objective,
    fit_test_trajectory,
    fit_validation_trajectory,
)
from paper.tuning import (
    same_learning_rates,
    select_learning_rates,
    select_rows_at_learning_rates,
)


PROTOCOL = "repeated_shuffle"

_TIMING_COLUMNS = ("train_seconds", "val_seconds", "test_seconds")


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
    test_checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class ScalingRunner[T]:
    schema: ScalingSchema
    checkpoints: tuple[int, ...]
    objective: Objective
    make_task_model: Callable[[RunConfig[T]], tuple[Task, nn.Module]]
    metadata: Callable[[RunConfig[T], nn.Module], Mapping[str, object]]

    def _format_trajectory(
        self,
        trajectory: pd.DataFrame,
        *,
        phase: str,
        config: RunConfig[T],
        model: nn.Module,
    ) -> pd.DataFrame:
        return trajectory.assign(
            phase=phase,
            **self.metadata(config, model),
        ).reindex(columns=(*self.schema.raw_columns, *_TIMING_COLUMNS))

    def tune(self, config: RunConfig[T]) -> pd.DataFrame:
        task, model = self.make_task_model(config)
        trajectory = fit_validation_trajectory(
            task,
            model,
            objective=self.objective,
            lr=config.lr,
            checkpoints=self.checkpoints,
        )
        return self._format_trajectory(
            trajectory,
            phase="tuning",
            config=config,
            model=model,
        )

    def evaluate(self, selected: SelectedRun[T]) -> pd.DataFrame:
        task, model = self.make_task_model(selected.config)
        checkpoints = tuple(
            checkpoint
            for checkpoint in self.checkpoints
            if checkpoint <= max(selected.test_checkpoints)
        )
        trajectory = fit_test_trajectory(
            task,
            model,
            objective=self.objective,
            lr=selected.config.lr,
            checkpoints=checkpoints,
            test_checkpoints=selected.test_checkpoints,
        )
        return self._format_trajectory(
            trajectory,
            phase="evaluation",
            config=selected.config,
            model=model,
        )


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
    runner: ScalingRunner[T],
    select_evaluation_runs: Callable[[pd.DataFrame], Sequence[SelectedRun[T]]],
    workers: int,
    progress: bool,
    progress_file: TextIO | None,
) -> pd.DataFrame:
    tuning_results = run_many(
        runner.tune,
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
        runner.evaluate,
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

    test_checkpoints: dict[RunConfig[T], list[int]] = {}
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
            test_checkpoints.setdefault(config, []).append(int(row.train_size))

    return tuple(
        SelectedRun(config, tuple(sorted(set(sizes))))
        for config, sizes in test_checkpoints.items()
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


def _row_set(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[object, ...]]:
    return set(frame[list(columns)].itertuples(index=False, name=None))


def validate_results(
    raw: pd.DataFrame,
    *,
    schema: ScalingSchema,
    expected_model_rows: Sequence[Mapping[str, object]],
    train_sizes: Sequence[int],
    learning_rates: Sequence[float],
    tuning_seeds: SeedGrid,
    evaluation_seeds: SeedGrid,
) -> dict[str, object]:
    """Validate one complete tuning-and-evaluation scaling experiment."""
    if tuple(raw.columns) != schema.raw_columns:
        raise ValueError("incompatible result schema")
    if raw[list(schema.identity_columns)].isna().any().any():
        raise ValueError("run identity columns must not contain missing values")
    if raw.duplicated(list(schema.identity_columns)).any():
        raise ValueError("results contain duplicate trajectory checkpoints")

    experiments = raw[list(schema.experiment_columns)].drop_duplicates()
    if len(experiments) != 1:
        raise ValueError("results must contain exactly one experiment")
    experiment = experiments.iloc[0].to_dict()

    if set(raw["phase"]) != {"tuning", "evaluation"}:
        raise ValueError("results must contain tuning and evaluation phases")

    expected_models = {
        tuple(row[column] for column in schema.model_columns)
        for row in expected_model_rows
    }
    observed_models = _row_set(raw, schema.model_columns)
    if observed_models != expected_models:
        raise ValueError(
            "model/capacity grid mismatch; capacity metadata must be consistent"
        )

    experiment_row = tuple(
        experiment[column] for column in schema.experiment_columns
    )
    expected_curves = {
        (*experiment_row, *model, train_size)
        for model in expected_models
        for train_size in train_sizes
    }
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    for phase, rows in (("tuning", tuning), ("evaluation", evaluation)):
        observed_curves = _row_set(rows, schema.curve_columns)
        if observed_curves != expected_curves:
            raise ValueError(f"{phase} has an incomplete model/checkpoint grid")

    if tuning[list(schema.test_metrics)].notna().any().any():
        raise ValueError("tuning rows must not contain test metrics")
    if evaluation[schema.validation_metric].notna().any():
        raise ValueError("evaluation rows must not contain validation metrics")
    if not np.isfinite(
        evaluation[list(schema.test_metrics)].to_numpy(dtype=float)
    ).all():
        raise ValueError("evaluation test metrics must be finite")

    expected_tuning_seeds = set(tuning_seeds)
    for curve, rows in tuning.groupby(list(schema.curve_columns)):
        if not same_learning_rates(rows["lr"], learning_rates):
            raise ValueError(f"incomplete tuning learning-rate grid for {curve}")
        for lr, lr_rows in rows.groupby("lr"):
            seeds = _row_set(lr_rows, ("data_seed", "init_seed"))
            if seeds != expected_tuning_seeds:
                raise ValueError(f"incomplete tuning seeds for {curve}, lr={lr:g}")

    selected_lrs = select_learning_rates(
        tuning,
        curve_columns=schema.curve_columns,
        validation_metric=schema.validation_metric,
    ).set_index(list(schema.curve_columns))["selected_lr"]
    expected_evaluation_seeds = set(evaluation_seeds)
    for curve, rows in evaluation.groupby(list(schema.curve_columns)):
        seeds = _row_set(rows, ("data_seed", "init_seed"))
        if seeds != expected_evaluation_seeds:
            raise ValueError(f"incomplete evaluation seeds for {curve}")
        lrs = rows["lr"].unique()
        if len(lrs) != 1 or not np.isclose(
            lrs[0], selected_lrs.loc[curve], rtol=1e-12, atol=0
        ):
            raise ValueError(f"evaluation does not use the selected LR for {curve}")

    return experiment
