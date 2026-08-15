import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch

from paper.experiments.results import (
    DEFAULT_RUNS_DIR,
    WRITE_MODES,
    summarize_quantiles,
    write_csv,
)
from paper.experiments.runner import run_many
from paper.models import ModelKind, ModelSpec, make_model
from paper.targets import ArrayTarget, TargetKind, TargetSpec
from paper.tasks import SyntheticTask
from paper.training import REGRESSION_OBJECTIVE, fit_and_evaluate
from paper.tuning import (
    same_learning_rates,
    select_learning_rates,
    select_rows_at_learning_rates,
)

type TargetFactory = Callable[[TargetSpec], ArrayTarget]
type TaskFactory = Callable[..., SyntheticTask]
type ProfileRunner = Callable[..., pd.DataFrame]


RAW_COLUMNS = [
    "target_kind",
    "complexity",
    "target_seed",
    "noise_std",
    "model",
    "dim",
    "lr",
    "init_seed",
    "batch_size",
    "step",
    "train_size",
    "val_rmse",
    "test_rmse",
]

FITS: tuple[tuple[TargetKind, ModelKind], ...] = (
    ("general", "unconstrained"),
    ("monotone", "unconstrained"),
    ("monotone", "monotone"),
)

RUN_ID_COLUMNS = [
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

CURVE_COLUMNS = [
    "target_kind",
    "complexity",
    "noise_std",
    "model",
    "dim",
    "batch_size",
    "train_size",
]


@dataclass(frozen=True)
class Profile:
    complexities: tuple[int, ...]
    target_seeds: range
    init_seeds: range
    dims: tuple[int, ...]
    lrs: tuple[float, ...]
    train_sizes: tuple[int, ...]
    batch_size: int = 32
    noise_stds: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class RunConfig:
    target_spec: TargetSpec
    noise_std: float
    model_spec: ModelSpec
    lr: float
    init_seed: int


@dataclass(frozen=True)
class RunGrid:
    profile: Profile

    def __len__(self) -> int:
        return (
            len(self.profile.dims)
            * len(self.profile.complexities)
            * len(self.profile.target_seeds)
            * len(FITS)
            * len(self.profile.noise_stds)
            * len(self.profile.lrs)
            * len(self.profile.init_seeds)
        )

    def __iter__(self) -> Iterator[RunConfig]:
        for (
            dim,
            complexity,
            target_seed,
            (target_kind, model_kind),
            noise_std,
            lr,
            init_seed,
        ) in product(
            self.profile.dims,
            self.profile.complexities,
            self.profile.target_seeds,
            FITS,
            self.profile.noise_stds,
            self.profile.lrs,
            self.profile.init_seeds,
        ):
            yield RunConfig(
                target_spec=TargetSpec(target_kind, complexity, target_seed),
                noise_std=noise_std,
                model_spec=ModelSpec(model_kind, dim),
                lr=lr,
                init_seed=init_seed,
            )


@dataclass(frozen=True)
class RunSettings:
    batch_size: int
    train_sizes: tuple[int, ...]
    val_size: int
    test_size: int


def _make_seeded_model(
    model_spec: ModelSpec, *, input_dim: int, init_seed: int
) -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(model_spec, input_dim)


def _with_metadata(
    df: pd.DataFrame, *, config: RunConfig, batch_size: int
) -> pd.DataFrame:
    metadata = {
        "target_kind": config.target_spec.kind,
        "complexity": config.target_spec.complexity,
        "target_seed": config.target_spec.seed,
        "noise_std": config.noise_std,
        "model": config.model_spec.kind,
        "dim": config.model_spec.dim,
        "lr": config.lr,
        "init_seed": config.init_seed,
    }
    return df.assign(
        **metadata,
        batch_size=batch_size,
    ).loc[:, RAW_COLUMNS]


def run_config(
    config: RunConfig,
    settings: RunSettings,
    *,
    make_target: TargetFactory,
    make_task: TaskFactory,
) -> pd.DataFrame:
    target = make_target(config.target_spec)
    task = make_task(
        target,
        lower=config.target_spec.lower,
        upper=config.target_spec.upper,
        batch_size=settings.batch_size,
        val_size=settings.val_size,
        test_size=settings.test_size,
        seed=config.target_spec.seed,
        train_seed=config.init_seed,
        noise_std=config.noise_std,
    )
    model = _make_seeded_model(
        config.model_spec,
        input_dim=task.input_dim,
        init_seed=config.init_seed,
    )
    if any(train_size % settings.batch_size for train_size in settings.train_sizes):
        raise ValueError("synthetic train sizes must be divisible by batch_size")
    df = fit_and_evaluate(
        task,
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
    )
    return _with_metadata(df, config=config, batch_size=settings.batch_size)


def run_profile(
    profile: Profile,
    *,
    make_target: TargetFactory,
    make_task: TaskFactory,
    val_size: int = 4096,
    test_size: int = 4096,
    workers: int = 1,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> pd.DataFrame:
    configs = RunGrid(profile)
    settings = RunSettings(
        batch_size=profile.batch_size,
        train_sizes=profile.train_sizes,
        val_size=val_size,
        test_size=test_size,
    )
    run = partial(
        run_config,
        settings=settings,
        make_target=make_target,
        make_task=make_task,
    )
    dfs = run_many(
        run,
        configs,
        workers=workers,
        progress=progress,
        unit="experiment",
        progress_file=progress_file,
    )

    if not dfs:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


def _checkpoint_selections(
    raw: pd.DataFrame, budgets: Sequence[int]
) -> Iterator[pd.DataFrame]:
    ranked = raw.assign(_row_order=range(len(raw)))
    for budget in budgets:
        eligible = ranked.loc[ranked["train_size"] <= budget]
        if eligible.empty:
            continue

        selected = (
            eligible.sort_values(
                [*RUN_ID_COLUMNS, "val_rmse", "step", "_row_order"],
                na_position="last",
            )
            .drop_duplicates(RUN_ID_COLUMNS)
            .drop(columns="_row_order")
            .copy()
        )
        selected["train_size"] = budget
        yield selected


def select_checkpoints(
    raw: pd.DataFrame, budgets: Sequence[int]
) -> pd.DataFrame:
    selected = list(_checkpoint_selections(raw, budgets))
    return pd.concat(selected, ignore_index=True) if selected else raw.head(0)


def select_evaluations(checkpoints: pd.DataFrame) -> pd.DataFrame:
    learning_rates = select_learning_rates(
        checkpoints,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_rmse",
    )
    return select_rows_at_learning_rates(
        checkpoints,
        learning_rates,
        curve_columns=CURVE_COLUMNS,
    )


def summarize_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    return summarize_quantiles(
        evaluations,
        group_columns=(*CURVE_COLUMNS, "selected_lr"),
        metrics=("test_rmse",),
    )


def _common_train_sizes(raw: pd.DataFrame) -> tuple[int, ...]:
    grids = raw.groupby(RUN_ID_COLUMNS, sort=False, dropna=False)[
        "train_size"
    ].agg(lambda values: tuple(sorted(values.unique())))
    if grids.empty:
        return ()
    if grids.nunique() != 1:
        raise ValueError("synthetic runs have inconsistent train_size checkpoints")
    return grids.iat[0]


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_evaluations(
        select_evaluations(select_checkpoints(raw, _common_train_sizes(raw)))
    )
    return (
        summary
        .sort_values([*CURVE_COLUMNS, "selected_lr"], kind="mergesort")
        .reset_index(drop=True)
    )


def validate_raw(raw: pd.DataFrame, profile: Profile) -> None:
    """Validate that raw results are a complete run of a synthetic profile."""
    if raw.columns.tolist() != RAW_COLUMNS:
        raise ValueError("synthetic result columns do not match the raw schema")

    base_columns = [column for column in RUN_ID_COLUMNS if column != "lr"]
    expected_configs = {
        (
            config.target_spec.kind,
            config.target_spec.complexity,
            config.target_spec.seed,
            config.noise_std,
            config.model_spec.kind,
            config.model_spec.dim,
            config.init_seed,
            profile.batch_size,
        )
        for config in RunGrid(profile)
    }
    actual_configs = set(
        raw[base_columns].drop_duplicates().itertuples(index=False, name=None)
    )
    if actual_configs != expected_configs:
        raise ValueError("synthetic run grid does not match the profile")
    lr_grids = raw.groupby(base_columns, sort=False, dropna=False)["lr"].agg(
        lambda values: same_learning_rates(values, profile.lrs)
    )
    if not lr_grids.all():
        raise ValueError("synthetic learning-rate grids do not match the profile")

    keys = RUN_ID_COLUMNS + ["train_size"]
    if raw.duplicated(keys).any():
        raise ValueError("synthetic results contain duplicate checkpoints")
    checkpoints = raw.groupby(RUN_ID_COLUMNS, sort=False, dropna=False)[
        "train_size"
    ].agg(lambda values: tuple(sorted(values)))
    if not checkpoints.isin([profile.train_sizes]).all():
        raise ValueError("synthetic checkpoint grids do not match the profile")
    if not (raw["train_size"] == raw["step"] * raw["batch_size"]).all():
        raise ValueError("synthetic train sizes do not match their steps")
    if not np.isfinite(raw[["val_rmse", "test_rmse"]]).all().all():
        raise ValueError("synthetic metrics must be finite")


def build_arg_parser(profiles: Mapping[str, Profile]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=profiles.keys(), default="sanity")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-mode", choices=WRITE_MODES, default="overwrite")
    return parser


def default_raw_path(experiment_name: str, profile_name: str) -> Path:
    return DEFAULT_RUNS_DIR / f"{experiment_name}_{profile_name}.csv.zst"


def run_cli(
    experiment_name: str,
    profiles: Mapping[str, Profile],
    run_profile: ProfileRunner,
    *,
    argv: list[str] | None = None,
) -> None:
    parser = build_arg_parser(profiles)
    args = parser.parse_args(argv)

    profile = profiles[args.profile]
    out = args.out or default_raw_path(experiment_name, args.profile)
    raw = run_profile(profile, workers=args.workers, progress=not args.quiet)
    write_csv(raw, out, write_mode=args.write_mode)
