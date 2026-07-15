import argparse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal, TextIO

import pandas as pd
import torch

from paper.experiments import run_many
from paper.models import ModelKind, ModelSpec, make_model
from paper.targets import ArrayTarget, TargetKind, TargetSpec
from paper.tasks import Task
from paper.training import run_one_stream

type TargetFactory = Callable[[TargetSpec], ArrayTarget]
type TaskFactory = Callable[..., Task]
type ProfileRunner = Callable[..., pd.DataFrame]
type WriteMode = Literal["overwrite", "append"]


RAW_COLUMNS = [
    "target_kind",
    "complexity",
    "target_seed",
    "noise_std",
    "model",
    "dim",
    "lr",
    "init_seed",
    "step",
    "val_rmse",
    "test_rmse",
]

DEFAULT_RUNS_DIR = Path("notebooks") / "runs"
WRITE_MODES: tuple[WriteMode, ...] = ("overwrite", "append")
FITS: tuple[tuple[TargetKind, ModelKind], ...] = (
    ("general", "unconstrained"),
    ("monotone", "unconstrained"),
    ("monotone", "monotone"),
)


@dataclass(frozen=True)
class Profile:
    complexities: tuple[int, ...]
    target_seeds: range
    init_seeds: range
    dims: tuple[int, ...]
    lrs: tuple[float, ...]
    budgets: tuple[int, ...]
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
    budgets: tuple[int, ...]
    val_size: int
    test_size: int


def _make_seeded_model(
    model_spec: ModelSpec, *, input_dim: int, init_seed: int
) -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(model_spec, input_dim)


def _with_metadata(df: pd.DataFrame, *, config: RunConfig) -> pd.DataFrame:
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
    return df.assign(**metadata).loc[:, RAW_COLUMNS]


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
        noise_std=config.noise_std,
    )
    model = _make_seeded_model(
        config.model_spec,
        input_dim=task.input_dim,
        init_seed=config.init_seed,
    )
    df = run_one_stream(
        task,
        model,
        lr=config.lr,
        train_seed=config.init_seed,
        checkpoints=settings.budgets,
    )
    return _with_metadata(df, config=config)


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
        budgets=profile.budgets,
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_arg_parser(profiles: Mapping[str, Profile]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=profiles.keys(), default="sanity")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-mode", choices=WRITE_MODES, default="overwrite")
    return parser


def default_raw_path(experiment_name: str, profile_name: str) -> Path:
    return DEFAULT_RUNS_DIR / f"{experiment_name}_{profile_name}.csv"


def write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    write_mode: WriteMode = "overwrite",
) -> None:
    if write_mode not in WRITE_MODES:
        raise ValueError(f"unknown write mode: {write_mode}")

    path.parent.mkdir(parents=True, exist_ok=True)
    append = write_mode == "append"
    has_content = path.exists() and path.stat().st_size > 0
    df.to_csv(
        path,
        mode="a" if append else "w",
        header=not append or not has_content,
        index=False,
    )


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
