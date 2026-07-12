import argparse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import TextIO

import pandas as pd
import torch
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from paper.models import ModelKind, ModelSpec, make_model
from paper.targets import ArrayTarget, TargetKind, TargetSpec
from paper.tasks import Task
from paper.training import run_one_stream
from paper.tuning import summarize_raw

type TargetFactory = Callable[[TargetSpec], ArrayTarget]
type TaskFactory = Callable[..., Task]
type ProfileRunner = Callable[..., pd.DataFrame]
type RawPathFactory = Callable[[str], Path]


RAW_COLUMNS = [
    "target_kind",
    "complexity",
    "target_seed",
    "noise_std",
    "model",
    "dim",
    "eig_idx",
    "lr",
    "init_seed",
    "step",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "elapsed_seconds",
]

WRITE_MODES = ("overwrite", "append")
DEFAULT_RUNS_DIR = Path("notebooks") / "runs"


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

    @property
    def steps(self) -> int:
        return max(self.budgets)


@dataclass(frozen=True)
class FitKind:
    target_kind: TargetKind
    model_kind: ModelKind


@dataclass(frozen=True)
class FitSpec:
    target_spec: TargetSpec
    model_spec: ModelSpec


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

    @property
    def fit_specs(self) -> tuple[FitSpec, ...]:
        return tuple(
            FitSpec(
                target_spec=TargetSpec(
                    kind=fit_kind.target_kind,
                    complexity=complexity,
                    seed=target_seed,
                ),
                model_spec=ModelSpec.from_kind_dim(fit_kind.model_kind, dim),
            )
            for dim in self.profile.dims
            for complexity in self.profile.complexities
            for target_seed in self.profile.target_seeds
            for fit_kind in _fit_kinds()
        )

    def __len__(self) -> int:
        return (
            len(self.fit_specs)
            * len(self.profile.noise_stds)
            * len(self.profile.lrs)
            * len(self.profile.init_seeds)
        )

    def __iter__(self) -> Iterator[RunConfig]:
        for fit_spec, noise_std, lr, init_seed in product(
            self.fit_specs,
            self.profile.noise_stds,
            self.profile.lrs,
            self.profile.init_seeds,
        ):
            yield RunConfig(
                target_spec=fit_spec.target_spec,
                noise_std=noise_std,
                model_spec=fit_spec.model_spec,
                lr=lr,
                init_seed=init_seed,
            )


@dataclass(frozen=True)
class RunSettings:
    batch_size: int
    steps: int
    budgets: tuple[int, ...]
    val_size: int
    test_size: int


def _fit_kinds() -> tuple[FitKind, ...]:
    return (
        FitKind(target_kind="general", model_kind="unconstrained"),
        FitKind(target_kind="monotone", model_kind="unconstrained"),
        FitKind(target_kind="monotone", model_kind="monotone"),
    )


def _resolved_eig_idx(spec: ModelSpec) -> int:
    return spec.dim // 2 if spec.eig_idx is None else spec.eig_idx


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
        "model": config.model_spec.name,
        "dim": config.model_spec.dim,
        "eig_idx": _resolved_eig_idx(config.model_spec),
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
        steps=settings.steps,
        checkpoints=set(settings.budgets),
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
    if workers < 1:
        raise ValueError(f"workers must be positive; got {workers}")

    configs = RunGrid(profile)
    settings = RunSettings(
        batch_size=profile.batch_size,
        steps=profile.steps,
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
    if workers == 1:
        items = tqdm(
            configs,
            total=len(configs),
            unit="experiment",
            disable=not progress,
            file=progress_file,
        )
        dfs = [run(config) for config in items]
    else:
        dfs = process_map(
            run,
            configs,
            max_workers=workers,
            chunksize=1,
            unit="experiment",
            disable=not progress,
            file=progress_file,
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
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--write-mode",
        choices=WRITE_MODES,
        default="overwrite",
        help="How to write output files (default: overwrite).",
    )
    return parser


def default_raw_path(experiment_name: str, profile_name: str) -> Path:
    return DEFAULT_RUNS_DIR / f"{experiment_name}_{profile_name}.csv"


def _write_csv(df: pd.DataFrame, path: Path, *, write_mode: str) -> None:
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
    profiles: Mapping[str, Profile],
    run_profile: ProfileRunner,
    *,
    default_raw_path: RawPathFactory,
    argv: list[str] | None = None,
) -> None:
    parser = build_arg_parser(profiles)
    args = parser.parse_args(argv)

    profile = profiles[args.profile]
    out = args.out or default_raw_path(args.profile)
    raw = run_profile(profile, workers=args.workers, progress=not args.quiet)
    _write_csv(raw, out, write_mode=args.write_mode)

    if args.summary_out is not None:
        summary = summarize_raw(raw, profile.budgets)
        _write_csv(summary, args.summary_out, write_mode=args.write_mode)
