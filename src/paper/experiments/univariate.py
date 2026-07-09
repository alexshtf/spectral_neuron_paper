import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from paper.models import ModelKind, ModelSpec, make_model
from paper.targets import TargetKind, TargetSpec, make_target
from paper.tasks import make_univariate_task
from paper.training import run_one_stream
from paper.tuning import summarize_raw

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
    "seconds",
]

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
                target_spec=TargetSpec(kind=fit_kind.target_kind, complexity=complexity, seed=target_seed),
                model_spec=ModelSpec.from_kind_dim(fit_kind.model_kind, dim)
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


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        complexities=(5,),
        target_seeds=range(2),
        init_seeds=range(1),
        dims=(5, 9),
        lrs=(1e-3, 1e-2),
        budgets=(1, 2, 5, 10, 30),
        batch_size=32,
    ),
    "small": Profile(
        complexities=(5, 10, 20),
        target_seeds=range(8),
        init_seeds=range(2),
        dims=(5, 9, 15),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 4).tolist()),
        budgets=(1, 2, 5, 10, 20, 50, 100, 200),
        batch_size=32,
    ),
    "full": Profile(
        complexities=(5, 10, 20),
        target_seeds=range(32),
        init_seeds=range(3),
        dims=(5, 9, 15),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 8).tolist()),
        budgets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
        batch_size=32,
        noise_stds=(0., 1e-1)
    ),
}


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


def _with_metadata(
    df: pd.DataFrame,
    *,
    config: RunConfig,
) -> pd.DataFrame:
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


def run_config(config: RunConfig, settings: RunSettings) -> pd.DataFrame:
    target = make_target(config.target_spec)
    task = make_univariate_task(
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
    if workers == 1:
        items = tqdm(
            configs,
            total=len(configs),
            unit="experiment",
            disable=not progress,
            file=progress_file,
        )
        dfs = [run_config(config, settings) for config in items]
    else:
        dfs = process_map(
            partial(run_config, settings=settings),
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


def _default_raw_path(profile_name: str) -> Path:
    return DEFAULT_RUNS_DIR / f"univariate_{profile_name}.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES.keys(), default="sanity")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_csv(df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    out = args.out or _default_raw_path(args.profile)

    raw = run_profile(profile, workers=args.workers, progress=not args.quiet)
    _write_csv(raw, out, overwrite=args.overwrite)

    if args.summary_out is not None:
        summary = summarize_raw(raw, profile.budgets)
        _write_csv(summary, args.summary_out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
