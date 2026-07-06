import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper.models import ModelSpec, make_model
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
    ),
}


def _model_specs(dim: int) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("unconstrained", "unconstrained", dim),
        ModelSpec("monotone", "monotone", dim),
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
    target_spec: TargetSpec,
    noise_std: float,
    model_spec: ModelSpec,
    lr: float,
    init_seed: int,
) -> pd.DataFrame:
    metadata = {
        "target_kind": target_spec.kind,
        "complexity": target_spec.complexity,
        "target_seed": target_spec.seed,
        "noise_std": noise_std,
        "model": model_spec.name,
        "dim": model_spec.dim,
        "eig_idx": _resolved_eig_idx(model_spec),
        "lr": lr,
        "init_seed": init_seed,
    }
    return df.assign(**metadata).loc[:, RAW_COLUMNS]


def run_profile(
    profile: Profile,
    *,
    target_kind: TargetKind = "monotone",
    val_size: int = 4096,
    test_size: int = 4096,
) -> pd.DataFrame:
    dfs = []

    for complexity in profile.complexities:
        for target_seed in profile.target_seeds:
            target_spec = TargetSpec(
                kind=target_kind,
                complexity=complexity,
                seed=target_seed,
            )
            target = make_target(target_spec)

            for noise_std in profile.noise_stds:
                task = make_univariate_task(
                    target,
                    lower=target_spec.lower,
                    upper=target_spec.upper,
                    batch_size=profile.batch_size,
                    val_size=val_size,
                    test_size=test_size,
                    seed=target_seed,
                    noise_std=noise_std,
                )

                for dim in profile.dims:
                    for model_spec in _model_specs(dim):
                        for lr in profile.lrs:
                            for init_seed in profile.init_seeds:
                                model = _make_seeded_model(
                                    model_spec,
                                    input_dim=task.input_dim,
                                    init_seed=init_seed,
                                )

                                df = run_one_stream(
                                    task,
                                    model,
                                    lr=lr,
                                    train_seed=init_seed,
                                    steps=profile.steps,
                                    checkpoints=set(profile.budgets),
                                )
                                dfs.append(
                                    _with_metadata(
                                        df,
                                        target_spec=target_spec,
                                        noise_std=noise_std,
                                        model_spec=model_spec,
                                        lr=lr,
                                        init_seed=init_seed,
                                    )
                                )

    return pd.concat(dfs, ignore_index=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES.keys(), default="sanity")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
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
    out = args.out or Path("runs") / f"univariate_{args.profile}.csv"

    raw = run_profile(profile)
    _write_csv(raw, out, overwrite=args.overwrite)

    if args.summary_out is not None:
        summary = summarize_raw(raw, profile.budgets)
        _write_csv(summary, args.summary_out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
