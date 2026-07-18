import argparse
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from paper.experiments import run_many
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, WRITE_MODES, write_csv
from paper.models import FactorizationMachine, SparseKthEigval, SparseLinear
from paper.movielens import MovieLensCorpus, MovieLensTask, prepare_corpus
from paper.training import (
    REGRESSION_OBJECTIVE,
    fit_and_test_scaling,
    tune_scaling_stream,
)


type Variant = Literal["linear", "fm", "spectral"]

VARIANTS: tuple[Variant, ...] = ("linear", "fm", "spectral")
NUM_FIELDS = 2
PROTOCOL = "one_pass_random_prefix"
OPTIMIZER = "adam+sparseadam"

RAW_COLUMNS = [
    "protocol",
    "optimizer",
    "split_seed",
    "phase",
    "train_size",
    "data_seed",
    "model",
    "dim",
    "rank",
    "parameters_per_identity",
    "num_parameters",
    "lr",
    "init_seed",
    "val_rmse",
    "val_warm_fraction",
    "test_rmse",
    "test_warm_fraction",
]
_TIMING_COLUMNS = ["train_seconds", "val_seconds", "test_seconds"]

EXPERIMENT_COLUMNS = ["protocol", "optimizer", "split_seed"]
MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "model",
    "dim",
    "rank",
    "parameters_per_identity",
    "num_parameters",
]
CURVE_COLUMNS = MODEL_COLUMNS + ["train_size"]


@dataclass(frozen=True)
class SeedGrid:
    data_seeds: range = range(1)
    init_seeds: range = range(1)

    def __len__(self) -> int:
        return len(self.data_seeds) * len(self.init_seeds)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return product(self.data_seeds, self.init_seeds)


@dataclass(frozen=True)
class Profile:
    train_sizes: tuple[int, ...]
    dims: tuple[int, ...]
    lrs: tuple[float, ...]
    tuning_seeds: SeedGrid
    evaluation_seeds: SeedGrid
    batch_size: int = 4096
    split_seed: int = 0

    def __post_init__(self) -> None:
        if (
            not self.train_sizes
            or self.train_sizes != tuple(sorted(set(self.train_sizes)))
            or self.train_sizes[0] <= 0
        ):
            raise ValueError("train_sizes must be positive, unique, and increasing")
        if (
            not self.dims
            or self.dims != tuple(sorted(set(self.dims)))
            or self.dims[0] <= 0
        ):
            raise ValueError("dims must be positive, unique, and increasing")
        if (
            not self.lrs
            or self.lrs != tuple(sorted(set(self.lrs)))
            or not np.isfinite(self.lrs).all()
            or self.lrs[0] <= 0
        ):
            raise ValueError("lrs must be finite, positive, unique, and increasing")
        if not self.tuning_seeds or not self.evaluation_seeds:
            raise ValueError("tuning and evaluation seed grids must be non-empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        train_sizes=(2**10,),
        dims=(3,),
        lrs=(1e-2,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=256,
    ),
    "small": Profile(
        train_sizes=(2**18, 2**20, 2**22),
        dims=(3, 7),
        lrs=(1e-3, 1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 3), init_seeds=range(2, 4)
        ),
    ),
    "full": Profile(
        train_sizes=(2**20, 2**21, 2**22, 2**23, 15_800_000),
        dims=(3, 7, 11),
        lrs=tuple(np.geomspace(1e-3, 1e-1, 6).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(4)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 3), init_seeds=range(4, 8)
        ),
    ),
}


@dataclass(frozen=True)
class MovieLensModelSpec:
    variant: Variant
    dim: int = 0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.variant == "linear":
            if self.dim != 0:
                raise ValueError("linear model requires dim=0")
        elif self.dim <= 0:
            raise ValueError(f"{self.variant} model requires a positive dim")

    @property
    def parameters_per_identity(self) -> int:
        return 1 if self.variant == "linear" else self.dim * (self.dim + 1) // 2

    @property
    def rank(self) -> int:
        return self.parameters_per_identity - 1 if self.variant == "fm" else 0


@dataclass(frozen=True)
class RunConfig:
    data_seed: int
    model_spec: MovieLensModelSpec
    lr: float
    init_seed: int


@dataclass(frozen=True)
class RunSettings:
    train_sizes: tuple[int, ...]
    batch_size: int
    corpus: MovieLensCorpus
    warm_coverage: dict[int, dict[int, tuple[float, float]]]
    threads_per_worker: int | None


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[MovieLensModelSpec, ...]:
    specs = [MovieLensModelSpec("linear")]
    specs.extend(
        MovieLensModelSpec(variant, dim)
        for dim in profile.dims
        for variant in ("fm", "spectral")
    )
    return tuple(spec for spec in specs if spec.variant in variants)


def _tuning_configs(
    profile: Profile, specs: tuple[MovieLensModelSpec, ...]
) -> tuple[RunConfig, ...]:
    return tuple(
        RunConfig(data_seed, spec, lr, init_seed)
        for data_seed, spec, lr, init_seed in product(
            profile.tuning_seeds.data_seeds,
            specs,
            profile.lrs,
            profile.tuning_seeds.init_seeds,
        )
    )


def make_model(spec: MovieLensModelSpec, num_features: int) -> nn.Module:
    match spec.variant:
        case "linear":
            return SparseLinear(num_features, NUM_FIELDS)
        case "fm":
            return FactorizationMachine(num_features, NUM_FIELDS, spec.rank)
        case "spectral":
            return SparseKthEigval(num_features, NUM_FIELDS, spec.dim)
        case _:
            raise ValueError(spec.variant)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _make_seeded_model(
    spec: MovieLensModelSpec, *, num_features: int, init_seed: int
) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(spec, num_features)


def _make_task_model(
    config: RunConfig, settings: RunSettings
) -> tuple[MovieLensTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)
    task = MovieLensTask(settings.corpus, config.data_seed, settings.batch_size)
    model = _make_seeded_model(
        config.model_spec,
        num_features=settings.corpus.num_features,
        init_seed=config.init_seed,
    )
    return task, model


def _metadata(
    config: RunConfig, settings: RunSettings, model: nn.Module
) -> dict[str, int | float | str]:
    spec = config.model_spec
    return {
        "protocol": PROTOCOL,
        "optimizer": OPTIMIZER,
        "split_seed": settings.corpus.split_seed,
        "data_seed": config.data_seed,
        "model": spec.variant,
        "dim": spec.dim,
        "rank": spec.rank,
        "parameters_per_identity": spec.parameters_per_identity,
        "num_parameters": trainable_parameter_count(model),
        "lr": config.lr,
        "init_seed": config.init_seed,
    }


def _format_result(
    result: pd.DataFrame,
    *,
    phase: Literal["tuning", "evaluation"],
    config: RunConfig,
    settings: RunSettings,
    model: nn.Module,
) -> pd.DataFrame:
    coverage = settings.warm_coverage[config.data_seed]
    holdout = 0 if phase == "tuning" else 1
    column = "val_warm_fraction" if phase == "tuning" else "test_warm_fraction"
    return result.assign(
        phase=phase,
        **{column: result["train_size"].map(lambda size: coverage[size][holdout])},
        **_metadata(config, settings, model),
    ).reindex(columns=RAW_COLUMNS + _TIMING_COLUMNS)


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


def run_config(config: RunConfig, settings: RunSettings) -> pd.DataFrame:
    task, model = _make_task_model(config, settings)
    result = tune_scaling_stream(
        task,
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
        validation_checkpoints=(settings.train_sizes[-1],),
    )
    return _format_result(
        result,
        phase="tuning",
        config=config,
        settings=settings,
        model=model,
    )


def run_selected(config: RunConfig, settings: RunSettings) -> pd.DataFrame:
    task, model = _make_task_model(config, settings)
    result = fit_and_test_scaling(
        task,
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
        test_checkpoints=settings.train_sizes,
    )
    return _format_result(
        result,
        phase="evaluation",
        config=config,
        settings=settings,
        model=model,
    )


def _best_lrs(tuning: pd.DataFrame) -> pd.DataFrame:
    if tuning.empty:
        raise ValueError("tuning results must not be empty")
    model_keys = tuning.loc[:, MODEL_COLUMNS].drop_duplicates()
    finite = tuning.loc[np.isfinite(tuning["val_rmse"])]
    scores = (
        finite.groupby(MODEL_COLUMNS + ["lr"], as_index=False)["val_rmse"]
        .median()
        .rename(columns={"val_rmse": "median_val_rmse"})
    )
    available = scores.loc[:, MODEL_COLUMNS].drop_duplicates()
    missing = model_keys.merge(
        available, on=MODEL_COLUMNS, how="left", indicator=True
    )
    missing = missing.loc[missing["_merge"] == "left_only", MODEL_COLUMNS]
    if not missing.empty:
        families = missing[["model", "dim"]].to_dict("records")
        raise ValueError(f"no finite validation RMSE for {families}")

    best = (
        scores.sort_values(
            MODEL_COLUMNS + ["median_val_rmse", "lr"], kind="mergesort"
        )
        .groupby(MODEL_COLUMNS, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )
    return best[MODEL_COLUMNS + ["selected_lr", "median_val_rmse"]]


def _selected_configs(
    tuning: pd.DataFrame, evaluation_seeds: SeedGrid
) -> tuple[RunConfig, ...]:
    return tuple(
        RunConfig(
            data_seed=data_seed,
            model_spec=MovieLensModelSpec(row.model, int(row.dim)),
            lr=row.selected_lr,
            init_seed=init_seed,
        )
        for row in _best_lrs(tuning).itertuples(index=False)
        for data_seed, init_seed in evaluation_seeds
    )


def run_profile(
    profile: Profile,
    *,
    raw_path: Path,
    cache_dir: Path,
    chunk_size: int = 1_000_000,
    workers: int = 1,
    variant: Variant | None = None,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError(f"workers must be positive; got {workers}")
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        split_seed=profile.split_seed,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    if max(profile.train_sizes) > corpus.train_rows:
        raise ValueError(
            f"profile requests {max(profile.train_sizes):,} training ratings, "
            f"but the training split contains {corpus.train_rows:,}"
        )

    variants = (variant,) if variant is not None else VARIANTS
    specs = _model_specs(profile, variants)
    configs = _tuning_configs(profile, specs)
    data_seeds = {
        *profile.tuning_seeds.data_seeds,
        *profile.evaluation_seeds.data_seeds,
    }
    warm_coverage = {
        data_seed: MovieLensTask(corpus, data_seed, profile.batch_size).warm_coverage(
            profile.train_sizes
        )
        for data_seed in data_seeds
    }

    settings = RunSettings(
        train_sizes=profile.train_sizes,
        batch_size=profile.batch_size,
        corpus=corpus,
        warm_coverage=warm_coverage,
        threads_per_worker=1 if workers > 1 else None,
    )
    tuning_results = run_many(
        partial(run_config, settings=settings),
        configs,
        workers=workers,
        desc="Tuning (train + final validation)",
        unit="trajectory",
        progress=progress,
        progress_file=progress_file,
    )
    if not tuning_results:
        return pd.DataFrame(columns=RAW_COLUMNS)
    if progress:
        _report_timings("Tuning", "val", tuning_results, progress_file)
    tuning = pd.concat(tuning_results, ignore_index=True)

    evaluation_results = run_many(
        partial(run_selected, settings=settings),
        _selected_configs(tuning, profile.evaluation_seeds),
        workers=workers,
        desc="Evaluation (retrain + test)",
        unit="trajectory",
        progress=progress,
        progress_file=progress_file,
    )
    if progress:
        _report_timings("Evaluation", "test", evaluation_results, progress_file)

    evaluation = pd.concat(evaluation_results, ignore_index=True)
    return pd.concat((tuning, evaluation), ignore_index=True).loc[:, RAW_COLUMNS]


def select_lr(raw: pd.DataFrame) -> pd.DataFrame:
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    best = _best_lrs(tuning)
    selected = evaluation.merge(best, on=MODEL_COLUMNS, how="inner")
    return selected.loc[selected["lr"] == selected["selected_lr"]].reset_index(
        drop=True
    )


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    selected = select_lr(raw)
    return (
        selected.groupby(CURVE_COLUMNS + ["selected_lr"])
        .agg(
            median_test_rmse=("test_rmse", "median"),
            q25_test_rmse=("test_rmse", lambda s: s.quantile(0.25)),
            q75_test_rmse=("test_rmse", lambda s: s.quantile(0.75)),
            median_test_warm_fraction=("test_warm_fraction", "median"),
            n=("test_rmse", "size"),
        )
        .reset_index()
    )


def validate_raw(
    raw: pd.DataFrame, profile: Profile, variant: Variant | None = None
) -> None:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("results do not have the MovieLens raw schema")
    if set(raw["phase"]) != {"tuning", "evaluation"}:
        raise ValueError("results must contain tuning and evaluation phases")

    variants = (variant,) if variant is not None else VARIANTS
    expected = {(spec.variant, spec.dim) for spec in _model_specs(profile, variants)}
    observed = set(
        raw[["model", "dim"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(f"expected model grid {sorted(expected)}; got {sorted(observed)}")

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    if set(tuning["train_size"]) != {profile.train_sizes[-1]}:
        raise ValueError("tuning must evaluate only the final checkpoint")
    if set(evaluation["train_size"]) != set(profile.train_sizes):
        raise ValueError("evaluation must contain every checkpoint")
    if tuning["test_rmse"].notna().any():
        raise ValueError("tuning rows must not contain test metrics")
    if tuning["test_warm_fraction"].notna().any():
        raise ValueError("tuning rows must not contain test diagnostics")
    if evaluation[["val_rmse", "val_warm_fraction"]].notna().any().any():
        raise ValueError("evaluation rows must not contain validation metrics")


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / f"movielens_scaling_{profile_name}{suffix}.csv"


def build_arg_parser(
    profiles: Mapping[str, Profile] = PROFILES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="MovieLens ratings.csv, its directory, or the official ZIP.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--profile", choices=profiles.keys(), default="sanity")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-mode", choices=WRITE_MODES, default="overwrite")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    profile = PROFILES[args.profile]
    cache_dir = args.cache_dir or (
        args.data / ".paper-cache-v1"
        if args.data.is_dir()
        else args.data.with_name(f".{args.data.name}.cache-v1")
    )
    raw = run_profile(
        profile,
        raw_path=args.data,
        cache_dir=cache_dir,
        chunk_size=args.chunk_size,
        workers=args.workers,
        variant=args.variant,
        progress=not args.quiet,
    )
    write_csv(
        raw,
        args.out or default_raw_path(args.profile, args.variant),
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
