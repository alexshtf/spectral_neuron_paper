import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from paper.experiments import run_many
from paper.experiments.scaling import (
    RunConfig,
    SeedGrid,
    SelectedRun,
    best_lrs,
    select_lr as select_scaling_lr,
    selected_runs,
    summarize_scaling,
    tuning_configs,
)
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, WRITE_MODES, write_csv
from paper.models import FactorizationMachine, SparseKthEigval, SparseLinear
from paper.movielens import MovieLensCorpus, MovieLensTask, prepare_corpus
from paper.shuffling import resolve_train_sizes
from paper.training import (
    REGRESSION_OBJECTIVE,
    fit_and_test_scaling,
    tune_scaling_stream,
)


type Variant = Literal["linear", "fm", "spectral"]

VARIANTS: tuple[Variant, ...] = ("linear", "fm", "spectral")
NUM_FIELDS = 2
PROTOCOL = "repeated_shuffle"
OPTIMIZER = "adam+sparseadam"

IDENTITY_COLUMNS = [
    "protocol",
    "optimizer",
    "split_seed",
    "train_pool_size",
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
]
METRIC_COLUMNS = [
    "val_rmse",
    "val_warm_fraction",
    "test_rmse",
    "test_warm_fraction",
]
RAW_COLUMNS = IDENTITY_COLUMNS + METRIC_COLUMNS
_TIMING_COLUMNS = ["train_seconds", "val_seconds", "test_seconds"]

EXPERIMENT_COLUMNS = ["protocol", "optimizer", "split_seed", "train_pool_size"]
MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "model",
    "dim",
    "rank",
    "parameters_per_identity",
    "num_parameters",
]
CURVE_COLUMNS = MODEL_COLUMNS + ["train_size"]


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
        train_sizes=(2**20, 2**21, 2**22, 2**23, 2**24),
        dims=(3, 7, 11),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 8).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(8)),
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
) -> tuple[RunConfig[MovieLensModelSpec], ...]:
    return tuning_configs(specs, profile.lrs, profile.tuning_seeds)


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
    config: RunConfig[MovieLensModelSpec], settings: RunSettings
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
    config: RunConfig[MovieLensModelSpec],
    settings: RunSettings,
    model: nn.Module,
) -> dict[str, int | float | str]:
    spec = config.model_spec
    return {
        "protocol": PROTOCOL,
        "optimizer": OPTIMIZER,
        "split_seed": settings.corpus.split_seed,
        "train_pool_size": settings.corpus.train_rows,
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
    config: RunConfig[MovieLensModelSpec],
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


def run_config(
    config: RunConfig[MovieLensModelSpec], settings: RunSettings
) -> pd.DataFrame:
    task, model = _make_task_model(config, settings)
    result = tune_scaling_stream(
        task,
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
    )
    return _format_result(
        result,
        phase="tuning",
        config=config,
        settings=settings,
        model=model,
    )


def run_selected(
    selected: SelectedRun[MovieLensModelSpec], settings: RunSettings
) -> pd.DataFrame:
    task, model = _make_task_model(selected.config, settings)
    checkpoints = tuple(
        size for size in settings.train_sizes if size <= max(selected.train_sizes)
    )
    result = fit_and_test_scaling(
        task,
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=selected.config.lr,
        checkpoints=checkpoints,
        test_checkpoints=selected.train_sizes,
    )
    return _format_result(
        result,
        phase="evaluation",
        config=selected.config,
        settings=settings,
        model=model,
    )


def _best_lrs(tuning: pd.DataFrame) -> pd.DataFrame:
    return best_lrs(
        tuning,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_rmse",
    )


def _selected_runs(
    tuning: pd.DataFrame, evaluation_seeds: SeedGrid
) -> tuple[SelectedRun[MovieLensModelSpec], ...]:
    return selected_runs(
        tuning,
        experiment_columns=EXPERIMENT_COLUMNS,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_rmse",
        evaluation_seeds=evaluation_seeds,
        make_model_spec=MovieLensModelSpec,
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
    train_sizes = resolve_train_sizes(
        profile.train_sizes,
        batch_size=profile.batch_size,
    )

    variants = (variant,) if variant is not None else VARIANTS
    specs = _model_specs(profile, variants)
    configs = _tuning_configs(profile, specs)
    data_seeds = {
        *profile.tuning_seeds.data_seeds,
        *profile.evaluation_seeds.data_seeds,
    }
    required_passes = (train_sizes[-1] + corpus.train_rows - 1) // corpus.train_rows
    for data_seed in data_seeds:
        corpus.shuffled_epochs(data_seed).prepare(required_passes)
    warm_coverage = {
        data_seed: MovieLensTask(corpus, data_seed, profile.batch_size).warm_coverage(
            train_sizes
        )
        for data_seed in data_seeds
    }

    settings = RunSettings(
        train_sizes=train_sizes,
        batch_size=profile.batch_size,
        corpus=corpus,
        warm_coverage=warm_coverage,
        threads_per_worker=1 if workers > 1 else None,
    )
    tuning_results = run_many(
        partial(run_config, settings=settings),
        configs,
        workers=workers,
        desc="Tuning (train + validation)",
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
        _selected_runs(tuning, profile.evaluation_seeds),
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
    return select_scaling_lr(
        raw,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_rmse",
    )


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_scaling(
        raw,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_rmse",
        quantile_metrics=("test_rmse",),
        median_metrics=("test_warm_fraction",),
    )


def _same_lrs(actual: pd.Series, expected: tuple[float, ...]) -> bool:
    values = np.sort(actual.unique())
    return len(values) == len(expected) and np.allclose(
        values, expected, rtol=1e-12, atol=0
    )


def _capacity_num_features(spec: MovieLensModelSpec, num_parameters: int) -> int:
    if isinstance(num_parameters, (bool, np.bool_)) or not isinstance(
        num_parameters, (int, np.integer)
    ):
        raise ValueError
    if spec.variant == "linear":
        num_features, remainder = num_parameters - 1, 0
    elif spec.variant == "fm":
        num_features, remainder = divmod(
            num_parameters - 1, spec.parameters_per_identity
        )
    else:
        matrices, remainder = divmod(num_parameters, spec.parameters_per_identity)
        num_features = matrices - 1
    if num_features <= 0 or remainder:
        raise ValueError
    return num_features


def validate_raw(
    raw: pd.DataFrame, profile: Profile, variant: Variant | None = None
) -> None:
    """Validate that raw results are a complete run of a MovieLens profile."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("incompatible MovieLens result schema")
    if raw[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError(
            "MovieLens run identity columns must not contain missing values"
        )
    if set(raw["protocol"]) != {PROTOCOL}:
        raise ValueError(f"expected protocol={PROTOCOL!r}")
    if set(raw["optimizer"]) != {OPTIMIZER}:
        raise ValueError(f"expected optimizer={OPTIMIZER!r}")
    if set(raw["split_seed"]) != {profile.split_seed}:
        raise ValueError(f"expected split_seed={profile.split_seed}")

    train_pool_sizes = raw["train_pool_size"].unique()
    if (
        len(train_pool_sizes) != 1
        or isinstance(train_pool_sizes[0], (bool, np.bool_))
        or not isinstance(train_pool_sizes[0], (int, np.integer))
        or train_pool_sizes[0] <= 0
    ):
        raise ValueError("results must contain one positive integer train_pool_size")
    train_pool_size = int(train_pool_sizes[0])
    if set(raw["phase"]) != {"tuning", "evaluation"}:
        raise ValueError("results must contain tuning and evaluation phases")
    if raw.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("results contain duplicate trajectory checkpoints")

    train_sizes = resolve_train_sizes(
        profile.train_sizes,
        batch_size=profile.batch_size,
    )

    variants = (variant,) if variant is not None else VARIANTS
    specs = _model_specs(profile, variants)
    expected_specs = {(spec.variant, spec.dim) for spec in specs}
    observed_specs = set(
        raw[["model", "dim"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_specs != expected_specs:
        raise ValueError(
            f"model/capacity grid mismatch: expected {sorted(expected_specs)}, "
            f"got {sorted(observed_specs)}"
        )

    num_features = set()
    for spec in specs:
        rows = raw.loc[
            (raw["model"] == spec.variant) & (raw["dim"] == spec.dim),
            ["rank", "parameters_per_identity", "num_parameters"],
        ].drop_duplicates()
        if (
            len(rows) != 1
            or tuple(rows.iloc[0, :2])
            != (spec.rank, spec.parameters_per_identity)
        ):
            raise ValueError(f"inconsistent capacity metadata for {spec}")
        try:
            num_features.add(_capacity_num_features(spec, rows.iloc[0, 2]))
        except ValueError:
            raise ValueError(f"inconsistent capacity metadata for {spec}") from None
    if len(num_features) != 1:
        raise ValueError("capacity metadata implies inconsistent feature counts")

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    models = raw[MODEL_COLUMNS].drop_duplicates()
    expected_curves = {
        (*model, train_size)
        for model in models.itertuples(index=False, name=None)
        for train_size in train_sizes
    }
    for phase, rows in (("tuning", tuning), ("evaluation", evaluation)):
        observed_curves = set(
            rows[CURVE_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
        )
        if observed_curves != expected_curves:
            raise ValueError(f"{phase} has an incomplete model/checkpoint grid")

    if tuning[["test_rmse", "test_warm_fraction"]].notna().any().any():
        raise ValueError("tuning rows must not contain test metrics")
    if evaluation[["val_rmse", "val_warm_fraction"]].notna().any().any():
        raise ValueError("evaluation rows must not contain validation metrics")

    if not np.isfinite(evaluation["test_rmse"].to_numpy(dtype=float)).all():
        raise ValueError("evaluation test RMSE must be finite")
    if (tuning["val_rmse"] < 0).any() or (evaluation["test_rmse"] < 0).any():
        raise ValueError("validation and test RMSE must be nonnegative")
    for phase, rows, warm in (
        ("tuning", tuning, "val_warm_fraction"),
        ("evaluation", evaluation, "test_warm_fraction"),
    ):
        if not np.isfinite(rows[warm].to_numpy(dtype=float)).all():
            raise ValueError(f"{phase} warm fractions must be finite")
        if not rows[warm].between(0, 1).all():
            raise ValueError(f"{phase} warm fractions are out of range")

    if not tuning.loc[
        tuning["train_size"] >= train_pool_size, "val_warm_fraction"
    ].eq(1.0).all():
        raise ValueError("validation warm coverage must saturate after one pass")
    if not evaluation.loc[
        evaluation["train_size"] >= train_pool_size, "test_warm_fraction"
    ].eq(1.0).all():
        raise ValueError("test warm coverage must saturate after one pass")

    if not _same_lrs(tuning["lr"], profile.lrs):
        raise ValueError("tuning learning-rate grid does not match the profile")
    tuning_seeds = set(profile.tuning_seeds)
    for curve, rows in tuning.groupby(CURVE_COLUMNS):
        if not _same_lrs(rows["lr"], profile.lrs):
            raise ValueError(f"incomplete tuning learning-rate grid for {curve}")
        for lr, lr_rows in rows.groupby("lr"):
            seeds = set(
                lr_rows[["data_seed", "init_seed"]].itertuples(
                    index=False, name=None
                )
            )
            if seeds != tuning_seeds:
                raise ValueError(f"incomplete tuning seeds for {curve}, lr={lr:g}")

    selected_lrs = {
        tuple(getattr(row, column) for column in CURVE_COLUMNS): row.selected_lr
        for row in _best_lrs(tuning).itertuples(index=False)
    }
    evaluation_seeds = set(profile.evaluation_seeds)
    for curve, rows in evaluation.groupby(CURVE_COLUMNS):
        seeds = set(
            rows[["data_seed", "init_seed"]].itertuples(index=False, name=None)
        )
        if seeds != evaluation_seeds:
            raise ValueError(f"incomplete evaluation seeds for {curve}")
        lrs = rows["lr"].unique()
        if len(lrs) != 1 or not np.isclose(
            lrs[0], selected_lrs[curve], rtol=1e-12, atol=0
        ):
            raise ValueError(f"evaluation does not use the selected LR for {curve}")


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return (
        DEFAULT_RUNS_DIR
        / f"movielens_scaling_{profile_name}_repeated_shuffle{suffix}.csv"
    )


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
