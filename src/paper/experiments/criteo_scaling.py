import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn

from paper.criteo import (
    NUM_FIELDS,
    CriteoTask,
    EncodedData,
    PreprocessingKind,
    default_cache_dir,
    fit_preprocessors,
    prepare_corpus,
    prepare_encoded_data,
)
from paper.experiments import scaling
from paper.experiments.results import DEFAULT_RUNS_DIR, WRITE_MODES, write_csv
from paper.experiments.scaling import (
    PROTOCOL,
    RunConfig,
    ScalingRunner,
    ScalingSchema,
    SeedGrid,
    run_tuning_and_evaluation,
    tuning_configs,
)
from paper.models import (
    FactorizationMachine,
    SparseLinear,
    SparseMiddleEigval,
    make_seeded_model,
    matched_fm_rank,
)
from paper.shuffling import ShuffledEpochs
from paper.training import BINARY_OBJECTIVE, training_checkpoints


type Variant = Literal[
    "linear-bucketed",
    "linear-continuous",
    "fm",
    "spectral-bucketed",
    "spectral-continuous",
]

VARIANTS: tuple[Variant, ...] = (
    "linear-bucketed",
    "linear-continuous",
    "fm",
    "spectral-bucketed",
    "spectral-continuous",
)

OPTIMIZER = "adam+sparseadam"

RESULT_SCHEMA = ScalingSchema(
    experiment_columns=(
        "protocol",
        "optimizer",
        "preprocessor_sample_size",
        "preprocessor_seed",
        "train_pool_size",
    ),
    model_columns=("model", "dim"),
    model_spec_columns=("model", "dim"),
    validation_metric="val_logloss",
    test_metrics=("test_logloss", "test_brier"),
)


@dataclass(frozen=True)
class Profile:
    train_sizes: tuple[int, ...]
    capacity_dims: tuple[int, ...]
    lrs: tuple[float, ...]
    tuning_seeds: SeedGrid
    evaluation_seeds: SeedGrid
    batch_size: int = 4096
    preprocessor_fraction: float = 0.1
    preprocessor_seed: int = 0
    min_count: int = 10


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        train_sizes=(2**10,),
        capacity_dims=(3,),
        lrs=(1e-2,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=256,
        min_count=10,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        capacity_dims=(3, 5),
        lrs=(1e-3, 1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(init_seeds=range(2)),
        batch_size=4096,
    ),
    "full": Profile(
        train_sizes=tuple(2**power for power in range(12, 29, 2)),
        capacity_dims=(3, 7, 11),
        lrs=tuple(np.geomspace(1e-3, 1e-1, 8).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(8)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 5),
            init_seeds=range(3, 9),
        ),
        batch_size=4096,
    ),
}


@dataclass(frozen=True)
class CriteoModelSpec:
    variant: Variant
    capacity_dim: int | None = None

    def __post_init__(self) -> None:
        linear = self.variant in ("linear-bucketed", "linear-continuous")
        if linear != (self.capacity_dim is None):
            raise ValueError("only linear models omit capacity_dim")
        if self.capacity_dim is not None and self.capacity_dim <= 0:
            raise ValueError("capacity_dim must be positive")

    @property
    def result_dim(self) -> int:
        """Return the dimension used by the persisted result schema."""
        return 0 if self.capacity_dim is None else self.capacity_dim

    @property
    def preprocessing(self) -> PreprocessingKind:
        if self.variant in ("linear-continuous", "spectral-continuous"):
            return "hybrid"
        return "bucket"


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[CriteoModelSpec, ...]:
    specs = [
        CriteoModelSpec("linear-bucketed"),
        CriteoModelSpec("linear-continuous"),
    ]
    specs.extend(
        CriteoModelSpec(variant, capacity_dim)
        for capacity_dim in profile.capacity_dims
        for variant in ("fm", "spectral-bucketed", "spectral-continuous")
    )
    return tuple(spec for spec in specs if spec.variant in variants)


@dataclass(frozen=True)
class RunSettings:
    batch_size: int
    encoded_data: dict[PreprocessingKind, EncodedData]
    orders: dict[int, ShuffledEpochs]
    train_pool_size: int
    preprocessor_sample_size: int
    preprocessor_seed: int
    threads_per_worker: int | None


def make_model(spec: CriteoModelSpec, num_features: int) -> nn.Module:
    match spec.variant:
        case "linear-bucketed" | "linear-continuous":
            return SparseLinear(num_features, NUM_FIELDS)
        case "fm":
            assert spec.capacity_dim is not None
            return FactorizationMachine(
                num_features,
                NUM_FIELDS,
                matched_fm_rank(spec.capacity_dim),
            )
        case "spectral-bucketed" | "spectral-continuous":
            assert spec.capacity_dim is not None
            return SparseMiddleEigval(
                num_features,
                NUM_FIELDS,
                spec.capacity_dim,
            )
        case _:
            raise ValueError(spec.variant)


def _make_task_model(
    config: RunConfig[CriteoModelSpec], settings: RunSettings
) -> tuple[CriteoTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)

    preprocessing = config.model_spec.preprocessing
    data = settings.encoded_data[preprocessing]
    task = CriteoTask(data, settings.orders[config.data_seed], settings.batch_size)
    model = make_seeded_model(
        partial(make_model, config.model_spec, data.num_features),
        seed=config.init_seed,
    )
    return task, model


def _metadata(
    config: RunConfig[CriteoModelSpec],
    _model: nn.Module,
    settings: RunSettings,
) -> dict[str, int | float | str]:
    return {
        "protocol": PROTOCOL,
        "optimizer": OPTIMIZER,
        "preprocessor_sample_size": settings.preprocessor_sample_size,
        "preprocessor_seed": settings.preprocessor_seed,
        "train_pool_size": settings.train_pool_size,
        "data_seed": config.data_seed,
        "model": config.model_spec.variant,
        "dim": config.model_spec.result_dim,
        "lr": config.lr,
        "init_seed": config.init_seed,
    }


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
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    train_sizes = training_checkpoints(
        profile.train_sizes,
        batch_size=profile.batch_size,
    )

    sample_size = max(1, round(profile.preprocessor_fraction * corpus.train_stop))
    variants = (variant,) if variant is not None else VARIANTS
    model_specs = _model_specs(profile, variants)
    configs = tuning_configs(model_specs, profile.lrs, profile.tuning_seeds)
    preprocessing_kinds = tuple(
        dict.fromkeys(spec.preprocessing for spec in model_specs)
    )
    preprocessor_paths = fit_preprocessors(
        corpus,
        preprocessing_kinds,
        sample_size=sample_size,
        sample_seed=profile.preprocessor_seed,
        min_count=profile.min_count,
        progress=progress,
        progress_file=progress_file,
    )
    data_seeds = tuple(
        sorted(
            set(profile.tuning_seeds.data_seeds)
            | set(profile.evaluation_seeds.data_seeds)
        )
    )
    orders = {seed: corpus.shuffled_epochs(seed) for seed in data_seeds}
    required_passes = (train_sizes[-1] + corpus.train_stop - 1) // corpus.train_stop
    for order in orders.values():
        order.prepare(required_passes)
    encoded_data = {
        kind: prepare_encoded_data(
            corpus,
            preprocessor_paths[kind],
            progress=progress,
            progress_file=progress_file,
        )
        for kind in preprocessing_kinds
    }

    settings = RunSettings(
        batch_size=profile.batch_size,
        encoded_data=encoded_data,
        orders=orders,
        train_pool_size=corpus.train_stop,
        preprocessor_sample_size=sample_size,
        preprocessor_seed=profile.preprocessor_seed,
        threads_per_worker=1 if workers > 1 else None,
    )
    runner = ScalingRunner(
        schema=RESULT_SCHEMA,
        checkpoints=train_sizes,
        objective=BINARY_OBJECTIVE,
        make_task_model=partial(_make_task_model, settings=settings),
        metadata=partial(_metadata, settings=settings),
    )
    results = run_tuning_and_evaluation(
        configs,
        runner=runner,
        select_evaluation_runs=partial(
            scaling.select_evaluation_runs,
            schema=RESULT_SCHEMA,
            evaluation_seeds=profile.evaluation_seeds,
            model_specs={
                (spec.variant, spec.result_dim): spec for spec in model_specs
            },
        ),
        workers=workers,
        progress=progress,
        progress_file=progress_file,
    )
    return results.reindex(columns=RESULT_SCHEMA.raw_columns)


def select_evaluations(raw: pd.DataFrame) -> pd.DataFrame:
    return scaling.select_evaluations(raw, schema=RESULT_SCHEMA)


def summarize_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    return scaling.summarize_evaluations(evaluations, schema=RESULT_SCHEMA)


def validate_raw(
    raw: pd.DataFrame,
    profile: Profile,
    variant: Variant | None = None,
) -> None:
    """Validate that raw results are a complete Criteo profile run."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    variants = (variant,) if variant is not None else VARIANTS
    specs = _model_specs(profile, variants)
    experiment = scaling.validate_results(
        raw,
        schema=RESULT_SCHEMA,
        expected_model_rows=[
            {"model": spec.variant, "dim": spec.result_dim} for spec in specs
        ],
        train_sizes=training_checkpoints(
            profile.train_sizes,
            batch_size=profile.batch_size,
        ),
        learning_rates=profile.lrs,
        tuning_seeds=profile.tuning_seeds,
        evaluation_seeds=profile.evaluation_seeds,
    )
    if experiment["protocol"] != PROTOCOL:
        raise ValueError(f"expected protocol={PROTOCOL!r}")
    if experiment["optimizer"] != OPTIMIZER:
        raise ValueError(f"expected optimizer={OPTIMIZER!r}")
    train_pool_size = int(experiment["train_pool_size"])
    sample_size = max(1, round(profile.preprocessor_fraction * train_pool_size))
    if experiment["preprocessor_sample_size"] != sample_size:
        raise ValueError("preprocessor sample size does not match the profile")
    if experiment["preprocessor_seed"] != profile.preprocessor_seed:
        raise ValueError("preprocessor seed does not match the profile")


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / (
        f"criteo_scaling_{profile_name}_{PROTOCOL}{suffix}.csv"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Headerless Criteo TSV."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="sanity")
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
    cache_dir = args.cache_dir or default_cache_dir(args.data)
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
