import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn

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
from paper.higgs import (
    NUM_FEATURES,
    OFFICIAL_LAYOUT,
    HiggsCorpus,
    HiggsLayout,
    HiggsTask,
    default_cache_dir,
    prepare_corpus,
)
from paper.models import KthEigval
from paper.training import BINARY_OBJECTIVE, training_checkpoints


type Variant = Literal["linear", "mlp-1", "mlp-2", "mlp-3", "spectral"]

VARIANTS: tuple[Variant, ...] = (
    "linear",
    "mlp-1",
    "mlp-2",
    "mlp-3",
    "spectral",
)
MLP_DEPTHS: dict[Variant, int] = {
    "mlp-1": 1,
    "mlp-2": 2,
    "mlp-3": 3,
}

OPTIMIZER = "adam"

RESULT_SCHEMA = ScalingSchema(
    experiment_columns=("protocol", "optimizer", "train_pool_size"),
    model_columns=("model", "dim", "width", "num_parameters"),
    model_spec_columns=("model", "dim"),
    validation_metric="val_logloss",
    test_metrics=("test_logloss", "test_brier"),
)


def spectral_parameter_count(input_dim: int, dim: int) -> int:
    coordinates = dim * (dim + 1) // 2
    return (input_dim + 1) * coordinates


def mlp_parameter_count(input_dim: int, width: int, depth: int) -> int:
    return (depth - 1) * width**2 + (input_dim + depth + 1) * width + 1


def matched_mlp_width(
    input_dim: int,
    depth: int,
    target_parameters: int,
) -> int:
    """Return the positive width closest to a target parameter count."""
    if mlp_parameter_count(input_dim, 1, depth) >= target_parameters:
        return 1

    lower = 1
    upper = 2
    while mlp_parameter_count(input_dim, upper, depth) < target_parameters:
        lower = upper
        upper *= 2

    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if mlp_parameter_count(input_dim, middle, depth) < target_parameters:
            lower = middle
        else:
            upper = middle

    return min(
        (lower, upper),
        key=lambda width: (
            abs(mlp_parameter_count(input_dim, width, depth) - target_parameters),
            width,
        ),
    )


@dataclass(frozen=True)
class Profile:
    train_sizes: tuple[int, ...]
    capacity_dims: tuple[int, ...]
    lrs: tuple[float, ...]
    tuning_seeds: SeedGrid
    evaluation_seeds: SeedGrid
    batch_size: int = 4096


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        train_sizes=(2**10,),
        capacity_dims=(3,),
        lrs=(1e-2,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=256,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        capacity_dims=(3, 7),
        lrs=(1e-3, 1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 2), init_seeds=range(2, 4)
        ),
        batch_size=4096,
    ),
    "full": Profile(
        train_sizes=(
            2**12,
            2**14,
            2**16,
            2**18,
            2**20,
            2**22,
            2**23,
            2**24,
            2**25,
            2**26,
        ),
        capacity_dims=(3, 7, 11),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 8).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(8)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 5),
            init_seeds=range(3, 9),
        ),
        batch_size=4096,
    ),
}


@dataclass(frozen=True)
class HiggsModelSpec:
    variant: Variant
    capacity_dim: int | None = None

    def __post_init__(self) -> None:
        if (self.variant == "linear") != (self.capacity_dim is None):
            raise ValueError("only linear models omit capacity_dim")
        if self.capacity_dim is not None and self.capacity_dim <= 0:
            raise ValueError("capacity_dim must be positive")

    @property
    def result_dim(self) -> int:
        """Return the dimension used by the persisted result schema."""
        return 0 if self.capacity_dim is None else self.capacity_dim

    @property
    def depth(self) -> int | None:
        return MLP_DEPTHS.get(self.variant)

    def width(self, input_dim: int) -> int:
        if self.depth is None:
            return 0
        assert self.capacity_dim is not None
        target = spectral_parameter_count(input_dim, self.capacity_dim)
        return matched_mlp_width(input_dim, self.depth, target)


@dataclass(frozen=True)
class RunSettings:
    batch_size: int
    corpus: HiggsCorpus
    threads_per_worker: int | None


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[HiggsModelSpec, ...]:
    specs = [HiggsModelSpec("linear")]
    specs.extend(
        HiggsModelSpec(variant, capacity_dim)
        for capacity_dim in profile.capacity_dims
        for variant in VARIANTS
        if variant != "linear"
    )
    return tuple(spec for spec in specs if spec.variant in variants)


def _make_mlp(input_dim: int, width: int, depth: int) -> nn.Sequential:
    dimensions = (input_dim,) + (width,) * depth + (1,)
    layers: list[nn.Module] = []
    for layer_index, (in_features, out_features) in enumerate(
        pairwise(dimensions)
    ):
        layers.append(nn.Linear(in_features, out_features))
        if layer_index < depth:
            layers.append(nn.ReLU())
    layers.append(nn.Flatten(start_dim=-2, end_dim=-1))
    return nn.Sequential(*layers)


def make_model(spec: HiggsModelSpec, input_dim: int = NUM_FEATURES) -> nn.Module:
    if spec.variant == "linear":
        return nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )
    if spec.variant == "spectral":
        assert spec.capacity_dim is not None
        return KthEigval(
            input_dim,
            spec.capacity_dim,
            eig_idx=spec.capacity_dim // 2,
        )
    depth = MLP_DEPTHS[spec.variant]
    return _make_mlp(input_dim, spec.width(input_dim), depth)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _expected_capacity(spec: HiggsModelSpec) -> tuple[int, int]:
    width = spec.width(NUM_FEATURES)
    if spec.variant == "linear":
        return width, NUM_FEATURES + 1
    if spec.variant == "spectral":
        assert spec.capacity_dim is not None
        return width, spectral_parameter_count(NUM_FEATURES, spec.capacity_dim)
    depth = MLP_DEPTHS[spec.variant]
    return width, mlp_parameter_count(NUM_FEATURES, width, depth)


def _make_seeded_model(spec: HiggsModelSpec, *, init_seed: int) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(spec)


def make_task_model(
    config: RunConfig[HiggsModelSpec], settings: RunSettings
) -> tuple[HiggsTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)
    task = HiggsTask(settings.corpus, config.data_seed, settings.batch_size)
    model = _make_seeded_model(config.model_spec, init_seed=config.init_seed)
    return task, model


def _metadata(
    config: RunConfig[HiggsModelSpec],
    model: nn.Module,
    *,
    train_pool_size: int,
) -> dict[str, int | float | str]:
    return {
        "protocol": PROTOCOL,
        "optimizer": OPTIMIZER,
        "data_seed": config.data_seed,
        "model": config.model_spec.variant,
        "dim": config.model_spec.result_dim,
        "width": config.model_spec.width(NUM_FEATURES),
        "num_parameters": trainable_parameter_count(model),
        "lr": config.lr,
        "init_seed": config.init_seed,
        "train_pool_size": train_pool_size,
    }


def run_profile(
    profile: Profile,
    *,
    raw_path: Path,
    cache_dir: Path,
    layout: HiggsLayout = OFFICIAL_LAYOUT,
    chunk_size: int = 250_000,
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
        layout=layout,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    train_sizes = training_checkpoints(
        profile.train_sizes,
        batch_size=profile.batch_size,
    )

    variants = (variant,) if variant is not None else VARIANTS
    model_specs = _model_specs(profile, variants)
    configs = tuning_configs(model_specs, profile.lrs, profile.tuning_seeds)
    data_seeds = sorted(
        set(profile.tuning_seeds.data_seeds)
        | set(profile.evaluation_seeds.data_seeds)
    )
    required_passes = (train_sizes[-1] + corpus.train_stop - 1) // corpus.train_stop
    for data_seed in data_seeds:
        corpus.shuffled_epochs(data_seed).prepare(required_passes)

    settings = RunSettings(
        batch_size=profile.batch_size,
        corpus=corpus,
        threads_per_worker=1 if workers > 1 else None,
    )
    runner = ScalingRunner(
        schema=RESULT_SCHEMA,
        checkpoints=train_sizes,
        objective=BINARY_OBJECTIVE,
        make_task_model=partial(make_task_model, settings=settings),
        metadata=partial(_metadata, train_pool_size=corpus.train_stop),
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
    """Validate that raw results are a complete run of a HIGGS profile."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    variants = (variant,) if variant is not None else VARIANTS
    specs = _model_specs(profile, variants)
    expected_models = []
    for spec in specs:
        width, num_parameters = _expected_capacity(spec)
        expected_models.append(
            {
                "model": spec.variant,
                "dim": spec.result_dim,
                "width": width,
                "num_parameters": num_parameters,
            }
        )

    experiment = scaling.validate_results(
        raw,
        schema=RESULT_SCHEMA,
        expected_model_rows=expected_models,
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


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / (
        f"higgs_scaling_{profile_name}_repeated_shuffle{suffix}.csv"
    )


def build_arg_parser(
    profiles: Mapping[str, Profile] = PROFILES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Headerless HIGGS CSV."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--profile", choices=profiles.keys(), default="sanity")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=250_000)
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
