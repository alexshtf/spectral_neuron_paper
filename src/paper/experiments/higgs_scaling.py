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
    SeedGrid,
    SelectedRun,
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
from paper.shuffling import resolve_train_sizes
from paper.training import (
    BINARY_OBJECTIVE,
    fit_and_test_scaling,
    tune_scaling_stream,
)
from paper.tuning import same_learning_rates, select_learning_rates


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

IDENTITY_COLUMNS = [
    "protocol",
    "optimizer",
    "train_pool_size",
    "phase",
    "train_size",
    "data_seed",
    "model",
    "dim",
    "width",
    "num_parameters",
    "lr",
    "init_seed",
]
METRIC_COLUMNS = [
    "val_logloss",
    "test_logloss",
    "test_brier",
]
RAW_COLUMNS = IDENTITY_COLUMNS + METRIC_COLUMNS

_TIMING_COLUMNS = ["train_seconds", "val_seconds", "test_seconds"]

EXPERIMENT_COLUMNS = ["protocol", "optimizer", "train_pool_size"]
MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "model",
    "dim",
    "width",
    "num_parameters",
]
CURVE_COLUMNS = MODEL_COLUMNS + ["train_size"]


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
    dims: tuple[int, ...]
    lrs: tuple[float, ...]
    tuning_seeds: SeedGrid
    evaluation_seeds: SeedGrid
    batch_size: int = 4096


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        train_sizes=(2**10,),
        dims=(3,),
        lrs=(1e-2,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=256,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        dims=(3, 7),
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
        dims=(3, 7, 11),
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
    dim: int = 0

    @property
    def depth(self) -> int | None:
        return MLP_DEPTHS.get(self.variant)

    def width(self, input_dim: int) -> int:
        if self.depth is None:
            return 0
        target = spectral_parameter_count(input_dim, self.dim)
        return matched_mlp_width(input_dim, self.depth, target)


@dataclass(frozen=True)
class RunSettings:
    train_sizes: tuple[int, ...]
    batch_size: int
    corpus: HiggsCorpus
    threads_per_worker: int | None


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[HiggsModelSpec, ...]:
    specs = [HiggsModelSpec("linear")]
    specs.extend(
        HiggsModelSpec(variant, dim)
        for dim in profile.dims
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
        return KthEigval(input_dim, spec.dim, eig_idx=spec.dim // 2)
    depth = MLP_DEPTHS[spec.variant]
    return _make_mlp(input_dim, spec.width(input_dim), depth)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


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
    config: RunConfig[HiggsModelSpec], model: nn.Module
) -> dict[str, int | float | str]:
    return {
        "protocol": PROTOCOL,
        "optimizer": OPTIMIZER,
        "data_seed": config.data_seed,
        "model": config.model_spec.variant,
        "dim": config.model_spec.dim,
        "width": config.model_spec.width(NUM_FEATURES),
        "num_parameters": trainable_parameter_count(model),
        "lr": config.lr,
        "init_seed": config.init_seed,
    }


def _format_result(
    result: pd.DataFrame,
    *,
    phase: Literal["tuning", "evaluation"],
    config: RunConfig[HiggsModelSpec],
    model: nn.Module,
    train_pool_size: int,
) -> pd.DataFrame:
    return result.assign(
        phase=phase,
        train_pool_size=train_pool_size,
        **_metadata(config, model),
    ).reindex(columns=RAW_COLUMNS + _TIMING_COLUMNS)


def run_config(
    config: RunConfig[HiggsModelSpec], settings: RunSettings
) -> pd.DataFrame:
    task, model = make_task_model(config, settings)
    result = tune_scaling_stream(
        task,
        model,
        objective=BINARY_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
    )
    return _format_result(
        result,
        phase="tuning",
        config=config,
        model=model,
        train_pool_size=settings.corpus.train_stop,
    )


def run_selected(
    selected: SelectedRun[HiggsModelSpec], settings: RunSettings
) -> pd.DataFrame:
    config = selected.config
    task, model = make_task_model(config, settings)
    checkpoints = tuple(
        size for size in settings.train_sizes if size <= max(selected.train_sizes)
    )
    result = fit_and_test_scaling(
        task,
        model,
        objective=BINARY_OBJECTIVE,
        lr=config.lr,
        checkpoints=checkpoints,
        test_checkpoints=selected.train_sizes,
    )
    return _format_result(
        result,
        phase="evaluation",
        config=config,
        model=model,
        train_pool_size=settings.corpus.train_stop,
    )


def _select_evaluation_runs(
    tuning: pd.DataFrame,
    evaluation_seeds: SeedGrid,
    model_specs: tuple[HiggsModelSpec, ...],
) -> tuple[SelectedRun[HiggsModelSpec], ...]:
    return scaling.select_evaluation_runs(
        tuning,
        experiment_columns=EXPERIMENT_COLUMNS,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_logloss",
        evaluation_seeds=evaluation_seeds,
        model_columns=("model", "dim"),
        model_specs={(spec.variant, spec.dim): spec for spec in model_specs},
    )


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
    train_sizes = resolve_train_sizes(
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
        train_sizes=train_sizes,
        batch_size=profile.batch_size,
        corpus=corpus,
        threads_per_worker=1 if workers > 1 else None,
    )
    results = run_tuning_and_evaluation(
        configs,
        tune=partial(run_config, settings=settings),
        select_evaluation_runs=partial(
            _select_evaluation_runs,
            evaluation_seeds=profile.evaluation_seeds,
            model_specs=model_specs,
        ),
        evaluate=partial(run_selected, settings=settings),
        workers=workers,
        progress=progress,
        progress_file=progress_file,
    )
    return results.reindex(columns=RAW_COLUMNS)


def select_evaluations(raw: pd.DataFrame) -> pd.DataFrame:
    return scaling.select_evaluations(
        raw,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_logloss",
    )


def summarize_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    return scaling.summarize_evaluations(
        evaluations,
        curve_columns=CURVE_COLUMNS,
        quantile_metrics=("test_logloss", "test_brier"),
    )


def _expected_capacity(spec: HiggsModelSpec) -> tuple[int, int]:
    width = spec.width(NUM_FEATURES)
    if spec.variant == "linear":
        return width, NUM_FEATURES + 1
    if spec.variant == "spectral":
        return width, spectral_parameter_count(NUM_FEATURES, spec.dim)
    depth = MLP_DEPTHS[spec.variant]
    return width, mlp_parameter_count(NUM_FEATURES, width, depth)


def validate_raw(
    raw: pd.DataFrame,
    profile: Profile,
    variant: Variant | None = None,
) -> None:
    """Validate that raw results are a complete run of a HIGGS profile."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("incompatible HIGGS result schema")
    if raw[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("HIGGS run identity columns must not contain missing values")
    if set(raw["protocol"]) != {PROTOCOL}:
        raise ValueError(f"expected protocol={PROTOCOL!r}")
    if set(raw["optimizer"]) != {OPTIMIZER}:
        raise ValueError(f"expected optimizer={OPTIMIZER!r}")
    train_pool_sizes = raw["train_pool_size"].unique()
    if len(train_pool_sizes) != 1:
        raise ValueError("results must contain one train_pool_size")
    train_pool_size = int(train_pool_sizes[0])
    if set(raw["phase"]) != {"tuning", "evaluation"}:
        raise ValueError("results must contain tuning and evaluation phases")
    if raw.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("results contain duplicate trajectory checkpoints")

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

    for spec in specs:
        rows = raw.loc[
            (raw["model"] == spec.variant) & (raw["dim"] == spec.dim),
            ["width", "num_parameters"],
        ].drop_duplicates()
        expected = _expected_capacity(spec)
        if len(rows) != 1 or tuple(rows.iloc[0]) != expected:
            raise ValueError(
                f"inconsistent capacity metadata for {(spec.variant, spec.dim)}; "
                f"expected width={expected[0]}, parameters={expected[1]}"
            )

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    train_sizes = resolve_train_sizes(
        profile.train_sizes,
        batch_size=profile.batch_size,
    )
    experiment = (PROTOCOL, OPTIMIZER, train_pool_size)
    expected_curves = {
        (*experiment, spec.variant, spec.dim, *_expected_capacity(spec), train_size)
        for spec in specs
        for train_size in train_sizes
    }
    for phase, rows in (("tuning", tuning), ("evaluation", evaluation)):
        observed_curves = set(
            rows[CURVE_COLUMNS]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if observed_curves != expected_curves:
            raise ValueError(f"{phase} has an incomplete model/checkpoint grid")

    if tuning[["test_logloss", "test_brier"]].notna().any().any():
        raise ValueError("tuning rows must not contain test metrics")
    if evaluation["val_logloss"].notna().any():
        raise ValueError("evaluation rows must not contain validation metrics")
    if not np.isfinite(
        evaluation[["test_logloss", "test_brier"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("evaluation test metrics must be finite")

    if not same_learning_rates(tuning["lr"], profile.lrs):
        raise ValueError("tuning learning-rate grid does not match the profile")
    tuning_seeds = set(profile.tuning_seeds)
    for curve, rows in tuning.groupby(CURVE_COLUMNS):
        if not same_learning_rates(rows["lr"], profile.lrs):
            raise ValueError(f"incomplete tuning learning-rate grid for {curve}")
        for lr, lr_rows in rows.groupby("lr"):
            seeds = set(
                lr_rows[["data_seed", "init_seed"]].itertuples(
                    index=False, name=None
                )
            )
            if seeds != tuning_seeds:
                raise ValueError(f"incomplete tuning seeds for {curve}, lr={lr:g}")

    selected_lrs = select_learning_rates(
        tuning,
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_logloss",
    ).set_index(CURVE_COLUMNS)["selected_lr"]
    evaluation_seeds = set(profile.evaluation_seeds)
    for curve, rows in evaluation.groupby(CURVE_COLUMNS):
        seeds = set(
            rows[["data_seed", "init_seed"]].itertuples(index=False, name=None)
        )
        if seeds != evaluation_seeds:
            raise ValueError(f"incomplete evaluation seeds for {curve}")
        lrs = rows["lr"].unique()
        if len(lrs) != 1 or not np.isclose(
            lrs[0], selected_lrs.loc[curve], rtol=1e-12, atol=0
        ):
            raise ValueError(f"evaluation does not use the selected LR for {curve}")


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
