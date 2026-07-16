import argparse
import sys
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from itertools import pairwise, product
from operator import index
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from paper.experiments import run_many
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, WRITE_MODES, write_csv
from paper.higgs import (
    NUM_FEATURES,
    OFFICIAL_LAYOUT,
    HiggsCorpus,
    HiggsLayout,
    HiggsTask,
    prepare_corpus,
)
from paper.models import KthEigval
from paper.training import (
    fit_and_test_binary_scaling,
    tune_binary_scaling_stream,
)


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

PROTOCOL = "one_pass"
OPTIMIZER = "adam"

IDENTITY_COLUMNS = [
    "protocol",
    "optimizer",
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

EXPERIMENT_COLUMNS = ["protocol", "optimizer"]
MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "model",
    "dim",
    "width",
    "num_parameters",
]
CURVE_COLUMNS = MODEL_COLUMNS + ["train_size"]


def _positive(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def spectral_parameter_count(input_dim: int, dim: int) -> int:
    input_dim = _positive("input_dim", input_dim)
    dim = _positive("dim", dim)
    coordinates = dim * (dim + 1) // 2
    return (input_dim + 1) * coordinates


def mlp_parameter_count(input_dim: int, width: int, depth: int) -> int:
    input_dim = _positive("input_dim", input_dim)
    width = _positive("width", width)
    depth = _positive("depth", depth)
    return (depth - 1) * width**2 + (input_dim + depth + 1) * width + 1


def matched_mlp_width(
    input_dim: int,
    depth: int,
    target_parameters: int,
) -> int:
    """Return the positive width closest to a target parameter count."""
    input_dim = _positive("input_dim", input_dim)
    depth = _positive("depth", depth)
    target_parameters = _positive("target_parameters", target_parameters)

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
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=256,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        dims=(3, 5),
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
            10_000_000,
        ),
        dims=(3, 5, 9, 15),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 6).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 3), init_seeds=range(2, 5)
        ),
        batch_size=4096,
    ),
}


@dataclass(frozen=True)
class HiggsModelSpec:
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
    def depth(self) -> int | None:
        return MLP_DEPTHS.get(self.variant)

    def width(self, input_dim: int) -> int:
        if self.depth is None:
            return 0
        target = spectral_parameter_count(input_dim, self.dim)
        return matched_mlp_width(input_dim, self.depth, target)


@dataclass(frozen=True)
class RunConfig:
    data_seed: int
    model_spec: HiggsModelSpec
    lr: float
    init_seed: int


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


def _tuning_configs(
    profile: Profile, model_specs: tuple[HiggsModelSpec, ...]
) -> tuple[RunConfig, ...]:
    return tuple(
        RunConfig(data_seed, model_spec, lr, init_seed)
        for data_seed, model_spec, lr, init_seed in product(
            profile.tuning_seeds.data_seeds,
            model_specs,
            profile.lrs,
            profile.tuning_seeds.init_seeds,
        )
    )


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
    try:
        depth = MLP_DEPTHS[spec.variant]
    except KeyError as error:
        raise ValueError(spec.variant) from error
    return _make_mlp(input_dim, spec.width(input_dim), depth)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _make_seeded_model(spec: HiggsModelSpec, *, init_seed: int) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(spec)


def _make_task_model(
    config: RunConfig, settings: RunSettings
) -> tuple[HiggsTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)
    task = HiggsTask(settings.corpus, config.data_seed, settings.batch_size)
    model = _make_seeded_model(config.model_spec, init_seed=config.init_seed)
    return task, model


def _metadata(config: RunConfig, model: nn.Module) -> dict[str, int | float | str]:
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
    config: RunConfig,
    model: nn.Module,
) -> pd.DataFrame:
    return result.assign(
        phase=phase,
        **_metadata(config, model),
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
    result = tune_binary_scaling_stream(
        task,
        model,
        lr=config.lr,
        checkpoints=settings.train_sizes,
        validation_checkpoints=(settings.train_sizes[-1],),
    )
    return _format_result(
        result,
        phase="tuning",
        config=config,
        model=model,
    )


def run_selected(config: RunConfig, settings: RunSettings) -> pd.DataFrame:
    task, model = _make_task_model(config, settings)
    result = fit_and_test_binary_scaling(
        task,
        model,
        lr=config.lr,
        checkpoints=settings.train_sizes,
        test_checkpoints=settings.train_sizes,
    )
    return _format_result(
        result,
        phase="evaluation",
        config=config,
        model=model,
    )


def _best_lrs(tuning: pd.DataFrame) -> pd.DataFrame:
    if tuning.empty:
        raise ValueError("tuning results must not be empty")
    model_keys = tuning.loc[:, MODEL_COLUMNS].drop_duplicates()
    finite = tuning.loc[np.isfinite(tuning["val_logloss"])]
    scores = (
        finite.groupby(MODEL_COLUMNS + ["lr"], as_index=False)["val_logloss"]
        .median()
        .rename(columns={"val_logloss": "median_val_logloss"})
    )
    available = scores.loc[:, MODEL_COLUMNS].drop_duplicates()
    missing = model_keys.merge(available, on=MODEL_COLUMNS, how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", MODEL_COLUMNS]
    if not missing.empty:
        families = missing[["model", "dim"]].to_dict("records")
        raise ValueError(f"no finite validation log loss for {families}")

    best = (
        scores.sort_values(
            MODEL_COLUMNS + ["median_val_logloss", "lr"], kind="mergesort"
        )
        .groupby(MODEL_COLUMNS, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )
    return best[MODEL_COLUMNS + ["selected_lr", "median_val_logloss"]]


def _selected_configs(
    tuning: pd.DataFrame,
    evaluation_seeds: SeedGrid,
) -> tuple[RunConfig, ...]:
    return tuple(
        RunConfig(
            data_seed=data_seed,
            model_spec=HiggsModelSpec(row.model, int(row.dim)),
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
    layout: HiggsLayout = OFFICIAL_LAYOUT,
    chunk_size: int = 250_000,
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
        layout=layout,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    if max(profile.train_sizes) > corpus.train_stop:
        raise ValueError(
            f"profile requests {max(profile.train_sizes)} training rows, "
            f"but the training split contains {corpus.train_stop}"
        )

    variants = (variant,) if variant is not None else VARIANTS
    model_specs = _model_specs(profile, variants)
    configs = _tuning_configs(profile, model_specs)
    data_seeds = sorted(
        set(profile.tuning_seeds.data_seeds)
        | set(profile.evaluation_seeds.data_seeds)
    )
    for data_seed in data_seeds:
        corpus.order_path(data_seed)

    settings = RunSettings(
        train_sizes=profile.train_sizes,
        batch_size=profile.batch_size,
        corpus=corpus,
        threads_per_worker=1 if workers > 1 else None,
    )
    tune = partial(run_config, settings=settings)
    tuning_results = run_many(
        tune,
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

    evaluate = partial(run_selected, settings=settings)
    evaluation_results = run_many(
        evaluate,
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
    groups = CURVE_COLUMNS + ["selected_lr"]
    return (
        selected.groupby(groups)
        .agg(
            median_test_logloss=("test_logloss", "median"),
            q25_test_logloss=("test_logloss", lambda s: s.quantile(0.25)),
            q75_test_logloss=("test_logloss", lambda s: s.quantile(0.75)),
            median_test_brier=("test_brier", "median"),
            q25_test_brier=("test_brier", lambda s: s.quantile(0.25)),
            q75_test_brier=("test_brier", lambda s: s.quantile(0.75)),
            n=("test_logloss", "size"),
        )
        .reset_index()
    )


def _expected_capacity(spec: HiggsModelSpec) -> tuple[int, int]:
    width = spec.width(NUM_FEATURES)
    if spec.variant == "linear":
        return width, NUM_FEATURES + 1
    if spec.variant == "spectral":
        return width, spectral_parameter_count(NUM_FEATURES, spec.dim)
    depth = MLP_DEPTHS[spec.variant]
    return width, mlp_parameter_count(NUM_FEATURES, width, depth)


def _same_lrs(actual: pd.Series, expected: tuple[float, ...]) -> bool:
    values = np.sort(actual.unique())
    return len(values) == len(expected) and np.allclose(
        values, expected, rtol=1e-12, atol=0
    )


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
    if set(raw["phase"]) != {"tuning", "evaluation"}:
        raise ValueError("results must contain tuning and evaluation phases")
    if raw.duplicated(IDENTITY_COLUMNS[2:]).any():
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
    if set(tuning["train_size"]) != {profile.train_sizes[-1]}:
        raise ValueError("tuning must contain only the final checkpoint")
    if set(evaluation["train_size"]) != set(profile.train_sizes):
        raise ValueError("evaluation must contain every profile checkpoint")
    if tuning[["test_logloss", "test_brier"]].notna().any().any():
        raise ValueError("tuning rows must not contain test metrics")
    if evaluation["val_logloss"].notna().any():
        raise ValueError("evaluation rows must not contain validation metrics")
    if not np.isfinite(
        evaluation[["test_logloss", "test_brier"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("evaluation test metrics must be finite")

    nonfinite_tuning = ~np.isfinite(tuning["val_logloss"].to_numpy(dtype=float))
    if nonfinite_tuning.any():
        warnings.warn(
            f"{nonfinite_tuning.sum()} tuning trajectories have nonfinite "
            "validation loss",
            RuntimeWarning,
            stacklevel=2,
        )

    for phase, rows in (("tuning", tuning), ("evaluation", evaluation)):
        phase_specs = set(
            rows[["model", "dim"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if phase_specs != expected_specs:
            raise ValueError(f"{phase} has an incomplete model/capacity grid")

    if not _same_lrs(tuning["lr"], profile.lrs):
        raise ValueError("tuning learning-rate grid does not match the profile")
    tuning_seeds = set(profile.tuning_seeds)
    for spec, rows in tuning.groupby(["model", "dim"]):
        if not _same_lrs(rows["lr"], profile.lrs):
            raise ValueError(f"incomplete tuning learning-rate grid for {spec}")
        for lr, lr_rows in rows.groupby("lr"):
            seeds = set(
                lr_rows[["data_seed", "init_seed"]].itertuples(
                    index=False, name=None
                )
            )
            if seeds != tuning_seeds:
                raise ValueError(f"incomplete tuning seeds for {spec}, lr={lr:g}")

    selected_lrs = {
        (row.model, row.dim): row.selected_lr
        for row in _best_lrs(tuning).itertuples(index=False)
    }
    evaluation_seeds = set(profile.evaluation_seeds)
    expected_checkpoints = set(profile.train_sizes)
    for spec, rows in evaluation.groupby(["model", "dim"]):
        seeds = set(
            rows[["data_seed", "init_seed"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if seeds != evaluation_seeds:
            raise ValueError(f"incomplete evaluation seeds for {spec}")
        lrs = rows["lr"].unique()
        if len(lrs) != 1 or not np.isclose(
            lrs[0], selected_lrs[spec], rtol=1e-12, atol=0
        ):
            raise ValueError(f"evaluation does not use the selected LR for {spec}")
        checkpoints = rows.groupby(["data_seed", "init_seed"])["train_size"].agg(
            set
        )
        if not checkpoints.map(lambda values: values == expected_checkpoints).all():
            raise ValueError(f"incomplete evaluation trajectory for {spec}")


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / f"higgs_scaling_{profile_name}{suffix}.csv"


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
    cache_dir = args.cache_dir or args.data.with_name(f".{args.data.name}.cache-v1")
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
