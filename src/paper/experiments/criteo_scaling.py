import argparse
import sys
import warnings
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

from paper.criteo import (
    NUM_FIELDS,
    CriteoTask,
    EncodedData,
    PreprocessingKind,
    fit_preprocessors,
    prepare_corpus,
    prepare_encoded_data,
)
from paper.experiments import run_many
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, WRITE_MODES, write_csv
from paper.models import FactorizationMachine, SparseKthEigval, SparseLinear
from paper.shuffling import ShuffledEpochs, resolve_train_sizes
from paper.training import (
    BINARY_OBJECTIVE,
    fit_and_test_scaling,
    tune_scaling_stream,
)


type Variant = Literal[
    "linear",
    "linear-new",
    "fm",
    "spectral-old",
    "spectral-new",
]

VARIANTS: tuple[Variant, ...] = (
    "linear",
    "linear-new",
    "fm",
    "spectral-old",
    "spectral-new",
)

PROTOCOL = "repeated_shuffle"
OPTIMIZER = "adam+sparseadam"

IDENTITY_COLUMNS = [
    "protocol",
    "optimizer",
    "preprocessor_sample_size",
    "preprocessor_seed",
    "train_pool_size",
    "phase",
    "train_size",
    "data_seed",
    "model",
    "dim",
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

EXPERIMENT_COLUMNS = [
    "protocol",
    "optimizer",
    "preprocessor_sample_size",
    "preprocessor_seed",
    "train_pool_size",
]

MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "train_size",
    "model",
    "dim",
]


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
    preprocessor_fraction: float = 0.1
    preprocessor_seed: int = 0
    min_count: int = 10
    buckets_per_field: int = 2**15
    passes: int | None = None

    def __post_init__(self) -> None:
        if not self.train_sizes or self.train_sizes != tuple(
            sorted(set(self.train_sizes))
        ):
            raise ValueError("train_sizes must be non-empty, unique, and increasing")
        if not self.tuning_seeds or not self.evaluation_seeds:
            raise ValueError("tuning and evaluation seed grids must be non-empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.preprocessor_fraction <= 1:
            raise ValueError("preprocessor_fraction must be in (0, 1]")
        if self.passes is not None and self.passes <= 0:
            raise ValueError("passes must be positive")


PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        train_sizes=(2**10,),
        dims=(3,),
        lrs=(1e-2,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=256,
        min_count=2,
        buckets_per_field=2**8,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        dims=(3, 5),
        lrs=(1e-3, 1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(init_seeds=range(2)),
        batch_size=4096,
    ),
    "full": Profile(
        train_sizes=tuple(2**power for power in range(12, 27, 2)),
        dims=(3, 7, 11),
        lrs=tuple(np.geomspace(1e-3, 1e-1, 8).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(8)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 5),
            init_seeds=range(3, 9),
        ),
        batch_size=4096,
        passes=2,
    ),
}


@dataclass(frozen=True)
class CriteoModelSpec:
    variant: Variant
    dim: int = 0

    @property
    def preprocessing(self) -> PreprocessingKind:
        return "hybrid" if self.variant.endswith("-new") else "bucket"


@dataclass(frozen=True)
class RunConfig:
    data_seed: int
    model_spec: CriteoModelSpec
    lr: float
    init_seed: int


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[CriteoModelSpec, ...]:
    specs = [CriteoModelSpec("linear"), CriteoModelSpec("linear-new")]
    specs.extend(
        CriteoModelSpec(variant, dim)
        for dim in profile.dims
        for variant in ("fm", "spectral-old", "spectral-new")
    )
    return tuple(spec for spec in specs if spec.variant in variants)


def _tuning_configs(
    profile: Profile, model_specs: tuple[CriteoModelSpec, ...]
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


@dataclass(frozen=True)
class RunSettings:
    train_sizes: tuple[int, ...]
    batch_size: int
    encoded_data: dict[PreprocessingKind, EncodedData]
    orders: dict[int, ShuffledEpochs]
    train_pool_size: int
    preprocessor_sample_size: int
    preprocessor_seed: int
    threads_per_worker: int | None


@dataclass(frozen=True)
class SelectedRun:
    config: RunConfig
    train_sizes: tuple[int, ...]


def make_model(spec: CriteoModelSpec, num_features: int) -> nn.Module:
    match spec.variant:
        case "linear" | "linear-new":
            return SparseLinear(num_features, NUM_FIELDS)
        case "fm":
            rank = spec.dim * (spec.dim + 1) // 2 - 1
            return FactorizationMachine(num_features, NUM_FIELDS, rank)
        case "spectral-old" | "spectral-new":
            return SparseKthEigval(num_features, NUM_FIELDS, spec.dim)
        case _:
            raise ValueError(spec.variant)


def _make_seeded_model(
    spec: CriteoModelSpec, *, num_features: int, init_seed: int
) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(spec, num_features)


def _make_task_model(
    config: RunConfig, settings: RunSettings
) -> tuple[CriteoTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)

    preprocessing = config.model_spec.preprocessing
    data = settings.encoded_data[preprocessing]
    task = CriteoTask(data, settings.orders[config.data_seed], settings.batch_size)
    model = _make_seeded_model(
        config.model_spec,
        num_features=data.num_features,
        init_seed=config.init_seed,
    )
    return task, model


def _metadata(
    config: RunConfig,
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
        "dim": config.model_spec.dim,
        "lr": config.lr,
        "init_seed": config.init_seed,
    }


def _format_result(
    result: pd.DataFrame,
    *,
    phase: Literal["tuning", "evaluation"],
    config: RunConfig,
    settings: RunSettings,
) -> pd.DataFrame:
    return result.assign(
        phase=phase,
        **_metadata(config, settings),
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
        objective=BINARY_OBJECTIVE,
        lr=config.lr,
        checkpoints=settings.train_sizes,
    )
    return _format_result(
        result,
        phase="tuning",
        config=config,
        settings=settings,
    )


def run_selected(selected: SelectedRun, settings: RunSettings) -> pd.DataFrame:
    task, model = _make_task_model(selected.config, settings)
    checkpoints = tuple(
        size for size in settings.train_sizes if size <= max(selected.train_sizes)
    )
    result = fit_and_test_scaling(
        task,
        model,
        objective=BINARY_OBJECTIVE,
        lr=selected.config.lr,
        checkpoints=checkpoints,
        test_checkpoints=selected.train_sizes,
    )
    return _format_result(
        result,
        phase="evaluation",
        config=selected.config,
        settings=settings,
    )


def _selected_runs(
    tuning: pd.DataFrame,
    evaluation_seeds: SeedGrid,
) -> list[SelectedRun]:
    train_sizes: dict[RunConfig, list[int]] = {}
    for row in _best_lrs(tuning).itertuples(index=False):
        for data_seed, init_seed in evaluation_seeds:
            config = RunConfig(
                data_seed=data_seed,
                model_spec=CriteoModelSpec(row.model, row.dim),
                lr=row.selected_lr,
                init_seed=init_seed,
            )
            train_sizes.setdefault(config, []).append(row.train_size)

    return [
        SelectedRun(config, tuple(sorted(set(sizes))))
        for config, sizes in train_sizes.items()
    ]


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
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    train_sizes = resolve_train_sizes(
        profile.train_sizes,
        train_pool_size=corpus.train_stop,
        batch_size=profile.batch_size,
        passes=profile.passes,
    )

    sample_size = max(1, round(profile.preprocessor_fraction * corpus.train_stop))
    variants = (variant,) if variant is not None else VARIANTS
    model_specs = _model_specs(profile, variants)
    configs = _tuning_configs(profile, model_specs)
    preprocessing_kinds = tuple(
        dict.fromkeys(spec.preprocessing for spec in model_specs)
    )
    preprocessor_paths = fit_preprocessors(
        corpus,
        preprocessing_kinds,
        sample_size=sample_size,
        sample_seed=profile.preprocessor_seed,
        min_count=profile.min_count,
        buckets_per_field=profile.buckets_per_field,
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
        train_sizes=train_sizes,
        batch_size=profile.batch_size,
        encoded_data=encoded_data,
        orders=orders,
        train_pool_size=corpus.train_stop,
        preprocessor_sample_size=sample_size,
        preprocessor_seed=profile.preprocessor_seed,
        threads_per_worker=1 if workers > 1 else None,
    )
    tune = partial(run_config, settings=settings)
    tuning_results = run_many(
        tune,
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
    selected_runs = _selected_runs(tuning, profile.evaluation_seeds)

    test = partial(run_selected, settings=settings)
    test_results = run_many(
        test,
        selected_runs,
        workers=workers,
        desc="Evaluation (retrain + test)",
        unit="trajectory",
        progress=progress,
        progress_file=progress_file,
    )
    if progress:
        _report_timings("Evaluation", "test", test_results, progress_file)

    evaluation = pd.concat(test_results, ignore_index=True)
    return pd.concat((tuning, evaluation), ignore_index=True).loc[:, RAW_COLUMNS]


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
        checkpoints = missing[["model", "dim", "train_size"]].to_dict("records")
        raise ValueError(f"no finite validation log loss for {checkpoints}")

    best = (
        scores.sort_values(
            MODEL_COLUMNS + ["median_val_logloss", "lr"], kind="mergesort"
        )
        .groupby(MODEL_COLUMNS, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "selected_lr"})
    )
    return best[MODEL_COLUMNS + ["selected_lr", "median_val_logloss"]]


def select_lr(raw: pd.DataFrame) -> pd.DataFrame:
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    best = _best_lrs(tuning)
    selected = evaluation.merge(
        best[MODEL_COLUMNS + ["selected_lr", "median_val_logloss"]],
        on=MODEL_COLUMNS,
        how="inner",
    )
    return selected.loc[selected["lr"] == selected["selected_lr"]].reset_index(
        drop=True
    )


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    selected = select_lr(raw)
    groups = MODEL_COLUMNS + ["selected_lr"]
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
    """Validate that raw results are a complete Criteo profile run."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("incompatible Criteo result schema")
    if raw[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("Criteo run identity columns must not contain missing values")
    if set(raw["protocol"]) != {PROTOCOL}:
        raise ValueError(f"expected protocol={PROTOCOL!r}")
    if set(raw["optimizer"]) != {OPTIMIZER}:
        raise ValueError(f"expected optimizer={OPTIMIZER!r}")

    train_pool_sizes = raw["train_pool_size"].unique()
    if (
        len(train_pool_sizes) != 1
        or isinstance(train_pool_sizes[0], (bool, np.bool_))
        or not isinstance(train_pool_sizes[0], (int, np.integer))
        or train_pool_sizes[0] <= 0
    ):
        raise ValueError("results must contain one positive integer train_pool_size")
    train_pool_size = int(train_pool_sizes[0])
    sample_size = max(1, round(profile.preprocessor_fraction * train_pool_size))
    if set(raw["preprocessor_sample_size"]) != {sample_size}:
        raise ValueError("preprocessor sample size does not match the profile")
    if set(raw["preprocessor_seed"]) != {profile.preprocessor_seed}:
        raise ValueError("preprocessor seed does not match the profile")
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
            f"model/dimension grid mismatch: expected {sorted(expected_specs)}, "
            f"got {sorted(observed_specs)}"
        )

    train_sizes = resolve_train_sizes(
        profile.train_sizes,
        train_pool_size=train_pool_size,
        batch_size=profile.batch_size,
        passes=profile.passes,
    )
    expected_curves = {
        (spec.variant, spec.dim, train_size)
        for spec in specs
        for train_size in train_sizes
    }
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    for phase, rows in (("tuning", tuning), ("evaluation", evaluation)):
        observed_curves = set(
            rows[["model", "dim", "train_size"]]
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

    nonfinite_tuning = ~np.isfinite(tuning["val_logloss"].to_numpy(dtype=float))
    if nonfinite_tuning.any():
        warnings.warn(
            f"{nonfinite_tuning.sum()} tuning trajectories have nonfinite "
            "validation loss",
            RuntimeWarning,
            stacklevel=2,
        )

    if not _same_lrs(tuning["lr"], profile.lrs):
        raise ValueError("tuning learning-rate grid does not match the profile")
    tuning_seeds = set(profile.tuning_seeds)
    for curve, rows in tuning.groupby(["model", "dim", "train_size"]):
        if not _same_lrs(rows["lr"], profile.lrs):
            raise ValueError(f"incomplete tuning learning-rate grid for {curve}")
        for lr, lr_rows in rows.groupby("lr"):
            seeds = set(
                lr_rows[["data_seed", "init_seed"]].itertuples(index=False, name=None)
            )
            if seeds != tuning_seeds:
                raise ValueError(f"incomplete tuning seeds for {curve}, lr={lr:g}")

    selected_lrs = {
        (row.model, row.dim, row.train_size): row.selected_lr
        for row in _best_lrs(tuning).itertuples(index=False)
    }
    evaluation_seeds = set(profile.evaluation_seeds)
    for curve, rows in evaluation.groupby(["model", "dim", "train_size"]):
        seeds = set(rows[["data_seed", "init_seed"]].itertuples(index=False, name=None))
        if seeds != evaluation_seeds:
            raise ValueError(f"incomplete evaluation seeds for {curve}")
        lrs = rows["lr"].unique()
        if len(lrs) != 1 or not np.isclose(
            lrs[0], selected_lrs[curve], rtol=1e-12, atol=0
        ):
            raise ValueError(f"evaluation does not use the selected LR for {curve}")


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / (
        f"criteo_scaling_{profile_name}_repeated_shuffle{suffix}.csv"
    )


def build_arg_parser(
    profiles: Mapping[str, Profile] = PROFILES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Headerless Criteo TSV."
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
    cache_dir = args.cache_dir or args.data.with_name(f".{args.data.name}.cache-v2")
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
