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

from paper.criteo import (
    NUM_FIELDS,
    CriteoCorpus,
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
from paper.training import (
    fit_and_test_binary_scaling,
    tune_binary_scaling_stream,
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

PROTOCOL = "one_pass"
OPTIMIZER = "adam+sparseadam"

RAW_COLUMNS = [
    "protocol",
    "optimizer",
    "preprocessor_sample_size",
    "preprocessor_seed",
    "phase",
    "train_size",
    "data_seed",
    "model",
    "dim",
    "lr",
    "init_seed",
    "val_logloss",
    "test_logloss",
    "test_brier",
]

_TIMING_COLUMNS = ["train_seconds", "val_seconds", "test_seconds"]

EXPERIMENT_COLUMNS = [
    "protocol",
    "optimizer",
    "preprocessor_sample_size",
    "preprocessor_seed",
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
        train_sizes=(
            *tuple(2**power for power in range(11, 23, 2)),
            2**22,
            36_672_493 // 8,
        ),
        dims=(3, 5, 9, 15),
        lrs=tuple(np.geomspace(1e-3, 1e-1, 8).tolist()),
        tuning_seeds=SeedGrid(init_seeds=range(3)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 5),
            init_seeds=range(3, 9),
        ),
        batch_size=4096,
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    variant: Variant
    dim: int = 0

    @property
    def preprocessing(self) -> PreprocessingKind:
        return "hybrid" if self.variant.endswith("-new") else "bucket"


@dataclass(frozen=True)
class RunConfig:
    data_seed: int
    model_spec: ModelSpec
    lr: float
    init_seed: int


def _model_specs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[ModelSpec, ...]:
    specs = [ModelSpec("linear"), ModelSpec("linear-new")]
    specs.extend(
        ModelSpec(variant, dim)
        for dim in profile.dims
        for variant in ("fm", "spectral-old", "spectral-new")
    )
    return tuple(spec for spec in specs if spec.variant in variants)


def _tuning_configs(
    profile: Profile, variants: tuple[Variant, ...]
) -> tuple[RunConfig, ...]:
    return tuple(
        RunConfig(data_seed, model_spec, lr, init_seed)
        for data_seed, model_spec, lr, init_seed in product(
            profile.tuning_seeds.data_seeds,
            _model_specs(profile, variants),
            profile.lrs,
            profile.tuning_seeds.init_seeds,
        )
    )


@dataclass(frozen=True)
class RunSettings:
    cache_dir: Path
    train_sizes: tuple[int, ...]
    batch_size: int
    encoded_data: dict[PreprocessingKind, EncodedData]
    preprocessor_sample_size: int
    preprocessor_seed: int
    threads_per_worker: int | None


@dataclass(frozen=True)
class SelectedRun:
    config: RunConfig
    train_sizes: tuple[int, ...]


def make_model(spec: ModelSpec, num_features: int) -> nn.Module:
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
    spec: ModelSpec, *, num_features: int, init_seed: int
) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        return make_model(spec, num_features)


def _make_task_model(
    config: RunConfig, settings: RunSettings
) -> tuple[CriteoTask, nn.Module]:
    if settings.threads_per_worker is not None:
        torch.set_num_threads(settings.threads_per_worker)

    corpus = CriteoCorpus.open(settings.cache_dir)
    preprocessing = config.model_spec.preprocessing
    data = settings.encoded_data[preprocessing]
    order = np.load(corpus.order_path(config.data_seed), mmap_mode="r")
    task = CriteoTask(
        corpus=corpus,
        train=data.train[config.data_seed],
        holdout=data.holdout,
        train_rows=order[: max(settings.train_sizes)],
        batch_size=settings.batch_size,
    )
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
    result = tune_binary_scaling_stream(
        task,
        model,
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
    result = fit_and_test_binary_scaling(
        task,
        model,
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
                model_spec=ModelSpec(row.model, row.dim),
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
    if not profile.train_sizes or profile.train_sizes != tuple(
        sorted(set(profile.train_sizes))
    ):
        raise ValueError("train_sizes must be non-empty, unique, and increasing")
    if not profile.tuning_seeds or not profile.evaluation_seeds:
        raise ValueError("tuning and evaluation seed grids must be non-empty")
    if not 0 < profile.preprocessor_fraction <= 1:
        raise ValueError("preprocessor_fraction must be in (0, 1]")

    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    if max(profile.train_sizes) > corpus.train_stop:
        raise ValueError(
            f"profile requests {max(profile.train_sizes)} training rows, "
            f"but the 80% split contains {corpus.train_stop}"
        )

    sample_size = max(1, round(profile.preprocessor_fraction * corpus.train_stop))
    variants = (variant,) if variant is not None else VARIANTS
    model_specs = _model_specs(profile, variants)
    configs = _tuning_configs(profile, variants)
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
    encoded_data = {
        kind: prepare_encoded_data(
            corpus,
            preprocessor_paths[kind],
            max(profile.train_sizes),
            data_seeds,
            progress=progress,
            progress_file=progress_file,
        )
        for kind in preprocessing_kinds
    }

    settings = RunSettings(
        cache_dir=cache_dir,
        train_sizes=profile.train_sizes,
        batch_size=profile.batch_size,
        encoded_data=encoded_data,
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
    scores = (
        tuning.groupby(MODEL_COLUMNS + ["lr"], as_index=False)["val_logloss"]
        .median()
        .rename(columns={"val_logloss": "median_val_logloss"})
    )
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


def default_raw_path(profile_name: str, variant: Variant | None = None) -> Path:
    suffix = f"_{variant}" if variant is not None else ""
    return DEFAULT_RUNS_DIR / f"criteo_scaling_{profile_name}{suffix}.csv"


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
