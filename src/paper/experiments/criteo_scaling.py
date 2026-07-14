import argparse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from paper.criteo import (
    NUM_FIELDS,
    CriteoCorpus,
    CriteoTask,
    PreprocessingKind,
    fit_preprocessors,
    load_preprocessor,
    prepare_corpus,
)
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, WRITE_MODES, _write_csv
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

RAW_COLUMNS = [
    "protocol",
    "preprocessor_sample_size",
    "preprocessor_seed",
    "train_size",
    "data_seed",
    "model",
    "preprocessing",
    "matrix_dim",
    "eig_idx",
    "fm_rank",
    "parameters_per_feature",
    "num_parameters",
    "lr",
    "init_seed",
    "train_logloss",
    "val_logloss",
    "val_brier",
    "test_logloss",
    "test_brier",
    "elapsed_seconds",
]

TEST_COLUMNS = ["test_logloss", "test_brier"]
TUNING_COLUMNS = [column for column in RAW_COLUMNS if column not in TEST_COLUMNS]

EXPERIMENT_COLUMNS = [
    "protocol",
    "preprocessor_sample_size",
    "preprocessor_seed",
]

MODEL_COLUMNS = EXPERIMENT_COLUMNS + [
    "train_size",
    "model",
    "preprocessing",
    "matrix_dim",
    "eig_idx",
    "fm_rank",
    "parameters_per_feature",
    "num_parameters",
]

RUN_COLUMNS = EXPERIMENT_COLUMNS + [
    "train_size",
    "data_seed",
    "model",
    "preprocessing",
    "matrix_dim",
    "eig_idx",
    "fm_rank",
    "parameters_per_feature",
    "num_parameters",
    "lr",
    "init_seed",
]


@dataclass(frozen=True)
class Profile:
    train_sizes: tuple[int, ...]
    dims: tuple[int, ...]
    lrs: tuple[float, ...]
    init_seeds: range
    data_seeds: range = range(1)
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
        init_seeds=range(1),
        batch_size=256,
        min_count=2,
        buckets_per_field=2**8,
    ),
    "small": Profile(
        train_sizes=(2**14, 2**18, 2**22),
        dims=(3, 5),
        lrs=(1e-3, 1e-2, 1e-1),
        init_seeds=range(2),
        batch_size=4096,
    ),
    "full": Profile(
        train_sizes=(*tuple(2**power for power in range(14, 26, 2)), 2**25, 36_672_493),
        dims=(3, 5, 9, 15),
        lrs=tuple(np.geomspace(1e-3, 1e-1, 8).tolist()),
        init_seeds=range(3),
        batch_size=4096,
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    variant: Variant
    matrix_dim: int = 0
    fm_rank: int = 0
    parameters_per_feature: int = 1

    @property
    def preprocessing(self) -> PreprocessingKind:
        return "hybrid" if self.variant.endswith("-new") else "bucket"

    @classmethod
    def fm_for_dim(cls, dim: int) -> "ModelSpec":
        parameters = dim * (dim + 1) // 2
        return cls(
            variant="fm",
            fm_rank=parameters - 1,
            parameters_per_feature=parameters,
        )

    @classmethod
    def spectral(
        cls,
        variant: Literal["spectral-old", "spectral-new"],
        dim: int,
    ) -> "ModelSpec":
        parameters = dim * (dim + 1) // 2
        return cls(
            variant=variant,
            matrix_dim=dim,
            parameters_per_feature=parameters,
        )


@dataclass(frozen=True)
class RunConfig:
    data_seed: int
    model_spec: ModelSpec
    lr: float
    init_seed: int


@dataclass(frozen=True)
class RunGrid:
    profile: Profile
    variants: tuple[Variant, ...] = VARIANTS

    @property
    def model_specs(self) -> tuple[ModelSpec, ...]:
        specs = [ModelSpec("linear"), ModelSpec("linear-new")]
        specs.extend(
            spec
            for dim in self.profile.dims
            for spec in (
                ModelSpec.fm_for_dim(dim),
                ModelSpec.spectral("spectral-old", dim),
                ModelSpec.spectral("spectral-new", dim),
            )
        )
        return tuple(spec for spec in specs if spec.variant in self.variants)

    def __len__(self) -> int:
        return (
            len(self.profile.data_seeds)
            * len(self.model_specs)
            * len(self.profile.lrs)
            * len(self.profile.init_seeds)
        )

    def __iter__(self) -> Iterator[RunConfig]:
        for data_seed, model_spec, lr, init_seed in product(
            self.profile.data_seeds,
            self.model_specs,
            self.profile.lrs,
            self.profile.init_seeds,
        ):
            yield RunConfig(data_seed, model_spec, lr, init_seed)


@dataclass(frozen=True)
class RunSettings:
    cache_dir: Path
    train_sizes: tuple[int, ...]
    batch_size: int
    preprocessor_paths: dict[PreprocessingKind, Path]
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
            return FactorizationMachine(num_features, NUM_FIELDS, spec.fm_rank)
        case "spectral-old" | "spectral-new":
            return SparseKthEigval(num_features, NUM_FIELDS, spec.matrix_dim)
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
    preprocessor = load_preprocessor(settings.preprocessor_paths[preprocessing])
    order = np.load(corpus.order_path(config.data_seed), mmap_mode="r")
    task = CriteoTask(
        corpus=corpus,
        preprocessor=preprocessor,
        train_rows=order[: max(settings.train_sizes)],
        batch_size=settings.batch_size,
    )
    model = _make_seeded_model(
        config.model_spec,
        num_features=preprocessor.num_features,
        init_seed=config.init_seed,
    )
    return task, model


def _metadata(
    config: RunConfig,
    model: nn.Module,
    settings: RunSettings,
) -> dict[str, int | float | str]:
    return {
        "protocol": PROTOCOL,
        "preprocessor_sample_size": settings.preprocessor_sample_size,
        "preprocessor_seed": settings.preprocessor_seed,
        "data_seed": config.data_seed,
        "model": config.model_spec.variant,
        "preprocessing": config.model_spec.preprocessing,
        "matrix_dim": config.model_spec.matrix_dim,
        "eig_idx": (
            config.model_spec.matrix_dim // 2
            if config.model_spec.variant.startswith("spectral-")
            else -1
        ),
        "fm_rank": config.model_spec.fm_rank,
        "parameters_per_feature": config.model_spec.parameters_per_feature,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "lr": config.lr,
        "init_seed": config.init_seed,
    }


def run_config(config: RunConfig, settings: RunSettings) -> pd.DataFrame:
    task, model = _make_task_model(config, settings)
    result = tune_binary_scaling_stream(
        task,
        model,
        lr=config.lr,
        checkpoints=settings.train_sizes,
    )
    return result.assign(**_metadata(config, model, settings)).loc[:, TUNING_COLUMNS]


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
    return result.assign(**_metadata(selected.config, model, settings)).loc[
        :, RUN_COLUMNS + TEST_COLUMNS
    ]


def _selected_runs(tuning: pd.DataFrame) -> list[SelectedRun]:
    train_sizes: dict[RunConfig, list[int]] = {}
    for row in select_lr(tuning).itertuples(index=False):
        config = RunConfig(
            data_seed=row.data_seed,
            model_spec=ModelSpec(
                row.model,
                matrix_dim=row.matrix_dim,
                fm_rank=row.fm_rank,
                parameters_per_feature=row.parameters_per_feature,
            ),
            lr=row.lr,
            init_seed=row.init_seed,
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
    configs = RunGrid(profile, variants)
    preprocessing_kinds = tuple(
        dict.fromkeys(spec.preprocessing for spec in configs.model_specs)
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
    for data_seed in profile.data_seeds:
        corpus.order_path(data_seed)

    settings = RunSettings(
        cache_dir=cache_dir,
        train_sizes=profile.train_sizes,
        batch_size=profile.batch_size,
        preprocessor_paths=preprocessor_paths,
        preprocessor_sample_size=sample_size,
        preprocessor_seed=profile.preprocessor_seed,
        threads_per_worker=1 if workers > 1 else None,
    )
    tune = partial(run_config, settings=settings)
    if workers == 1:
        items = tqdm(
            configs,
            total=len(configs),
            desc="Tuning",
            unit="trajectory",
            disable=not progress,
            file=progress_file,
        )
        tuning_results = [tune(config) for config in items]
    else:
        tuning_results = process_map(
            tune,
            configs,
            max_workers=workers,
            chunksize=1,
            desc="Tuning",
            unit="trajectory",
            disable=not progress,
            file=progress_file,
        )

    if not tuning_results:
        return pd.DataFrame(columns=RAW_COLUMNS)
    tuning = pd.concat(tuning_results, ignore_index=True)
    selected_runs = _selected_runs(tuning)

    test = partial(run_selected, settings=settings)
    if workers == 1:
        items = tqdm(
            selected_runs,
            desc="Testing",
            unit="trajectory",
            disable=not progress,
            file=progress_file,
        )
        test_results = [test(selected) for selected in items]
    else:
        test_results = process_map(
            test,
            selected_runs,
            max_workers=workers,
            chunksize=1,
            desc="Testing",
            unit="trajectory",
            disable=not progress,
            file=progress_file,
        )

    tests = pd.concat(test_results, ignore_index=True)
    return tuning.merge(tests, on=RUN_COLUMNS, how="left").loc[:, RAW_COLUMNS]


def select_lr(raw: pd.DataFrame) -> pd.DataFrame:
    scores = (
        raw.groupby(MODEL_COLUMNS + ["lr"], as_index=False)["val_logloss"]
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
    selected = raw.merge(
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
    parser.add_argument("--summary-out", type=Path, default=None)
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
    _write_csv(
        raw,
        args.out or default_raw_path(args.profile, args.variant),
        write_mode=args.write_mode,
    )
    if args.summary_out is not None:
        _write_csv(summarize_raw(raw), args.summary_out, write_mode=args.write_mode)


if __name__ == "__main__":
    main()
