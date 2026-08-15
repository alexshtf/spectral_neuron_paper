import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import TextIO

import fitstream as fts
import numpy as np
import pandas as pd
import torch

from paper.experiments.higgs_scaling import (
    OPTIMIZER,
    PROFILES,
    RESULT_SCHEMA,
    HiggsModelSpec,
    Profile,
    RunSettings,
    default_raw_path as default_scaling_path,
    make_task_model,
    validate_raw,
)
from paper.experiments.results import DEFAULT_RUNS_DIR, write_csv
from paper.experiments.runner import run_many
from paper.experiments.scaling import (
    PROTOCOL,
    SelectedRun,
    select_evaluation_runs,
)
from paper.higgs import (
    FEATURE_NAMES,
    NUM_FEATURES,
    OFFICIAL_LAYOUT,
    HiggsLayout,
    default_cache_dir,
    prepare_corpus,
)
from paper.models import KthEigval
from paper.shuffling import resolve_train_sizes
from paper.tasks import Batch
from paper.training import BINARY_OBJECTIVE, train_events


NOISE_LEVEL = 0.5
MAGNITUDE_BINS = 16
RATIO_BINS = 100
PERTURBATION_SEED = 0

_BASE_RESULT_COLUMNS = [
    "protocol",
    "optimizer",
    "train_pool_size",
    "train_size",
    "model",
    "dim",
    "lr",
    "data_seed",
    "init_seed",
    "perturbation_seed",
    "noise_level",
    "feature_index",
    "feature_name",
    "feature_matrix_norm",
    "magnitude_bin_index",
    "magnitude_left",
    "magnitude_right",
    "total_count",
    "zero_bound_count",
    "above_bound_count",
    "max_ratio",
]


def ratio_count_columns(bins: int = RATIO_BINS) -> list[str]:
    return [f"ratio_bin_{bin_index:03d}_count" for bin_index in range(bins)]


def result_columns(ratio_bins: int = RATIO_BINS) -> list[str]:
    return _BASE_RESULT_COLUMNS + ratio_count_columns(ratio_bins)


RESULT_COLUMNS = result_columns()

_RUN_COLUMNS = ["dim", "data_seed", "init_seed"]
_FEATURE_COLUMNS = _RUN_COLUMNS + ["feature_index"]
_HISTOGRAM_COLUMNS = _FEATURE_COLUMNS + ["magnitude_bin_index"]


def _noise_level(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("noise level must be finite and positive")
    return value


def _num_bins(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise TypeError("bins must be an integer")
    if value <= 0:
        raise ValueError("bins must be positive")
    return int(value)


def selected_runs(
    scaling_results: pd.DataFrame,
    profile: Profile,
) -> tuple[SelectedRun[HiggsModelSpec], ...]:
    """Select final-checkpoint spectral runs from a HIGGS scaling result."""
    if "model" not in scaling_results:
        raise ValueError("incompatible HIGGS result schema")
    spectral = scaling_results.loc[scaling_results["model"] == "spectral"].copy()
    validate_raw(spectral, profile, variant="spectral")

    train_size = resolve_train_sizes(
        profile.train_sizes, batch_size=profile.batch_size
    )[-1]
    tuning = spectral.loc[
        (spectral["phase"] == "tuning")
        & (spectral["train_size"] == train_size)
    ]
    return select_evaluation_runs(
        tuning,
        schema=RESULT_SCHEMA,
        evaluation_seeds=profile.evaluation_seeds,
        model_specs={
            ("spectral", capacity_dim): HiggsModelSpec("spectral", capacity_dim)
            for capacity_dim in profile.capacity_dims
        },
    )


def feature_matrices(
    model: KthEigval,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return A_1, ..., A_n for a dense spectral neuron."""
    weight = model.lin.weight if dtype is None else model.lin.weight.to(dtype)
    return model.tril_emb(weight.mT)


@dataclass(frozen=True)
class _HistogramChunk:
    ratio_counts: torch.Tensor
    total_counts: torch.Tensor
    zero_bound_counts: torch.Tensor
    above_bound_counts: torch.Tensor
    max_ratios: torch.Tensor


@dataclass
class _HistogramAccumulator:
    ratio_counts: torch.Tensor
    total_counts: torch.Tensor
    zero_bound_counts: torch.Tensor
    above_bound_counts: torch.Tensor
    max_ratios: torch.Tensor

    def accumulate_feature_(
        self, feature_index: int, chunk: _HistogramChunk
    ) -> None:
        self.ratio_counts[feature_index] += chunk.ratio_counts
        self.total_counts[feature_index] += chunk.total_counts
        self.zero_bound_counts[feature_index] += chunk.zero_bound_counts
        self.above_bound_counts[feature_index] += chunk.above_bound_counts
        self.max_ratios[feature_index] = torch.maximum(
            self.max_ratios[feature_index], chunk.max_ratios
        )


def _joint_histogram(
    actual: torch.Tensor,
    bound: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    noise_level: float,
    magnitude_bins: int,
    ratio_bins: int,
) -> _HistogramChunk:
    if not (actual.shape == bound.shape == magnitude.shape):
        raise ValueError("actual deviations, bounds, and magnitudes must align")
    if (
        not torch.isfinite(actual).all()
        or not torch.isfinite(bound).all()
        or not torch.isfinite(magnitude).all()
        or (actual < 0).any()
        or (bound < 0).any()
        or (magnitude < 0).any()
        or (magnitude > noise_level).any()
    ):
        raise ValueError("deviations, bounds, and magnitudes are invalid")

    magnitude_indices = (
        (magnitude * magnitude_bins / noise_level)
        .floor()
        .to(torch.int64)
        .clamp(max=magnitude_bins - 1)
    )
    totals = torch.bincount(magnitude_indices, minlength=magnitude_bins)

    defined = bound > 0
    ratios = torch.zeros_like(actual)
    ratios[defined] = actual[defined] / bound[defined]
    if not torch.isfinite(ratios[defined]).all():
        raise ValueError("deviation ratios must be finite when the bound is nonzero")

    above = defined & (ratios > 1)
    included = defined & ~above
    ratio_indices = (
        (ratios * ratio_bins)
        .floor()
        .to(torch.int64)
        .clamp(max=ratio_bins - 1)
    )
    joint_indices = magnitude_indices * ratio_bins + ratio_indices
    counts = torch.bincount(
        joint_indices[included], minlength=magnitude_bins * ratio_bins
    ).reshape(magnitude_bins, ratio_bins)
    zero_bounds = torch.bincount(
        magnitude_indices[~defined], minlength=magnitude_bins
    )
    above_bounds = torch.bincount(
        magnitude_indices[above], minlength=magnitude_bins
    )
    maxima = torch.full(
        (magnitude_bins,), -torch.inf, dtype=actual.dtype, device=actual.device
    )
    maxima.scatter_reduce_(
        0,
        magnitude_indices[defined],
        ratios[defined],
        reduce="amax",
        include_self=True,
    )
    return _HistogramChunk(counts, totals, zero_bounds, above_bounds, maxima)


def deviation_histograms(
    model: KthEigval,
    batches: Iterable[Batch],
    *,
    noise_level: float = NOISE_LEVEL,
    magnitude_bins: int = MAGNITUDE_BINS,
    ratio_bins: int = RATIO_BINS,
    perturbation_seed: int = PERTURBATION_SEED,
) -> pd.DataFrame:
    """Measure featurewise deviations under one signed uniform perturbation."""
    noise_level = _noise_level(noise_level)
    magnitude_bins = _num_bins(magnitude_bins)
    ratio_bins = _num_bins(ratio_bins)
    was_training = model.training

    weight = model.lin.weight.detach().to(torch.float64)
    bias = model.lin.bias.detach().to(torch.float64)
    matrices = feature_matrices(model, dtype=torch.float64).detach()
    matrix_norms = torch.linalg.matrix_norm(matrices, ord=2)
    device = weight.device
    total_counts = torch.zeros(
        NUM_FEATURES,
        magnitude_bins,
        dtype=torch.int64,
        device=device,
    )
    histograms = _HistogramAccumulator(
        ratio_counts=torch.zeros(
            NUM_FEATURES,
            magnitude_bins,
            ratio_bins,
            dtype=torch.int64,
            device=device,
        ),
        total_counts=total_counts,
        zero_bound_counts=torch.zeros_like(total_counts),
        above_bound_counts=torch.zeros_like(total_counts),
        max_ratios=torch.full(
            (NUM_FEATURES, magnitude_bins),
            -torch.inf,
            dtype=torch.float64,
            device=device,
        ),
    )
    rng = np.random.default_rng(perturbation_seed)
    test_rows = 0

    model.eval()
    try:
        with torch.inference_mode():
            for model_inputs, _ in batches:
                (features,) = model_inputs
                features = features.to(device=device, dtype=torch.float64)
                coordinates = features.matmul(weight.mT) + bias
                base_matrices = model.tril_emb(coordinates)
                base_logits = torch.linalg.eigvalsh(base_matrices)[..., model.eig_idx]
                perturbations = torch.from_numpy(
                    rng.uniform(-noise_level, noise_level, size=features.shape)
                ).to(device=device)

                for feature_index in range(NUM_FEATURES):
                    perturbation = perturbations[:, feature_index]
                    perturbed_matrices = (
                        base_matrices
                        + perturbation[:, None, None] * matrices[feature_index]
                    )
                    perturbed_logits = torch.linalg.eigvalsh(perturbed_matrices)[
                        ..., model.eig_idx
                    ]
                    actual = (perturbed_logits - base_logits).abs()
                    magnitude = perturbation.abs()
                    bound = magnitude * matrix_norms[feature_index]
                    histograms.accumulate_feature_(
                        feature_index,
                        _joint_histogram(
                            actual,
                            bound,
                            magnitude,
                            noise_level=noise_level,
                            magnitude_bins=magnitude_bins,
                            ratio_bins=ratio_bins,
                        ),
                    )
                test_rows += len(features)
    finally:
        model.train(was_training)

    if test_rows == 0:
        raise ValueError("test data must not be empty")

    magnitude_edges = np.linspace(0.0, noise_level, magnitude_bins + 1)
    feature_indices = np.repeat(np.arange(NUM_FEATURES), magnitude_bins)
    magnitude_indices = np.tile(np.arange(magnitude_bins), NUM_FEATURES)
    maxima = histograms.max_ratios.cpu().numpy()
    ratio_counts = histograms.ratio_counts.cpu().numpy()
    maxima[np.isneginf(maxima)] = np.nan
    data = {
        "noise_level": noise_level,
        "feature_index": feature_indices,
        "feature_name": np.asarray(FEATURE_NAMES)[feature_indices],
        "feature_matrix_norm": np.repeat(
            matrix_norms.cpu().numpy(), magnitude_bins
        ),
        "magnitude_bin_index": magnitude_indices,
        "magnitude_left": magnitude_edges[magnitude_indices],
        "magnitude_right": magnitude_edges[magnitude_indices + 1],
        "total_count": histograms.total_counts.cpu().numpy().reshape(-1),
        "zero_bound_count": histograms.zero_bound_counts.cpu().numpy().reshape(-1),
        "above_bound_count": histograms.above_bound_counts.cpu().numpy().reshape(-1),
        "max_ratio": maxima.reshape(-1),
    }
    data.update(
        {
            column: ratio_counts[..., bin_index].reshape(-1)
            for bin_index, column in enumerate(ratio_count_columns(ratio_bins))
        }
    )
    return pd.DataFrame(data)


def run_selected(
    selected: SelectedRun[HiggsModelSpec],
    settings: RunSettings,
    *,
    noise_level: float = NOISE_LEVEL,
    magnitude_bins: int = MAGNITUDE_BINS,
    ratio_bins: int = RATIO_BINS,
    perturbation_seed: int = PERTURBATION_SEED,
) -> pd.DataFrame:
    config = selected.config
    task, model = make_task_model(config, settings)
    if not isinstance(model, KthEigval):
        raise TypeError("HIGGS robustness requires a spectral model")
    (train_size,) = selected.test_checkpoints
    fts.collect(
        train_events(
            task,
            model,
            lr=config.lr,
            checkpoints=(train_size,),
            loss=BINARY_OBJECTIVE.loss,
        ),
        include=(),
    )

    return deviation_histograms(
        model,
        task.test_batches(),
        noise_level=noise_level,
        magnitude_bins=magnitude_bins,
        ratio_bins=ratio_bins,
        perturbation_seed=perturbation_seed,
    ).assign(
        protocol=PROTOCOL,
        optimizer=OPTIMIZER,
        train_pool_size=settings.corpus.train_stop,
        train_size=train_size,
        model=config.model_spec.variant,
        dim=config.model_spec.result_dim,
        lr=config.lr,
        data_seed=config.data_seed,
        init_seed=config.init_seed,
        perturbation_seed=perturbation_seed,
    ).loc[:, result_columns(ratio_bins)]


def run_profile(
    profile: Profile,
    scaling_results: pd.DataFrame,
    *,
    raw_path: Path,
    cache_dir: Path,
    layout: HiggsLayout = OFFICIAL_LAYOUT,
    chunk_size: int = 250_000,
    workers: int = 1,
    noise_level: float = NOISE_LEVEL,
    magnitude_bins: int = MAGNITUDE_BINS,
    ratio_bins: int = RATIO_BINS,
    perturbation_seed: int = PERTURBATION_SEED,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> pd.DataFrame:
    noise_level = _noise_level(noise_level)
    magnitude_bins = _num_bins(magnitude_bins)
    ratio_bins = _num_bins(ratio_bins)
    runs = selected_runs(scaling_results, profile)
    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        layout=layout,
        chunk_size=chunk_size,
        progress=progress,
        progress_file=progress_file,
    )
    spectral_pool_sizes = set(
        scaling_results.loc[
            scaling_results["model"] == "spectral", "train_pool_size"
        ]
    )
    if spectral_pool_sizes != {corpus.train_stop}:
        raise ValueError(
            "scaling results and HIGGS corpus use different training pools"
        )

    (train_size,) = runs[0].test_checkpoints
    required_passes = (train_size + corpus.train_stop - 1) // corpus.train_stop
    for data_seed in sorted({run.config.data_seed for run in runs}):
        corpus.shuffled_epochs(data_seed).prepare(required_passes)

    settings = RunSettings(
        batch_size=profile.batch_size,
        corpus=corpus,
        threads_per_worker=1 if workers > 1 else None,
    )
    run = partial(
        run_selected,
        settings=settings,
        noise_level=noise_level,
        magnitude_bins=magnitude_bins,
        ratio_bins=ratio_bins,
        perturbation_seed=perturbation_seed,
    )
    results = run_many(
        run,
        runs,
        workers=workers,
        desc="Train + perturb",
        unit="model",
        progress=progress,
        progress_file=progress_file,
    )
    if not results:
        return pd.DataFrame(columns=result_columns(ratio_bins))
    return pd.concat(results, ignore_index=True).loc[:, result_columns(ratio_bins)]


def _validate_result_schema(results: pd.DataFrame, ratio_bins: int) -> None:
    columns = result_columns(ratio_bins)
    if list(results.columns) != columns:
        raise ValueError("incompatible HIGGS robustness result schema")
    if results.empty:
        raise ValueError("HIGGS robustness results must not be empty")
    required = [column for column in columns if column != "max_ratio"]
    if results[required].isna().any().any():
        raise ValueError("HIGGS robustness results contain missing values")


def _validate_run_metadata(
    results: pd.DataFrame,
    profile: Profile,
    *,
    noise_level: float,
    perturbation_seed: int,
) -> None:
    if set(results["protocol"]) != {PROTOCOL}:
        raise ValueError(f"expected protocol={PROTOCOL!r}")
    if set(results["optimizer"]) != {OPTIMIZER}:
        raise ValueError(f"expected optimizer={OPTIMIZER!r}")
    if set(results["model"]) != {"spectral"}:
        raise ValueError("HIGGS robustness results must contain spectral models")
    if set(results["perturbation_seed"]) != {perturbation_seed}:
        raise ValueError("unexpected perturbation seed")
    if not np.allclose(results["noise_level"], noise_level, rtol=0, atol=0):
        raise ValueError("unexpected noise level")

    train_size = resolve_train_sizes(
        profile.train_sizes, batch_size=profile.batch_size
    )[-1]
    if set(results["train_size"]) != {train_size}:
        raise ValueError("unexpected robustness training checkpoint")
    if results.groupby("dim")["lr"].nunique().ne(1).any():
        raise ValueError("each matrix dimension must use one learning rate")


def _validate_histogram_grid(
    results: pd.DataFrame,
    profile: Profile,
    *,
    noise_level: float,
    magnitude_bins: int,
) -> None:
    feature_labels = set(
        results[["feature_index", "feature_name"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if feature_labels != set(enumerate(FEATURE_NAMES)):
        raise ValueError("HIGGS feature labels are incomplete")

    expected_histograms = {
        (dim, data_seed, init_seed, feature_index, magnitude_bin_index)
        for dim, (data_seed, init_seed), feature_index, magnitude_bin_index in product(
            profile.capacity_dims,
            profile.evaluation_seeds,
            range(NUM_FEATURES),
            range(magnitude_bins),
        )
    }
    observed_histograms = set(
        results[_HISTOGRAM_COLUMNS].itertuples(index=False, name=None)
    )
    if observed_histograms != expected_histograms:
        raise ValueError("incomplete robustness histogram grid")
    if results.duplicated(_HISTOGRAM_COLUMNS).any():
        raise ValueError("duplicate robustness histogram bins")

    magnitude_indices = results["magnitude_bin_index"].to_numpy(dtype=int)
    magnitude_edges = np.linspace(0.0, noise_level, magnitude_bins + 1)
    if not (
        np.allclose(results["magnitude_left"], magnitude_edges[magnitude_indices])
        and np.allclose(
            results["magnitude_right"], magnitude_edges[magnitude_indices + 1]
        )
    ):
        raise ValueError("invalid magnitude bin edges")


def _valid_counts(values: np.ndarray) -> bool:
    return bool(
        np.isfinite(values).all()
        and (values >= 0).all()
        and np.equal(values, np.floor(values)).all()
    )


def _validate_histogram_values(
    results: pd.DataFrame,
    *,
    ratio_bins: int,
) -> None:
    ratio_counts = results[ratio_count_columns(ratio_bins)].to_numpy()
    if not _valid_counts(ratio_counts):
        raise ValueError("invalid robustness histogram counts")

    total_counts = results["total_count"].to_numpy()
    zero_bound_counts = results["zero_bound_count"].to_numpy()
    above_bound_counts = results["above_bound_count"].to_numpy()
    if not all(
        _valid_counts(counts)
        for counts in (total_counts, zero_bound_counts, above_bound_counts)
    ) or not (
        ratio_counts.sum(axis=1) + zero_bound_counts + above_bound_counts
        == total_counts
    ).all():
        raise ValueError("invalid robustness histogram count accounting")

    feature_groups = results.groupby(_FEATURE_COLUMNS, sort=False)
    if feature_groups["feature_matrix_norm"].nunique(dropna=False).ne(1).any():
        raise ValueError("inconsistent feature matrix norms")
    feature_norms = feature_groups["feature_matrix_norm"].first()
    if not np.isfinite(feature_norms).all() or (feature_norms < 0).any():
        raise ValueError("invalid feature matrix norms")
    test_counts = feature_groups["total_count"].sum()
    if (test_counts <= 0).any() or test_counts.nunique() != 1:
        raise ValueError("inconsistent test-set counts")

    defined = total_counts > zero_bound_counts
    max_ratios = results["max_ratio"].to_numpy()
    if (
        not np.isfinite(max_ratios[defined]).all()
        or (max_ratios[defined] < 0).any()
        or not np.isnan(max_ratios[~defined]).all()
    ):
        raise ValueError("invalid maximum deviation ratios")


def validate_results(
    results: pd.DataFrame,
    profile: Profile,
    *,
    noise_level: float = NOISE_LEVEL,
    magnitude_bins: int = MAGNITUDE_BINS,
    ratio_bins: int = RATIO_BINS,
    perturbation_seed: int = PERTURBATION_SEED,
) -> None:
    """Validate a complete HIGGS robustness result."""
    noise_level = _noise_level(noise_level)
    magnitude_bins = _num_bins(magnitude_bins)
    ratio_bins = _num_bins(ratio_bins)
    _validate_result_schema(results, ratio_bins)
    _validate_run_metadata(
        results,
        profile,
        noise_level=noise_level,
        perturbation_seed=perturbation_seed,
    )
    _validate_histogram_grid(
        results,
        profile,
        noise_level=noise_level,
        magnitude_bins=magnitude_bins,
    )
    _validate_histogram_values(results, ratio_bins=ratio_bins)


def _float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def default_result_path(
    profile_name: str,
    noise_level: float = NOISE_LEVEL,
) -> Path:
    return DEFAULT_RUNS_DIR / (
        f"higgs_robustness_{profile_name}_noise_{_float_label(_noise_level(noise_level))}"
        "_repeated_shuffle.csv.zst"
    )


def build_arg_parser(
    profiles: Mapping[str, Profile] = PROFILES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Headerless HIGGS CSV."
    )
    parser.add_argument("--profile", choices=profiles.keys(), default="sanity")
    parser.add_argument("--scaling-results", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--noise-level",
        type=_noise_level,
        default=NOISE_LEVEL,
        help="Maximum absolute perturbation in standardized feature units.",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    profile = PROFILES[args.profile]
    scaling_path = args.scaling_results or default_scaling_path(args.profile)
    results = run_profile(
        profile,
        pd.read_csv(scaling_path),
        raw_path=args.data,
        cache_dir=args.cache_dir or default_cache_dir(args.data),
        chunk_size=args.chunk_size,
        workers=args.workers,
        noise_level=args.noise_level,
        progress=not args.quiet,
    )
    validate_results(results, profile, noise_level=args.noise_level)
    write_csv(
        results,
        args.out or default_result_path(args.profile, args.noise_level),
    )


if __name__ == "__main__":
    main()
