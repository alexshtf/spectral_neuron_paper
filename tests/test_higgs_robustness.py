from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.experiments.higgs_robustness import (
    RESULT_COLUMNS,
    _joint_histogram,
    build_arg_parser,
    default_result_path,
    deviation_histograms,
    feature_matrices,
    ratio_count_columns,
    result_columns,
    run_profile,
    selected_configs,
    validate_results,
)
from paper.experiments.higgs_scaling import (
    RAW_COLUMNS,
    HiggsModelSpec,
    Profile,
    SeedGrid,
    spectral_parameter_count,
)
from paper.higgs import FEATURE_NAMES, NUM_FEATURES, HiggsLayout, default_cache_dir
from paper.models import KthEigval


TINY_LAYOUT = HiggsLayout(rows=16, train_stop=8, val_stop=12)


def _profile() -> Profile:
    return Profile(
        train_sizes=(4, 8),
        capacity_dims=(1, 2),
        lrs=(1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(
            data_seeds=range(1, 3), init_seeds=range(2, 4)
        ),
        batch_size=4,
    )


def _scaling_results(profile: Profile) -> pd.DataFrame:
    selected_lrs = {
        (1, 4): 1e-2,
        (1, 8): 1e-1,
        (2, 4): 1e-1,
        (2, 8): 1e-2,
    }
    rows = []
    for dim, train_size, lr, (data_seed, init_seed) in product(
        profile.capacity_dims,
        profile.train_sizes,
        profile.lrs,
        profile.tuning_seeds,
    ):
        rows.append(
            {
                "protocol": "repeated_shuffle",
                "optimizer": "adam",
                "train_pool_size": TINY_LAYOUT.train_stop,
                "phase": "tuning",
                "train_size": train_size,
                "data_seed": data_seed,
                "model": "spectral",
                "dim": dim,
                "width": 0,
                "num_parameters": spectral_parameter_count(NUM_FEATURES, dim),
                "lr": lr,
                "init_seed": init_seed,
                "val_logloss": (0.1 if lr == selected_lrs[dim, train_size] else 0.3)
                + init_seed / 100,
                "test_logloss": np.nan,
                "test_brier": np.nan,
            }
        )
    for dim, train_size, (data_seed, init_seed) in product(
        profile.capacity_dims,
        profile.train_sizes,
        profile.evaluation_seeds,
    ):
        rows.append(
            {
                "protocol": "repeated_shuffle",
                "optimizer": "adam",
                "train_pool_size": TINY_LAYOUT.train_stop,
                "phase": "evaluation",
                "train_size": train_size,
                "data_seed": data_seed,
                "model": "spectral",
                "dim": dim,
                "width": 0,
                "num_parameters": spectral_parameter_count(NUM_FEATURES, dim),
                "lr": selected_lrs[dim, train_size],
                "init_seed": init_seed,
                "val_logloss": np.nan,
                "test_logloss": 0.5,
                "test_brier": 0.25,
            }
        )
    return pd.DataFrame(rows).loc[:, RAW_COLUMNS]


def _write_tiny_higgs(path: Path) -> None:
    row = np.arange(TINY_LAYOUT.rows, dtype=np.float32)[:, None]
    field = np.arange(NUM_FEATURES, dtype=np.float32)[None, :]
    features = row + field / 10
    labels = (np.arange(TINY_LAYOUT.rows) % 2).astype(np.float32)[:, None]
    np.savetxt(path, np.concatenate((labels, features), axis=1), delimiter=",")


def test_selected_configs_use_final_validation_lrs_and_evaluation_seed_grid():
    profile = _profile()
    configs = selected_configs(_scaling_results(profile), profile)

    assert len(configs) == len(profile.capacity_dims) * len(
        profile.evaluation_seeds
    )
    assert {
        (config.model_spec.capacity_dim, config.lr) for config in configs
    } == {(1, 1e-1), (2, 1e-2)}
    assert {
        (config.data_seed, config.init_seed) for config in configs
    } == set(profile.evaluation_seeds)
    assert all(config.model_spec.variant == "spectral" for config in configs)


def test_feature_matrices_use_the_model_tril_embedding():
    model = KthEigval(NUM_FEATURES, dim=2)
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[:, 0] = torch.tensor([1.0, 2 * sqrt(2), 3.0])

    matrices = feature_matrices(model)

    torch.testing.assert_close(
        matrices[0], torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    )
    torch.testing.assert_close(matrices[1:], torch.zeros_like(matrices[1:]))


def test_feature_matrices_promote_before_embedding():
    with torch.random.fork_rng():
        torch.manual_seed(2)
        model = KthEigval(NUM_FEATURES, dim=4)
        with torch.no_grad():
            model.lin.weight.normal_()

    matrices = feature_matrices(model, dtype=torch.float64)
    expected = model.tril_emb(model.lin.weight.to(torch.float64).mT)
    embedded_then_promoted = model.tril_emb(model.lin.weight.mT).to(torch.float64)

    torch.testing.assert_close(matrices, expected, rtol=0, atol=0)
    assert not torch.equal(matrices, embedded_then_promoted)
    assert matrices.dtype == torch.float64


def test_joint_histogram_uses_disjoint_magnitude_bins_and_preserves_diagnostics():
    actual = torch.tensor([0.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    bound = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    magnitude = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

    counts, total, zero, above, maximum = _joint_histogram(
        actual,
        bound,
        magnitude,
        noise_level=1.0,
        magnitude_bins=4,
        ratio_bins=4,
    )

    assert total.tolist() == [1, 1, 1, 2]
    assert counts.nonzero().tolist() == [[1, 0], [2, 2], [3, 3]]
    assert zero.tolist() == [1, 0, 0, 0]
    assert above.tolist() == [0, 0, 0, 1]
    assert maximum.tolist() == [-torch.inf, 0.0, 0.5, 2.0]


def test_dim_one_reference_has_one_unit_ratio_per_row_and_restores_model():
    model = KthEigval(NUM_FEATURES, dim=1, eig_idx=0)
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[0, 0] = 1
        model.lin.bias.zero_()
    model.train()
    features = torch.zeros(3, NUM_FEATURES)
    labels = torch.zeros(3)

    results = deviation_histograms(
        model,
        [((features,), labels)],
        noise_level=0.5,
        magnitude_bins=4,
        ratio_bins=100,
    )

    first = results.loc[results["feature_index"] == 0]
    assert first["ratio_bin_099_count"].sum() == 3
    assert first["above_bound_count"].sum() == 0
    assert first["max_ratio"].max() == pytest.approx(1.0)
    zero = results.loc[results["feature_index"] == 1]
    assert zero[ratio_count_columns()].to_numpy().sum() == 0
    assert zero["zero_bound_count"].sum() == 3
    assert zero["max_ratio"].isna().all()
    assert results.groupby("feature_index")["total_count"].sum().eq(3).all()
    assert model.training
    assert model.lin.weight.dtype == torch.float32


def test_uniform_perturbations_use_both_signs():
    model = KthEigval(NUM_FEATURES, dim=2, eig_idx=1)
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[0, 0] = 1
        model.lin.bias.zero_()
    features = torch.zeros(12, NUM_FEATURES)
    labels = torch.zeros(12)
    seed = 5

    results = deviation_histograms(
        model,
        [((features,), labels)],
        noise_level=1.0,
        magnitude_bins=1,
        ratio_bins=2,
        perturbation_seed=seed,
    )

    perturbations = np.random.default_rng(seed).uniform(
        -1, 1, size=features.shape
    )[:, 0]
    feature = results.loc[results["feature_index"] == 0].iloc[0]
    assert feature["ratio_bin_000_count"] == np.count_nonzero(perturbations < 0)
    assert feature["ratio_bin_001_count"] == np.count_nonzero(perturbations > 0)


def test_histograms_do_not_depend_on_test_batch_partition():
    with torch.random.fork_rng():
        torch.manual_seed(8)
        model = KthEigval(NUM_FEATURES, dim=3, eig_idx=1)
        features = torch.randn(7, NUM_FEATURES)
    labels = torch.zeros(7)

    whole = deviation_histograms(
        model,
        [((features,), labels)],
        noise_level=0.25,
        magnitude_bins=4,
        ratio_bins=20,
    )
    split = deviation_histograms(
        model,
        [
            ((features[:2],), labels[:2]),
            ((features[2:],), labels[2:]),
        ],
        noise_level=0.25,
        magnitude_bins=4,
        ratio_bins=20,
    )

    pd.testing.assert_frame_equal(whole, split)


def test_matrix_perturbations_match_model_and_satisfy_weyl_bound():
    with torch.random.fork_rng():
        torch.manual_seed(13)
        model = KthEigval(NUM_FEATURES, dim=4, eig_idx=2).double()
        with torch.no_grad():
            model.lin.weight.normal_()
            model.lin.bias.normal_()
        features = torch.randn(9, NUM_FEATURES, dtype=torch.float64)
        perturbation = torch.empty(9, dtype=torch.float64).uniform_(-0.5, 0.5)

    feature_index = 7
    matrices = feature_matrices(model)
    base_matrices = model.tril_emb(model.lin(features))
    perturbed_matrices = (
        base_matrices + perturbation[:, None, None] * matrices[feature_index]
    )
    direct = torch.linalg.eigvalsh(perturbed_matrices)[..., model.eig_idx]
    perturbed_features = features.clone()
    perturbed_features[:, feature_index] += perturbation
    through_model = model(perturbed_features)

    torch.testing.assert_close(through_model, direct, rtol=1e-12, atol=1e-12)
    actual = (through_model - model(features)).abs()
    bound = perturbation.abs() * torch.linalg.matrix_norm(
        matrices[feature_index], ord=2
    )
    assert torch.all(actual <= bound + 1e-12)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("higgs-robustness")
    raw_path = directory / "HIGGS.csv"
    _write_tiny_higgs(raw_path)
    profile = _profile()
    results = run_profile(
        profile,
        _scaling_results(profile),
        raw_path=raw_path,
        cache_dir=directory / "cache",
        layout=TINY_LAYOUT,
        chunk_size=3,
        noise_level=0.5,
        magnitude_bins=4,
        ratio_bins=5,
    )
    return profile, results


def test_tiny_profile_runs_end_to_end_and_validates(tiny_run):
    profile, results = tiny_run

    validate_results(
        results,
        profile,
        noise_level=0.5,
        magnitude_bins=4,
        ratio_bins=5,
    )
    assert list(results.columns) == result_columns(5)
    assert RESULT_COLUMNS == result_columns()
    assert len(results) == (
        len(profile.capacity_dims) * len(profile.evaluation_seeds) * 28 * 4
    )
    totals = results.groupby(
        ["dim", "data_seed", "init_seed", "feature_index"]
    )["total_count"].sum()
    assert set(totals) == {TINY_LAYOUT.rows - TINY_LAYOUT.val_stop}
    assert set(results["feature_name"]) == set(FEATURE_NAMES)


def test_result_validation_rejects_missing_bins_and_bad_accounting(tiny_run):
    profile, results = tiny_run
    with pytest.raises(ValueError, match="histogram|bins"):
        validate_results(
            results.iloc[1:],
            profile,
            noise_level=0.5,
            magnitude_bins=4,
            ratio_bins=5,
        )

    invalid = results.copy()
    invalid.loc[0, "ratio_bin_000_count"] += 1
    with pytest.raises(ValueError, match="accounting"):
        validate_results(
            invalid,
            profile,
            noise_level=0.5,
            magnitude_bins=4,
            ratio_bins=5,
        )


def test_result_validation_rejects_a_truncated_model_run(tiny_run):
    profile, results = tiny_run
    truncated = results.copy()
    run = (
        (truncated["dim"] == profile.capacity_dims[0])
        & (truncated["data_seed"] == profile.evaluation_seeds.data_seeds[0])
        & (truncated["init_seed"] == profile.evaluation_seeds.init_seeds[0])
    )
    count_columns = ratio_count_columns(5) + [
        "zero_bound_count",
        "above_bound_count",
    ]
    for feature_index in range(NUM_FEATURES):
        row = truncated.index[
            run
            & (truncated["feature_index"] == feature_index)
            & (truncated["total_count"] > 0)
        ][0]
        count_column = next(
            column for column in count_columns if truncated.at[row, column] > 0
        )
        truncated.at[row, count_column] -= 1
        truncated.at[row, "total_count"] -= 1
        if (
            truncated.at[row, "total_count"]
            == truncated.at[row, "zero_bound_count"]
        ):
            truncated.at[row, "max_ratio"] = np.nan

    with pytest.raises(ValueError, match="test-set counts"):
        validate_results(
            truncated,
            profile,
            noise_level=0.5,
            magnitude_bins=4,
            ratio_bins=5,
        )


def test_result_validation_accepts_the_requested_perturbation_seed(tiny_run):
    profile, results = tiny_run
    results = results.assign(perturbation_seed=17)

    validate_results(
        results,
        profile,
        noise_level=0.5,
        magnitude_bins=4,
        ratio_bins=5,
        perturbation_seed=17,
    )


def test_cli_and_default_path_include_the_noise_level():
    args = build_arg_parser().parse_args(
        ["--data", "HIGGS.csv", "--noise-level", "0.25"]
    )

    assert args.noise_level == 0.25
    assert default_result_path("full", 0.25).name == (
        "higgs_robustness_full_noise_0p25_repeated_shuffle.csv.zst"
    )
    assert default_cache_dir(Path("HIGGS.csv")).name == ".HIGGS.csv.cache-v1"
