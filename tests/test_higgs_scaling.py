from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.experiments.higgs_scaling import (
    VARIANTS,
    HiggsModelSpec,
    Profile,
    SeedGrid,
    _make_mlp,
    default_raw_path,
    make_model,
    matched_mlp_width,
    mlp_parameter_count,
    run_profile,
    spectral_parameter_count,
    trainable_parameter_count,
    validate_raw,
)
from paper.higgs import NUM_FEATURES, HiggsLayout
from paper.models import KthEigval


def _write_tiny_higgs(path: Path, rows: int) -> None:
    row = np.arange(rows, dtype=np.float32)[:, None]
    field = np.arange(NUM_FEATURES, dtype=np.float32)[None, :]
    features = ((row * (field + 1)) % 17 - 8) / 4
    labels = (np.arange(rows) % 2).astype(np.float32)[:, None]
    np.savetxt(path, np.concatenate((labels, features), axis=1), delimiter=",")


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(8, 16),
        capacity_dims=(3,),
        lrs=(1e-2, 1e-1),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=8,
    )


@pytest.mark.parametrize(
    "spec",
    [
        HiggsModelSpec("linear"),
        HiggsModelSpec("spectral", 3),
        HiggsModelSpec("mlp-1", 3),
    ],
)
def test_model_spec_separates_capacity_from_persisted_dimension(spec):
    assert spec.result_dim == (spec.capacity_dim or 0)


@pytest.mark.parametrize(
    "variant, capacity_dim",
    [("linear", 3), ("spectral", None), ("mlp-1", 0)],
)
def test_model_spec_rejects_incoherent_capacity(variant, capacity_dim):
    with pytest.raises(ValueError, match="capacity_dim"):
        HiggsModelSpec(variant, capacity_dim)


@pytest.fixture(scope="module")
def complete_raw(tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-higgs-run")
    raw_path = directory / "HIGGS.csv"
    _write_tiny_higgs(raw_path, 32)
    return run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=directory / "cache",
        layout=HiggsLayout(rows=32, train_stop=16, val_stop=24),
        chunk_size=7,
    )


def test_matched_width_agrees_with_brute_force():
    for input_dim in (1, 2, 7, 28):
        for depth in (1, 2, 3, 5):
            for target in (1, 2, 7, 31, 100, 499):
                candidates = range(1, target + 2)
                expected = min(
                    candidates,
                    key=lambda width: (
                        abs(
                            mlp_parameter_count(input_dim, width, depth) - target
                        ),
                        width,
                    ),
                )
                assert matched_mlp_width(input_dim, depth, target) == expected


def test_matched_width_uses_smaller_width_to_break_ties():
    assert mlp_parameter_count(2, 1, 1) == 5
    assert mlp_parameter_count(2, 2, 1) == 9
    assert matched_mlp_width(2, 1, 7) == 1
    assert matched_mlp_width(28, 3, 1) == 1


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (3, (6, 5, 4)),
        (5, (14, 10, 9)),
        (9, (43, 24, 19)),
        (15, (116, 45, 34)),
    ],
)
def test_higgs_widths_are_computed_from_parameter_budget(dim, expected):
    widths = tuple(
        HiggsModelSpec(f"mlp-{depth}", dim).width(NUM_FEATURES)
        for depth in (1, 2, 3)
    )
    assert widths == expected


def test_parameter_formulas_match_constructed_models():
    for input_dim in (1, 7, NUM_FEATURES):
        for dim in (2, 3, 5):
            spectral = KthEigval(input_dim, dim, eig_idx=dim // 2)
            assert trainable_parameter_count(
                spectral
            ) == spectral_parameter_count(input_dim, dim)
        for depth in (1, 2, 3):
            for width in (1, 4, 11):
                mlp = _make_mlp(input_dim, width, depth)
                assert trainable_parameter_count(mlp) == mlp_parameter_count(
                    input_dim, width, depth
                )


def test_make_model_uses_the_dynamically_matched_width():
    spec = HiggsModelSpec("mlp-3", 9)

    model = make_model(spec)
    width = matched_mlp_width(
        NUM_FEATURES,
        3,
        spectral_parameter_count(NUM_FEATURES, 9),
    )
    assert isinstance(model, torch.nn.Sequential)
    linear_layers = [
        layer for layer in model if isinstance(layer, torch.nn.Linear)
    ]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (NUM_FEATURES, width),
        (width, width),
        (width, width),
        (width, 1),
    ]
    assert trainable_parameter_count(model) == mlp_parameter_count(
        NUM_FEATURES, width, 3
    )


@pytest.mark.parametrize(
    "spec",
    [HiggsModelSpec("linear"), HiggsModelSpec("mlp-2", 3)],
)
def test_dense_higgs_models_preserve_batch_dimensions(spec):
    model = make_model(spec)

    assert model(torch.zeros(7, NUM_FEATURES)).shape == (7,)
    assert model(torch.zeros(2, 7, NUM_FEATURES)).shape == (2, 7)


def test_tiny_profile_runs_all_model_families(complete_raw):
    raw = complete_raw

    assert set(raw["model"]) == set(VARIANTS)
    assert set(raw["protocol"]) == {"repeated_shuffle"}
    assert set(raw["optimizer"]) == {"adam"}
    assert set(raw["train_pool_size"]) == {16}


def test_default_result_path_does_not_target_legacy_one_pass_results():
    assert (
        default_raw_path("full").name
        == "higgs_scaling_full_repeated_shuffle.csv"
    )
    assert (
        default_raw_path("full", "spectral").name
        == "higgs_scaling_full_repeated_shuffle_spectral.csv"
    )


def test_validate_raw_accepts_complete_and_variant_sharded_results(complete_raw):
    validate_raw(complete_raw, _tiny_profile())
    linear = complete_raw.loc[complete_raw["model"] == "linear"].copy()
    validate_raw(linear, _tiny_profile(), variant="linear")


@pytest.mark.parametrize(
    ("column", "value"),
    [("protocol", "other"), ("optimizer", "sgd")],
)
def test_validate_raw_checks_experiment_metadata(complete_raw, column, value):
    with pytest.raises(ValueError, match=column):
        validate_raw(complete_raw.assign(**{column: value}), _tiny_profile())


def test_validate_raw_checks_capacity_metadata(complete_raw):
    wrong_capacity = complete_raw.copy()
    index = wrong_capacity.index[wrong_capacity["model"] == "mlp-1"][0]
    wrong_capacity.loc[index, "width"] += 1
    with pytest.raises(ValueError, match="capacity metadata"):
        validate_raw(wrong_capacity, _tiny_profile())


def test_parallel_profile_matches_serial_results(tmp_path):
    raw_path = tmp_path / "HIGGS.csv"
    cache_dir = tmp_path / "cache"
    _write_tiny_higgs(raw_path, 32)
    kwargs = {
        "raw_path": raw_path,
        "cache_dir": cache_dir,
        "layout": HiggsLayout(rows=32, train_stop=16, val_stop=24),
        "chunk_size": 7,
    }

    serial = run_profile(_tiny_profile(), workers=1, **kwargs)
    parallel = run_profile(_tiny_profile(), workers=2, **kwargs)

    sort_by = [
        "phase",
        "model",
        "dim",
        "lr",
        "data_seed",
        "init_seed",
        "train_size",
    ]
    serial = serial.sort_values(sort_by).reset_index(drop=True)
    parallel = parallel.sort_values(sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-6)
