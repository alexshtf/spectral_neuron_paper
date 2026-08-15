from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.experiments.higgs_scaling import (
    RESULT_SCHEMA,
    VARIANTS,
    HiggsModelSpec,
    Profile,
    SeedGrid,
    _make_mlp,
    _make_seeded_model,
    default_raw_path,
    make_model,
    matched_mlp_width,
    mlp_parameter_count,
    run_profile,
    select_evaluations,
    spectral_parameter_count,
    summarize_evaluations,
    trainable_parameter_count,
    validate_raw,
)
from paper.higgs import NUM_FEATURES, HiggsLayout
from paper.models import KthEigval
from paper.tuning import select_learning_rates


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


@pytest.mark.parametrize(
    "spec",
    [
        HiggsModelSpec("linear"),
        HiggsModelSpec("mlp-1", 3),
        HiggsModelSpec("spectral", 3),
    ],
)
def test_seeded_model_construction_does_not_change_global_rng(spec):
    torch.manual_seed(17)
    state = torch.random.get_rng_state()

    first = _make_seeded_model(spec, init_seed=3)
    second = _make_seeded_model(spec, init_seed=3)

    assert torch.equal(torch.random.get_rng_state(), state)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_tiny_profile_runs_all_families_with_per_checkpoint_selection(complete_raw):
    raw = complete_raw
    selected = select_evaluations(raw)
    summary = summarize_evaluations(selected)

    assert tuple(raw.columns) == RESULT_SCHEMA.raw_columns
    assert set(raw["model"]) == set(VARIANTS)
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["protocol"]) == {"repeated_shuffle"}
    assert set(raw["optimizer"]) == {"adam"}
    assert set(raw["train_pool_size"]) == {16}

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert set(tuning["train_size"]) == {8, 16}
    assert tuning["val_logloss"].notna().all()
    assert tuning[["test_logloss", "test_brier"]].isna().all().all()
    assert set(evaluation["train_size"]) == {8, 16}
    assert evaluation["val_logloss"].isna().all()
    assert evaluation[["test_logloss", "test_brier"]].notna().all().all()
    winners = select_learning_rates(
        tuning,
        curve_columns=RESULT_SCHEMA.curve_columns,
        validation_metric=RESULT_SCHEMA.validation_metric,
    ).set_index(list(RESULT_SCHEMA.curve_columns))["selected_lr"]
    actual = evaluation.set_index(list(RESULT_SCHEMA.curve_columns))["lr"]
    assert np.allclose(actual, winners.loc[actual.index])
    identity = list(RESULT_SCHEMA.identity_columns)
    assert raw[identity].notna().all().all()
    assert not raw.duplicated(identity).any()
    assert len(tuning) == 20
    assert len(evaluation) == 10
    assert len(summary) == 10


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


def test_validate_raw_rejects_schema_identity_and_capacity_errors(complete_raw):
    with pytest.raises(ValueError, match="schema"):
        validate_raw(complete_raw.drop(columns="test_brier"), _tiny_profile())

    missing_identity = complete_raw.copy()
    missing_identity.loc[0, "width"] = np.nan
    with pytest.raises(ValueError, match="identity"):
        validate_raw(missing_identity, _tiny_profile())

    mixed_pool_sizes = complete_raw.copy()
    mixed_pool_sizes.loc[0, "train_pool_size"] = 8
    with pytest.raises(ValueError, match="one train_pool_size"):
        validate_raw(mixed_pool_sizes, _tiny_profile())

    duplicate = pd.concat((complete_raw, complete_raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw(duplicate, _tiny_profile())

    wrong_capacity = complete_raw.copy()
    index = wrong_capacity.index[wrong_capacity["model"] == "mlp-1"][0]
    wrong_capacity.loc[index, "width"] += 1
    with pytest.raises(ValueError, match="capacity metadata"):
        validate_raw(wrong_capacity, _tiny_profile())


def test_validate_raw_rejects_incomplete_grids_and_invalid_phase_metrics(
    complete_raw,
):
    missing_model = complete_raw.loc[complete_raw["model"] != "spectral"]
    with pytest.raises(ValueError, match="model/capacity grid"):
        validate_raw(missing_model, _tiny_profile())

    incomplete_tuning = complete_raw.drop(
        complete_raw.index[complete_raw["phase"] == "tuning"][0]
    )
    with pytest.raises(ValueError, match="tuning"):
        validate_raw(incomplete_tuning, _tiny_profile())

    incomplete_trajectory = complete_raw.drop(
        complete_raw.index[complete_raw["phase"] == "evaluation"][0]
    )
    with pytest.raises(ValueError, match="evaluation"):
        validate_raw(incomplete_trajectory, _tiny_profile())

    extra_seed = complete_raw.loc[
        complete_raw["phase"] == "evaluation"
    ].iloc[[0]].copy()
    extra_seed["init_seed"] = 99
    unexpected_evaluation_seed = pd.concat(
        (complete_raw, extra_seed), ignore_index=True
    )
    with pytest.raises(ValueError, match="evaluation seeds"):
        validate_raw(unexpected_evaluation_seed, _tiny_profile())

    leaked_validation = complete_raw.copy()
    index = leaked_validation.index[leaked_validation["phase"] == "evaluation"][0]
    leaked_validation.loc[index, "val_logloss"] = 0.5
    with pytest.raises(ValueError, match="validation metrics"):
        validate_raw(leaked_validation, _tiny_profile())


def test_validate_raw_warns_for_nonfinite_tuning_and_checks_selected_lr(
    complete_raw,
):
    tuning = complete_raw.loc[complete_raw["phase"] == "tuning"]
    row = tuning.iloc[0]
    curve = (
        (tuning["model"] == row["model"])
        & (tuning["dim"] == row["dim"])
        & (tuning["train_size"] == row["train_size"])
    )
    worst_lr = tuning.loc[curve].sort_values("val_logloss").iloc[-1]["lr"]
    nonfinite = complete_raw.copy()
    mask = (
        (nonfinite["phase"] == "tuning")
        & (nonfinite["model"] == row["model"])
        & (nonfinite["dim"] == row["dim"])
        & (nonfinite["train_size"] == row["train_size"])
        & (nonfinite["lr"] == worst_lr)
    )
    nonfinite.loc[mask, "val_logloss"] = np.nan
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        validate_raw(nonfinite, _tiny_profile())

    wrong_lr = complete_raw.copy()
    evaluation_row = wrong_lr.loc[wrong_lr["phase"] == "evaluation"].iloc[0]
    evaluation_curve = (
        (wrong_lr["phase"] == "evaluation")
        & (wrong_lr["model"] == evaluation_row["model"])
        & (wrong_lr["dim"] == evaluation_row["dim"])
        & (wrong_lr["train_size"] == evaluation_row["train_size"])
    )
    selected_lr = evaluation_row["lr"]
    alternate_lr = next(lr for lr in _tiny_profile().lrs if lr != selected_lr)
    wrong_lr.loc[evaluation_curve, "lr"] = alternate_lr
    with pytest.raises(ValueError, match="selected LR"):
        validate_raw(wrong_lr, _tiny_profile())


def test_validate_raw_rejects_checkpoint_without_finite_tuning(complete_raw):
    raw = complete_raw.copy()
    row = raw.loc[raw["phase"] == "tuning"].iloc[0]
    curve = (
        (raw["phase"] == "tuning")
        & (raw["model"] == row["model"])
        & (raw["dim"] == row["dim"])
        & (raw["train_size"] == row["train_size"])
    )
    raw.loc[curve, "val_logloss"] = np.nan

    with pytest.warns(RuntimeWarning, match="nonfinite"):
        with pytest.raises(ValueError, match="no finite validation"):
            validate_raw(raw, _tiny_profile())


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
