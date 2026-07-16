from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.experiments.higgs_scaling import (
    MODEL_COLUMNS,
    PROFILES,
    RAW_COLUMNS,
    VARIANTS,
    HiggsModelSpec,
    Profile,
    SeedGrid,
    _best_lrs,
    _make_mlp,
    _make_seeded_model,
    _model_specs,
    _selected_configs,
    _tuning_configs,
    make_model,
    matched_mlp_width,
    mlp_parameter_count,
    run_profile,
    spectral_parameter_count,
    summarize_raw,
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
        dims=(3,),
        lrs=(1e-2, 1e-1),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=8,
    )


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
    ("function", "args"),
    [
        (spectral_parameter_count, (0, 3)),
        (spectral_parameter_count, (28, 0)),
        (mlp_parameter_count, (0, 4, 2)),
        (mlp_parameter_count, (28, 0, 2)),
        (mlp_parameter_count, (28, 4, 0)),
        (matched_mlp_width, (0, 2, 100)),
        (matched_mlp_width, (28, 0, 100)),
        (matched_mlp_width, (28, 2, 0)),
    ],
)
def test_parameter_helpers_reject_nonpositive_inputs(function, args):
    with pytest.raises(ValueError, match="positive"):
        function(*args)


def test_parameter_helpers_reject_nonintegral_inputs():
    with pytest.raises(TypeError, match="integer"):
        matched_mlp_width(28.0, 2, 100)
    with pytest.raises(TypeError, match="integer"):
        matched_mlp_width(True, 2, 100)


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


def test_model_spec_validation():
    with pytest.raises(ValueError, match="linear model requires dim=0"):
        HiggsModelSpec("linear", 3)
    with pytest.raises(ValueError, match="positive dim"):
        HiggsModelSpec("mlp-1")
    with pytest.raises(ValueError, match="positive dim"):
        HiggsModelSpec("spectral")


def _tuning_row(*, lr: float, score: float, init_seed: int) -> dict[str, object]:
    return {
        "protocol": "one_pass",
        "optimizer": "adam",
        "phase": "tuning",
        "train_size": 100,
        "data_seed": 0,
        "model": "mlp-1",
        "dim": 3,
        "width": 6,
        "num_parameters": 181,
        "lr": lr,
        "init_seed": init_seed,
        "val_logloss": score,
    }


def test_lr_selection_filters_nonfinite_scores_and_breaks_ties_toward_lower_lr():
    tuning = pd.DataFrame(
        [
            _tuning_row(lr=1e-3, score=0.4, init_seed=0),
            _tuning_row(lr=1e-3, score=0.6, init_seed=1),
            _tuning_row(lr=1e-2, score=0.5, init_seed=0),
            _tuning_row(lr=1e-2, score=0.5, init_seed=1),
            _tuning_row(lr=1e-1, score=np.nan, init_seed=0),
            _tuning_row(lr=1e-1, score=np.inf, init_seed=1),
        ]
    )

    best = _best_lrs(tuning)

    assert best[["selected_lr", "median_val_logloss"]].to_dict("records") == [
        {"selected_lr": 1e-3, "median_val_logloss": 0.5}
    ]


def test_lr_selection_rejects_a_family_without_finite_validation():
    tuning = pd.DataFrame(
        [
            _tuning_row(lr=1e-2, score=np.nan, init_seed=0),
            _tuning_row(lr=1e-1, score=np.inf, init_seed=1),
        ]
    )

    with pytest.raises(ValueError, match="no finite validation"):
        _best_lrs(tuning)


@pytest.mark.parametrize(
    ("profile_name", "expected_specs", "expected_tuning", "expected_evaluation"),
    [
        ("sanity", 5, 5, 5),
        ("small", 9, 54, 18),
        ("full", 17, 204, 102),
    ],
)
def test_profiles_have_the_documented_trajectory_counts(
    profile_name, expected_specs, expected_tuning, expected_evaluation
):
    profile = PROFILES[profile_name]
    specs = _model_specs(profile, VARIANTS)

    assert len(specs) == expected_specs
    assert len(_tuning_configs(profile, specs)) == expected_tuning
    tuning = pd.DataFrame(
        [
            {
                **dict.fromkeys(MODEL_COLUMNS),
                "protocol": "one_pass",
                "optimizer": "adam",
                "model": spec.variant,
                "dim": spec.dim,
                "width": spec.width(NUM_FEATURES),
                "num_parameters": trainable_parameter_count(make_model(spec)),
                "lr": 1e-3,
                "val_logloss": 0.5,
            }
            for spec in specs
        ]
    )
    assert (
        len(_selected_configs(tuning, profile.evaluation_seeds))
        == expected_evaluation
    )


def test_tiny_profile_runs_all_families_with_one_coherent_selected_lr(complete_raw):
    raw = complete_raw
    summary = summarize_raw(raw)

    assert list(raw.columns) == RAW_COLUMNS
    assert set(raw["model"]) == set(VARIANTS)
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["protocol"]) == {"one_pass"}
    assert set(raw["optimizer"]) == {"adam"}

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert set(tuning["train_size"]) == {16}
    assert tuning["val_logloss"].notna().all()
    assert tuning[["test_logloss", "test_brier"]].isna().all().all()
    assert set(evaluation["train_size"]) == {8, 16}
    assert evaluation["val_logloss"].isna().all()
    assert evaluation[["test_logloss", "test_brier"]].notna().all().all()
    assert evaluation.groupby(MODEL_COLUMNS)["lr"].nunique().eq(1).all()
    identity = [
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
    assert raw[identity].notna().all().all()
    assert not raw.duplicated(identity).any()
    assert len(summary) == 10


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
    with pytest.raises(ValueError, match="evaluation trajectory"):
        validate_raw(incomplete_trajectory, _tiny_profile())

    leaked_validation = complete_raw.copy()
    index = leaked_validation.index[leaked_validation["phase"] == "evaluation"][0]
    leaked_validation.loc[index, "val_logloss"] = 0.5
    with pytest.raises(ValueError, match="validation metrics"):
        validate_raw(leaked_validation, _tiny_profile())


def test_validate_raw_warns_for_nonfinite_tuning_and_checks_selected_lr(
    complete_raw,
):
    tuning = complete_raw.loc[complete_raw["phase"] == "tuning"]
    model = tuning.iloc[0]["model"]
    model_tuning = tuning.loc[tuning["model"] == model]
    worst_lr = model_tuning.sort_values("val_logloss").iloc[-1]["lr"]
    nonfinite = complete_raw.copy()
    mask = (
        (nonfinite["phase"] == "tuning")
        & (nonfinite["model"] == model)
        & (nonfinite["lr"] == worst_lr)
    )
    nonfinite.loc[mask, "val_logloss"] = np.nan
    with pytest.warns(RuntimeWarning, match="nonfinite validation loss"):
        validate_raw(nonfinite, _tiny_profile())

    wrong_lr = complete_raw.copy()
    evaluation = wrong_lr.loc[
        (wrong_lr["phase"] == "evaluation") & (wrong_lr["model"] == model)
    ]
    selected_lr = evaluation["lr"].iloc[0]
    alternate_lr = next(lr for lr in _tiny_profile().lrs if lr != selected_lr)
    wrong_lr.loc[evaluation.index, "lr"] = alternate_lr
    with pytest.raises(ValueError, match="selected LR"):
        validate_raw(wrong_lr, _tiny_profile())


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
