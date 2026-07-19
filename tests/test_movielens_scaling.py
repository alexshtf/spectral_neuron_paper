from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.experiments.movielens_scaling import (
    CURVE_COLUMNS,
    PROFILES,
    RAW_COLUMNS,
    MovieLensModelSpec,
    Profile,
    SeedGrid,
    _make_seeded_model,
    default_raw_path,
    make_model,
    run_profile,
    summarize_raw,
    validate_raw,
)
from paper.models import FactorizationMachine


def _write_ratings(path: Path, users: int = 6, movies: int = 10) -> None:
    rows = [
        {
            "userId": user + 1,
            "movieId": 10 * (movie + 1),
            "rating": 0.5 + ((3 * user + movie) % 10) / 2,
            "timestamp": 1_000_000 + 100 * user + movie,
        }
        for user in range(users)
        for movie in range(movies)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(8, 16, 56),
        dims=(3,),
        lrs=(1e-2, 1e-1),
        tuning_seeds=SeedGrid(init_seeds=range(2)),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=8,
    )


@pytest.fixture(scope="module")
def complete_raw(tmp_path_factory):
    directory = tmp_path_factory.mktemp("movielens-scaling")
    ratings = directory / "ratings.csv"
    _write_ratings(ratings)
    return run_profile(
        _tiny_profile(),
        raw_path=ratings,
        cache_dir=directory / "cache",
        chunk_size=13,
    )


def test_two_field_fm_is_biased_matrix_factorization():
    model = FactorizationMachine(num_features=4, num_fields=2, rank=2)
    with torch.no_grad():
        model.bias.fill_(0.25)
        model.weight.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model.embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    ids = torch.tensor([[0, 2], [1, 3]])

    expected = torch.tensor(
        [
            0.25 + 1.0 + 3.0 + 1.0 * 5.0 + 2.0 * 6.0,
            0.25 + 2.0 + 4.0 + 3.0 * 7.0 + 4.0 * 8.0,
        ]
    )

    torch.testing.assert_close(model(ids), expected)


def test_parameter_matching_is_per_identity():
    fm_spec = MovieLensModelSpec("fm", 3)
    spectral_spec = MovieLensModelSpec("spectral", 3)
    num_features = 17

    fm = make_model(fm_spec, num_features)
    spectral = make_model(spectral_spec, num_features)

    assert fm_spec.rank == 5
    assert fm_spec.parameters_per_identity == spectral_spec.parameters_per_identity == 6
    assert sum(p.numel() for p in spectral.parameters()) - sum(
        p.numel() for p in fm.parameters()
    ) == 5


def test_seeded_construction_preserves_global_rng():
    torch.manual_seed(17)
    state = torch.random.get_rng_state()

    first = _make_seeded_model(
        MovieLensModelSpec("spectral", 3), num_features=17, init_seed=5
    )
    second = _make_seeded_model(
        MovieLensModelSpec("spectral", 3), num_features=17, init_seed=5
    )

    assert torch.equal(torch.random.get_rng_state(), state)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_tiny_profile_runs_end_to_end(complete_raw):
    profile = _tiny_profile()
    raw = complete_raw
    validate_raw(raw, profile)
    summary = summarize_raw(raw)

    assert list(raw.columns) == RAW_COLUMNS
    assert set(raw["model"]) == {"linear", "fm", "spectral"}
    assert set(raw["protocol"]) == {"repeated_shuffle"}
    assert set(raw["optimizer"]) == {"adam+sparseadam"}
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["train_pool_size"]) == {48}

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert set(tuning["train_size"]) == {8, 16, 56}
    assert tuning["val_rmse"].notna().all()
    assert tuning["val_warm_fraction"].between(0, 1).all()
    assert (tuning.loc[tuning["train_size"] < 48, "val_warm_fraction"] < 1).any()
    assert tuning.loc[tuning["train_size"] >= 48, "val_warm_fraction"].eq(1).all()
    assert tuning["test_rmse"].isna().all()
    assert tuning["test_warm_fraction"].isna().all()
    assert evaluation["val_rmse"].isna().all()
    assert evaluation["val_warm_fraction"].isna().all()
    assert np.isfinite(evaluation["test_rmse"]).all()
    assert evaluation["test_warm_fraction"].between(0, 1).all()
    assert (
        evaluation.loc[evaluation["train_size"] < 48, "test_warm_fraction"] < 1
    ).any()
    assert evaluation.loc[
        evaluation["train_size"] >= 48, "test_warm_fraction"
    ].eq(1).all()

    expected = (
        tuning.groupby(CURVE_COLUMNS + ["lr"], as_index=False)["val_rmse"]
        .median()
        .sort_values(CURVE_COLUMNS + ["val_rmse", "lr"], kind="mergesort")
        .groupby(CURVE_COLUMNS, as_index=False, sort=False)
        .head(1)
        .rename(columns={"lr": "expected_lr"})
    )
    observed = evaluation.merge(
        expected[CURVE_COLUMNS + ["expected_lr"]],
        on=CURVE_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    np.testing.assert_allclose(observed["lr"], observed["expected_lr"])

    assert set(summary["train_size"]) == {8, 16, 56}
    assert len(summary) == 9

    combined = pd.concat(
        (raw, raw.assign(train_pool_size=raw["train_pool_size"] + 8)),
        ignore_index=True,
    )
    assert len(summarize_raw(combined)) == 2 * len(summary)




def test_default_path_isolated_from_legacy_one_pass_results():
    assert (
        default_raw_path("full").name
        == "movielens_scaling_full_repeated_shuffle.csv"
    )
    assert (
        default_raw_path("full", "fm").name
        == "movielens_scaling_full_repeated_shuffle_fm.csv"
    )


def test_validate_raw_accepts_complete_and_variant_sharded_results(complete_raw):
    validate_raw(complete_raw, _tiny_profile())
    linear = complete_raw.loc[complete_raw["model"] == "linear"].copy()
    validate_raw(linear, _tiny_profile(), variant="linear")


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("protocol", "legacy", "protocol"),
        ("optimizer", "sgd", "optimizer"),
        ("split_seed", 1, "split_seed"),
    ],
)
def test_validate_raw_rejects_wrong_experiment_identity(
    complete_raw, column, value, match
):
    invalid = complete_raw.assign(**{column: value})

    with pytest.raises(ValueError, match=match):
        validate_raw(invalid, _tiny_profile())


def test_validate_raw_rejects_schema_pool_identity_and_capacity_errors(complete_raw):
    with pytest.raises(ValueError, match="schema"):
        validate_raw(complete_raw.drop(columns="test_rmse"), _tiny_profile())

    fractional_pool = complete_raw.assign(train_pool_size=48.5)
    with pytest.raises(ValueError, match="positive integer train_pool_size"):
        validate_raw(fractional_pool, _tiny_profile())

    missing_identity = complete_raw.copy()
    missing_identity.loc[0, "rank"] = np.nan
    with pytest.raises(ValueError, match="identity"):
        validate_raw(missing_identity, _tiny_profile())

    duplicate = pd.concat((complete_raw, complete_raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw(duplicate, _tiny_profile())

    wrong_capacity = complete_raw.copy()
    wrong_capacity.loc[wrong_capacity["model"] == "fm", "rank"] += 1
    with pytest.raises(ValueError, match="capacity metadata"):
        validate_raw(wrong_capacity, _tiny_profile())


def test_validate_raw_rejects_incomplete_grids_and_wrong_selected_lr(complete_raw):
    one_phase = complete_raw.loc[complete_raw["phase"] == "tuning"]
    with pytest.raises(ValueError, match="phases"):
        validate_raw(one_phase, _tiny_profile())

    tuning_index = complete_raw.index[complete_raw["phase"] == "tuning"][0]
    with pytest.raises(ValueError, match="tuning"):
        validate_raw(complete_raw.drop(tuning_index), _tiny_profile())

    evaluation_index = complete_raw.index[complete_raw["phase"] == "evaluation"][0]
    with pytest.raises(ValueError, match="evaluation"):
        validate_raw(complete_raw.drop(evaluation_index), _tiny_profile())

    wrong_lr = complete_raw.copy()
    index = wrong_lr.index[wrong_lr["phase"] == "evaluation"][0]
    selected_lr = wrong_lr.loc[index, "lr"]
    alternate_lr = next(lr for lr in _tiny_profile().lrs if lr != selected_lr)
    wrong_lr.loc[index, "lr"] = alternate_lr
    with pytest.raises(ValueError, match="selected LR"):
        validate_raw(wrong_lr, _tiny_profile())


def test_validate_raw_rejects_nonfinite_and_out_of_range_metrics(complete_raw):
    nonfinite = complete_raw.copy()
    index = nonfinite.index[nonfinite["phase"] == "tuning"][0]
    nonfinite.loc[index, "val_rmse"] = np.nan
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        validate_raw(nonfinite, _tiny_profile())

    no_finite_checkpoint = complete_raw.copy()
    row = no_finite_checkpoint.loc[index]
    mask = (
        no_finite_checkpoint["phase"].eq("tuning")
        & no_finite_checkpoint["model"].eq(row["model"])
        & no_finite_checkpoint["dim"].eq(row["dim"])
        & no_finite_checkpoint["train_size"].eq(row["train_size"])
    )
    no_finite_checkpoint.loc[mask, "val_rmse"] = np.nan
    with (
        pytest.warns(RuntimeWarning, match="nonfinite"),
        pytest.raises(ValueError, match="no finite validation"),
    ):
        validate_raw(no_finite_checkpoint, _tiny_profile())

    invalid_warm_fraction = complete_raw.copy()
    index = invalid_warm_fraction.index[
        invalid_warm_fraction["phase"] == "evaluation"
    ][0]
    invalid_warm_fraction.loc[index, "test_warm_fraction"] = 1.1
    with pytest.raises(ValueError, match="out of range"):
        validate_raw(invalid_warm_fraction, _tiny_profile())


def test_parallel_profile_matches_serial_results(tmp_path):
    ratings = tmp_path / "ratings.csv"
    cache_dir = tmp_path / "cache"
    _write_ratings(ratings)

    serial = run_profile(
        _tiny_profile(), raw_path=ratings, cache_dir=cache_dir, workers=1
    )
    parallel = run_profile(
        _tiny_profile(), raw_path=ratings, cache_dir=cache_dir, workers=2
    )

    pd.testing.assert_frame_equal(serial, parallel)
