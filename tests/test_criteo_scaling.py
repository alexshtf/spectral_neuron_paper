from io import StringIO

import numpy as np
import pandas as pd
import pytest

from paper.experiments.criteo_scaling import (
    RESULT_SCHEMA,
    CriteoModelSpec,
    Profile,
    SeedGrid,
    build_arg_parser,
    default_raw_path,
    run_profile,
    select_evaluations,
    summarize_evaluations,
    validate_raw,
)
from criteo_test_data import write_tiny_criteo


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(16, 32),
        capacity_dims=(3,),
        lrs=(1e-3, 1e-2),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=16,
        min_count=2,
    )


@pytest.mark.parametrize(
    "spec",
    [
        CriteoModelSpec("linear-bucketed"),
        CriteoModelSpec("fm", 3),
        CriteoModelSpec("spectral-continuous", 3),
    ],
)
def test_model_spec_separates_capacity_from_persisted_dimension(spec):
    assert spec.result_dim == (spec.capacity_dim or 0)


@pytest.mark.parametrize(
    "variant, capacity_dim",
    [("linear-continuous", 3), ("fm", None), ("spectral-bucketed", 0)],
)
def test_model_spec_rejects_incoherent_capacity(variant, capacity_dim):
    with pytest.raises(ValueError, match="capacity_dim"):
        CriteoModelSpec(variant, capacity_dim)


@pytest.fixture(scope="module")
def complete_raw(tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-criteo-run")
    raw_path = directory / "train.txt"
    write_tiny_criteo(raw_path)
    return run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=directory / "cache",
    )


def test_cli_accepts_append_mode():
    args = build_arg_parser().parse_args(
        ["--data", "train.txt", "--write-mode", "append"]
    )

    assert args.write_mode == "append"


def test_default_path_is_protocol_specific():
    assert default_raw_path("full").name == (
        "criteo_scaling_full_repeated_shuffle.csv"
    )
    assert default_raw_path("full", "linear-continuous").name == (
        "criteo_scaling_full_repeated_shuffle_linear-continuous.csv"
    )


def test_profile_evaluates_only_its_requested_train_sizes(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path)
    profile = Profile(
        train_sizes=(16, 161),
        capacity_dims=(3,),
        lrs=(1e-3,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=16,
        min_count=2,
    )

    raw = run_profile(
        profile,
        raw_path=raw_path,
        cache_dir=cache_dir,
        variant="linear-bucketed",
    )

    assert raw["train_pool_size"].unique().item() == 80
    assert set(raw["train_size"]) == {16, 161}
    assert raw.groupby("phase")["train_size"].nunique().eq(2).all()


def test_tiny_profile_runs_end_to_end(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path)

    output = StringIO()
    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        chunk_size=23,
        progress=True,
        progress_file=output,
    )
    selected = select_evaluations(raw)
    summary = summarize_evaluations(selected)

    assert set(raw["model"]) == {
        "linear-bucketed",
        "linear-continuous",
        "fm",
        "spectral-bucketed",
        "spectral-continuous",
    }
    assert set(raw["protocol"]) == {"repeated_shuffle"}
    assert set(raw["optimizer"]) == {"adam+sparseadam"}
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["preprocessor_sample_size"]) == {8}
    assert set(raw["train_pool_size"]) == {80}
    assert tuple(raw.columns) == RESULT_SCHEMA.raw_columns
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert np.isfinite(tuning["val_logloss"]).all()
    assert evaluation["val_logloss"].isna().all()
    assert (
        raw.groupby(["phase", "data_seed", "model", "lr", "init_seed"])[
            "train_size"
        ]
        .nunique()
        .eq(2)
        .all()
    )
    assert raw["test_logloss"].notna().sum() == 10
    assert raw["test_brier"].notna().sum() == 10
    assert {"q25_test_brier", "q75_test_brier"} <= set(summary)
    assert len(summary) == 10
    printed = output.getvalue()
    assert "Encoding bucket train" in printed
    assert "Encoding bucket holdout" in printed
    assert "Encoding hybrid holdout" in printed
    assert "Tuning aggregate trajectory time: training=" in printed
    assert "validation=" in printed
    assert "Evaluation aggregate trajectory time: training=" in printed
    assert "test=" in printed


def test_variant_run_uses_only_its_preprocessor(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path)

    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        variant="linear-continuous",
    )

    assert set(raw["model"]) == {"linear-continuous"}
    assert len(list(cache_dir.glob("preprocessor-v*_*.pkl.zstd"))) == 1
    assert len(list((cache_dir / "encoded-v4").iterdir())) == 1


def test_validate_raw_accepts_complete_and_variant_sharded_results(complete_raw):
    validate_raw(complete_raw, _tiny_profile())
    linear = complete_raw.loc[complete_raw["model"] == "linear-bucketed"].copy()
    validate_raw(linear, _tiny_profile(), variant="linear-bucketed")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("protocol", "other", "protocol"),
        ("optimizer", "sgd", "optimizer"),
        ("preprocessor_sample_size", 7, "sample size"),
        ("preprocessor_seed", 1, "seed"),
    ],
)
def test_validate_raw_checks_experiment_metadata(
    complete_raw, column, value, message
):
    with pytest.raises(ValueError, match=message):
        validate_raw(complete_raw.assign(**{column: value}), _tiny_profile())


def test_validate_raw_rejects_identity_and_incomplete_grids(complete_raw):
    with pytest.raises(ValueError, match="schema"):
        validate_raw(complete_raw.drop(columns="test_brier"), _tiny_profile())

    duplicate = pd.concat((complete_raw, complete_raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw(duplicate, _tiny_profile())

    incomplete = complete_raw.drop(
        complete_raw.index[complete_raw["phase"] == "tuning"][0]
    )
    with pytest.raises(ValueError, match="tuning"):
        validate_raw(incomplete, _tiny_profile())


def test_validate_raw_checks_metrics_and_checkpoint_selected_lrs(complete_raw):
    nonfinite = complete_raw.copy()
    index = nonfinite.index[nonfinite["phase"] == "evaluation"][0]
    nonfinite.loc[index, "test_logloss"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_raw(nonfinite, _tiny_profile())

    wrong_lr = complete_raw.copy()
    row = wrong_lr.loc[wrong_lr["phase"] == "evaluation"].iloc[0]
    mask = (
        (wrong_lr["phase"] == "evaluation")
        & (wrong_lr["model"] == row["model"])
        & (wrong_lr["dim"] == row["dim"])
        & (wrong_lr["train_size"] == row["train_size"])
    )
    alternate_lr = next(lr for lr in _tiny_profile().lrs if lr != row["lr"])
    wrong_lr.loc[mask, "lr"] = alternate_lr
    with pytest.raises(ValueError, match="selected LR"):
        validate_raw(wrong_lr, _tiny_profile())


def test_parallel_profile_matches_serial_results(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path)

    serial = run_profile(
        _tiny_profile(), raw_path=raw_path, cache_dir=cache_dir, workers=1
    )
    parallel = run_profile(
        _tiny_profile(), raw_path=raw_path, cache_dir=cache_dir, workers=2
    )

    pd.testing.assert_frame_equal(serial, parallel)


def test_lr_selection_uses_median_tuning_validation_only():
    common = {
        "protocol": "repeated_shuffle",
        "optimizer": "adam+sparseadam",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_pool_size": 80,
        "train_size": 32,
        "model": "linear-bucketed",
        "dim": 0,
        "data_seed": 0,
    }
    tuning = [
        common
        | {
            "phase": "tuning",
            "lr": lr,
            "init_seed": seed,
            "val_logloss": score,
        }
        for lr, scores in ((0.01, (0.1, 10.0, 10.0)), (0.1, (1.0, 1.0, 100.0)))
        for seed, score in enumerate(scores)
    ]
    evaluation = [
        common
        | {
            "phase": "evaluation",
            "lr": lr,
            "init_seed": 3,
            "val_logloss": val,
            "test_logloss": test,
        }
        for lr, val, test in ((0.01, 0.1, 0.01), (0.1, 9.0, 4.0))
    ]

    selected = select_evaluations(pd.DataFrame(tuning + evaluation))

    assert selected[["lr", "median_val_logloss", "test_logloss"]].to_dict(
        "records"
    ) == [{"lr": 0.1, "median_val_logloss": 1.0, "test_logloss": 4.0}]


def test_lr_selection_keeps_training_pools_separate():
    rows = []
    for train_pool_size, selected_lr in ((40, 0.01), (80, 0.1)):
        common = {
            "protocol": "repeated_shuffle",
            "optimizer": "adam+sparseadam",
            "preprocessor_sample_size": 8,
            "preprocessor_seed": 0,
            "train_pool_size": train_pool_size,
            "train_size": 32,
            "model": "linear-bucketed",
            "dim": 0,
            "data_seed": 0,
            "init_seed": 0,
        }
        for lr in (0.01, 0.1):
            rows.append(
                common
                | {
                    "phase": "tuning",
                    "lr": lr,
                    "val_logloss": float(lr != selected_lr),
                }
            )
            rows.append(
                common
                | {
                    "phase": "evaluation",
                    "lr": lr,
                    "test_logloss": train_pool_size + lr,
                }
            )

    selected = select_evaluations(pd.DataFrame(rows))

    assert selected[["train_pool_size", "lr"]].to_dict("records") == [
        {"train_pool_size": 40, "lr": 0.01},
        {"train_pool_size": 80, "lr": 0.1},
    ]
