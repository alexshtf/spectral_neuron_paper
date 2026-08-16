from io import StringIO

import pandas as pd
import pytest

from paper.experiments.criteo_scaling import (
    CriteoModelSpec,
    Profile,
    SeedGrid,
    build_arg_parser,
    default_raw_path,
    run_profile,
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
