from io import StringIO

import pandas as pd
import pytest
import torch

from paper.experiments.synthetic import (
    RAW_COLUMNS,
    Profile,
    RunGrid,
    _make_seeded_model,
    build_arg_parser,
    write_csv,
)
from paper.experiments.univariate import run_profile
from paper.models import ModelSpec
from paper.tuning import summarize_raw


EXPECTED_FIT_PAIRS = {
    ("general", "unconstrained"),
    ("monotone", "unconstrained"),
    ("monotone", "monotone"),
}


def tiny_profile() -> Profile:
    return Profile(
        complexities=(3,),
        target_seeds=range(1),
        init_seeds=range(1),
        dims=(3,),
        lrs=(1e-2,),
        budgets=(1, 2),
        batch_size=4,
    )


def test_tiny_profile_produces_raw_logs_and_summary():
    profile = tiny_profile()

    raw = run_profile(profile, val_size=16, test_size=16)
    summary = summarize_raw(raw, profile.budgets)

    assert not raw.empty
    assert not summary.empty
    assert set(RAW_COLUMNS).issubset(raw.columns)
    assert {
        "target_kind",
        "complexity",
        "noise_std",
        "model",
        "dim",
        "budget",
        "selected_lr",
        "median_test_rmse",
        "q25_test_rmse",
        "q75_test_rmse",
        "n",
    }.issubset(summary.columns)


def test_run_grid_declares_the_scientific_fit_pairs():
    grid = RunGrid(tiny_profile())
    configs = list(grid)

    assert {
        (config.target_spec.kind, config.model_spec.kind) for config in configs
    } == EXPECTED_FIT_PAIRS
    assert len(configs) == len(grid) == len(EXPECTED_FIT_PAIRS)


def test_write_mode_appends_or_replaces_results(tmp_path):
    parser = build_arg_parser({"tiny": tiny_profile()})
    assert parser.parse_args(["--write-mode", "append"]).write_mode == "append"

    path = tmp_path / "results.csv"
    write_csv(pd.DataFrame({"value": [1]}), path, write_mode="append")
    write_csv(pd.DataFrame({"value": [2]}), path, write_mode="append")
    assert pd.read_csv(path)["value"].tolist() == [1, 2]

    write_csv(pd.DataFrame({"value": [3]}), path)
    assert pd.read_csv(path)["value"].tolist() == [3]


def test_append_rejects_an_incompatible_csv_schema(tmp_path):
    path = tmp_path / "results.csv"
    original = "value,metric\n1,2\n"
    path.write_text(original)

    with pytest.raises(ValueError, match="CSV header"):
        write_csv(
            pd.DataFrame({"metric": [3], "value": [4]}), path, write_mode="append"
        )

    assert path.read_text() == original


def test_run_profile_reports_progress():
    progress = StringIO()

    run_profile(
        tiny_profile(),
        val_size=8,
        test_size=8,
        progress=True,
        progress_file=progress,
    )

    text = progress.getvalue()
    assert "3/3" in text
    assert "experiment" in text


def test_parallel_profile_matches_serial_results():
    profile = tiny_profile()

    serial = run_profile(profile, val_size=8, test_size=8, workers=1)
    parallel = run_profile(profile, val_size=8, test_size=8, workers=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_make_seeded_model_preserves_torch_rng_state():
    spec = ModelSpec("unconstrained", dim=3)
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    _make_seeded_model(spec, input_dim=1, init_seed=999)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
