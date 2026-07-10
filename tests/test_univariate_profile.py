from io import StringIO

import pandas as pd
import torch

from paper.experiments.univariate import (
    DEFAULT_RUNS_DIR,
    RAW_COLUMNS,
    Profile,
    RunGrid,
    _default_raw_path,
    _make_seeded_model,
    run_profile,
)
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
        "eig_idx",
        "budget",
        "selected_lr",
        "median_test_rmse",
        "q25_test_rmse",
        "q75_test_rmse",
        "mean_test_rmse",
        "n",
    }.issubset(summary.columns)


def test_run_grid_has_known_length():
    profile = Profile(
        complexities=(3, 5),
        target_seeds=range(2),
        init_seeds=range(3),
        dims=(3, 5),
        lrs=(1e-2, 1e-3),
        budgets=(1,),
        batch_size=4,
        noise_stds=(0.0, 0.1),
    )

    grid = RunGrid(profile)
    configs = list(grid)

    assert len(grid) == (
        len(profile.dims)
        * len(profile.complexities)
        * len(profile.target_seeds)
        * len(EXPECTED_FIT_PAIRS)
        * len(profile.noise_stds)
        * len(profile.lrs)
        * len(profile.init_seeds)
    )
    assert len(configs) == len(grid)
    assert {
        (config.target_spec.kind, config.model_spec.name)
        for config in configs
    } == EXPECTED_FIT_PAIRS


def test_default_output_path_uses_notebook_runs_dir():
    assert _default_raw_path("sanity") == DEFAULT_RUNS_DIR / "univariate_sanity.csv"


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

    result_columns = [col for col in RAW_COLUMNS if col != "elapsed_seconds"]
    pd.testing.assert_frame_equal(
        serial[result_columns],
        parallel[result_columns],
    )


def test_make_seeded_model_preserves_torch_rng_state():
    spec = ModelSpec("unconstrained", "unconstrained", dim=3)
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    _make_seeded_model(spec, input_dim=1, init_seed=999)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
