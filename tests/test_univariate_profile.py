import torch

from paper.experiments.univariate import (
    RAW_COLUMNS,
    Profile,
    _make_seeded_model,
    run_profile,
)
from paper.models import ModelSpec
from paper.tuning import summarize_raw


def test_tiny_profile_produces_raw_logs_and_summary():
    profile = Profile(
        complexities=(3,),
        target_seeds=range(1),
        init_seeds=range(1),
        dims=(3,),
        lrs=(1e-2,),
        budgets=(1, 2),
        batch_size=4,
    )

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


def test_make_seeded_model_preserves_torch_rng_state():
    spec = ModelSpec("spectral", "spectral", dim=3)
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    _make_seeded_model(spec, input_dim=1, init_seed=999)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
