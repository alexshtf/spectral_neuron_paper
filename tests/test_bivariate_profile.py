import numpy as np
import pytest

from paper.experiments.bivariate import (
    _default_raw_path,
    build_arg_parser,
    run_profile,
)
from paper.experiments.synthetic import DEFAULT_RUNS_DIR, RAW_COLUMNS, Profile
from paper.targets import TargetSpec, make_bivariate_target
from paper.tasks import make_bivariate_task
from paper.tuning import summarize_raw


def tiny_profile() -> Profile:
    return Profile(
        complexities=(5,),
        target_seeds=range(1),
        init_seeds=range(1),
        dims=(3,),
        lrs=(1e-2,),
        budgets=(1, 2),
        batch_size=4,
    )


def test_bivariate_task_uses_a_square_tensor_product_test_grid():
    target = make_bivariate_target(TargetSpec(kind="general", complexity=5, seed=1))
    task = make_bivariate_task(
        target,
        lower=-1.0,
        upper=1.0,
        batch_size=4,
        val_size=3,
        test_size=9,
        seed=2,
    )

    grid = np.linspace(-1.0, 1.0, 3)
    expected = np.array([(x1, x2) for x1 in grid for x2 in grid])
    assert task.input_dim == 2
    assert task.x_val.shape == (3, 2)
    np.testing.assert_allclose(task.x_test.numpy(), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        task.y_test.numpy(), target(expected), rtol=1e-6, atol=1e-6
    )


def test_bivariate_task_requires_a_perfect_square_test_size():
    with pytest.raises(ValueError, match="perfect square"):
        make_bivariate_task(
            lambda x: x[..., 0],
            lower=-1.0,
            upper=1.0,
            batch_size=4,
            val_size=3,
            test_size=8,
            seed=2,
        )


def test_tiny_bivariate_profile_produces_raw_logs_and_summary():
    profile = tiny_profile()

    raw = run_profile(profile, val_size=16, test_size=16)
    summary = summarize_raw(raw, profile.budgets)

    assert not raw.empty
    assert not summary.empty
    assert set(RAW_COLUMNS).issubset(raw.columns)


def test_bivariate_cli_matches_the_shared_arguments():
    parser = build_arg_parser()

    assert parser.parse_args([]).write_mode == "overwrite"
    assert parser.parse_args(["--profile", "small"]).profile == "small"
    assert _default_raw_path("sanity") == DEFAULT_RUNS_DIR / "bivariate_sanity.csv"
