from io import StringIO
from typing import TextIO

import pandas as pd
import pytest
import torch

from paper.experiments.bivariate import run_profile as run_bivariate
from paper.experiments.synthetic import (
    Profile,
    RunGrid,
    _make_seeded_model,
    build_arg_parser,
    select_checkpoints,
    select_evaluations,
    summarize_results,
    validate_raw,
)
from paper.experiments.univariate import run_profile as run_univariate
from paper.models import ModelSpec


_EXPECTED_FIT_PAIRS = {
    ("general", "unconstrained"),
    ("monotone", "unconstrained"),
    ("monotone", "monotone"),
}


def _tiny_profile() -> Profile:
    return Profile(
        complexities=(3,),
        target_seeds=range(1),
        init_seeds=range(1),
        dims=(3,),
        lrs=(1e-2,),
        train_sizes=(4, 8),
        batch_size=4,
    )


def _run_tiny_profile(
    *,
    runner=run_univariate,
    workers: int = 1,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> pd.DataFrame:
    return runner(
        _tiny_profile(),
        val_size=16,
        test_size=16,
        workers=workers,
        progress=progress,
        progress_file=progress_file,
    )


def _row(
    *,
    step: int,
    train_size: int,
    val_rmse: float,
    test_rmse: float,
    lr: float = 0.01,
    target_seed: int = 0,
    init_seed: int = 0,
):
    return {
        "target_kind": "monotone",
        "complexity": 5,
        "target_seed": target_seed,
        "noise_std": 0.0,
        "model": "unconstrained",
        "dim": 5,
        "lr": lr,
        "init_seed": init_seed,
        "batch_size": 4,
        "step": step,
        "train_size": train_size,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
    }


@pytest.mark.parametrize("runner", [run_univariate, run_bivariate])
def test_tiny_profile_produces_valid_raw_logs_and_summary(runner):
    raw = _run_tiny_profile(runner=runner)

    validate_raw(raw, _tiny_profile())

    assert not summarize_results(raw).empty


def test_validation_rejects_an_incomplete_profile():
    raw = _run_tiny_profile()

    with pytest.raises(ValueError, match="checkpoint grids"):
        validate_raw(raw.iloc[:-1], _tiny_profile())


def test_run_grid_declares_the_scientific_fit_pairs():
    grid = RunGrid(_tiny_profile())
    configs = list(grid)

    assert {
        (config.target_spec.kind, config.model_spec.kind) for config in configs
    } == _EXPECTED_FIT_PAIRS
    assert len(configs) == len(grid) == len(_EXPECTED_FIT_PAIRS)


def test_cli_uses_the_requested_write_mode():
    parser = build_arg_parser({"tiny": _tiny_profile()})

    assert parser.parse_args(["--write-mode", "append"]).write_mode == "append"


def test_profile_reports_progress():
    progress = StringIO()

    _run_tiny_profile(progress=True, progress_file=progress)

    output = progress.getvalue()
    assert "3/3" in output
    assert "experiment" in output


def test_parallel_profile_matches_serial_results():
    serial = _run_tiny_profile()
    parallel = _run_tiny_profile(workers=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_make_seeded_model_preserves_torch_rng_state():
    spec = ModelSpec("unconstrained", dim=3)
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    _make_seeded_model(spec, input_dim=1, init_seed=999)

    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_checkpoint_selection_uses_validation_not_test():
    raw = pd.DataFrame(
        [
            _row(step=1, train_size=4, val_rmse=0.5, test_rmse=10.0),
            _row(step=2, train_size=8, val_rmse=1.0, test_rmse=0.1),
        ]
    )

    selected = select_checkpoints(raw, [8])

    assert selected["step"].tolist() == [1]


def test_checkpoint_selection_is_per_run_and_keeps_the_earliest_tie():
    raw = pd.DataFrame(
        [
            _row(
                step=step,
                train_size=4 * step,
                target_seed=seed,
                val_rmse=score,
                test_rmse=0.0,
            )
            for seed, scores in enumerate(((0.5, 0.5), (0.6, 0.4)))
            for step, score in enumerate(scores, start=1)
        ]
    ).sample(frac=1, random_state=0)

    selected = select_checkpoints(raw, [4, 8]).sort_values(
        ["train_size", "target_seed"]
    )

    assert selected["step"].tolist() == [1, 1, 1, 2]


def test_learning_rate_selection_uses_validation_not_test():
    checkpoints = pd.DataFrame(
        [
            _row(step=1, train_size=4, lr=0.01, val_rmse=2.0, test_rmse=0.1),
            _row(step=1, train_size=4, lr=0.1, val_rmse=1.0, test_rmse=10.0),
        ]
    )

    selected = select_evaluations(checkpoints)

    assert selected["selected_lr"].unique().tolist() == [0.1]


def test_summary_uses_the_checkpoint_grid_in_the_raw_results():
    raw = pd.DataFrame(
        [
            _row(step=1, train_size=4, val_rmse=0.5, test_rmse=0.5),
            _row(step=2, train_size=8, val_rmse=0.4, test_rmse=0.4),
        ]
    )

    summary = summarize_results(raw)

    assert summary["train_size"].tolist() == [4, 8]


def test_summary_orders_each_curve_by_training_budget():
    raw = pd.DataFrame(
        [
            _row(
                step=step,
                train_size=4 * step,
                val_rmse=1 / step,
                test_rmse=1 / step,
            )
            | {"model": model}
            for model in ("unconstrained", "monotone")
            for step in (1, 2)
        ]
    ).sample(frac=1, random_state=0)

    summary = summarize_results(raw)

    assert list(zip(summary["model"], summary["train_size"])) == [
        ("monotone", 4),
        ("monotone", 8),
        ("unconstrained", 4),
        ("unconstrained", 8),
    ]


def test_summary_rejects_inconsistent_checkpoint_grids():
    raw = pd.DataFrame(
        [
            _row(step=1, train_size=4, val_rmse=0.5, test_rmse=0.5),
            _row(step=2, train_size=8, val_rmse=0.4, test_rmse=0.4),
            _row(
                step=1,
                train_size=4,
                target_seed=1,
                val_rmse=0.5,
                test_rmse=0.5,
            ),
        ]
    )

    with pytest.raises(ValueError, match="inconsistent train_size checkpoints"):
        summarize_results(raw)
