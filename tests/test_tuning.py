import numpy as np
import pandas as pd
import pytest

from paper.tuning import best_checkpoints, select_lr, summarize_raw


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


def test_checkpoint_selection_uses_validation_not_test():
    raw = pd.DataFrame(
        [
            _row(step=1, train_size=4, val_rmse=0.5, test_rmse=10.0),
            _row(step=2, train_size=8, val_rmse=1.0, test_rmse=0.1),
        ]
    )

    selected = best_checkpoints(raw, [8])

    assert selected["step"].tolist() == [1]


def test_lr_selection_uses_validation_not_test():
    best = pd.DataFrame(
        [
            _row(step=1, train_size=4, lr=0.01, val_rmse=2.0, test_rmse=0.1),
            _row(step=1, train_size=4, lr=0.1, val_rmse=1.0, test_rmse=10.0),
        ]
    )

    selected = select_lr(best)

    assert selected["selected_lr"].unique().tolist() == [0.1]


def test_seed_aggregation_is_median_not_minimum():
    rows = []
    for seed, val_rmse in enumerate([0.1, 10.0, 10.0]):
        rows.append(
            _row(
                step=1,
                train_size=4,
                lr=0.01,
                target_seed=seed,
                val_rmse=val_rmse,
                test_rmse=val_rmse,
            )
        )
    for seed, val_rmse in enumerate([1.0, 1.0, 100.0]):
        rows.append(
            _row(
                step=1,
                train_size=4,
                lr=0.1,
                target_seed=seed,
                val_rmse=val_rmse,
                test_rmse=val_rmse,
            )
        )

    best = pd.DataFrame(rows)
    selected = select_lr(best)

    assert selected["selected_lr"].unique().tolist() == [0.1]


def test_lr_is_rejected_if_any_validation_trial_is_nonfinite():
    best = pd.DataFrame(
        [
            _row(
                step=1,
                train_size=4,
                lr=lr,
                target_seed=seed,
                val_rmse=val_rmse,
                test_rmse=0.0,
            )
            for lr, scores in ((0.01, (0.1, np.nan)), (0.1, (1.0, 1.0)))
            for seed, val_rmse in enumerate(scores)
        ]
    )

    with pytest.warns(RuntimeWarning, match="nonfinite"):
        selected = select_lr(best)

    assert selected["selected_lr"].unique().tolist() == [0.1]


def test_summary_uses_the_checkpoint_grid_in_the_raw_results():
    raw = pd.DataFrame(
        [
            _row(step=1, train_size=4, val_rmse=0.5, test_rmse=0.5),
            _row(step=2, train_size=8, val_rmse=0.4, test_rmse=0.4),
        ]
    )

    summary = summarize_raw(raw)

    assert summary["train_size"].tolist() == [4, 8]


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
        summarize_raw(raw)
