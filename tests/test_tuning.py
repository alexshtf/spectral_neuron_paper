import numpy as np
import pandas as pd
import pytest

from paper.experiments.synthetic import (
    select_checkpoints,
    select_evaluations,
    summarize_results,
)
from paper.tuning import select_learning_rates, select_rows_at_learning_rates


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


def test_learning_rate_selection_uses_median_validation_score_and_lower_ties():
    tuning = pd.DataFrame(
        {
            "curve": ["a"] * 4,
            "lr": [0.1, 0.1, 0.01, 0.01],
            "validation": [0.0, 2.0, 1.0, 1.0],
        }
    )

    selected = select_learning_rates(
        tuning,
        curve_columns=["curve"],
        validation_metric="validation",
    )

    assert selected.to_dict("records") == [
        {"curve": "a", "selected_lr": 0.01, "median_validation": 1.0}
    ]


def test_rows_at_learning_rates_keeps_only_the_selected_candidate():
    rows = pd.DataFrame(
        {
            "curve": ["a", "a", "b"],
            "lr": [0.01, 0.1, 0.1],
            "value": [1, 2, 3],
        }
    )
    learning_rates = pd.DataFrame(
        {"curve": ["a", "b"], "selected_lr": [0.1, 0.1]}
    )

    selected = select_rows_at_learning_rates(
        rows,
        learning_rates,
        curve_columns=["curve"],
    )

    assert selected[["curve", "value"]].to_dict("records") == [
        {"curve": "a", "value": 2},
        {"curve": "b", "value": 3},
    ]


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


def test_checkpoint_selection_orders_runs_and_keeps_nan_ties_stable():
    raw = pd.DataFrame(
        [
            _row(
                step=2,
                train_size=8,
                target_seed=1,
                val_rmse=np.nan,
                test_rmse=20.0,
            ),
            _row(
                step=1,
                train_size=4,
                target_seed=0,
                val_rmse=0.5,
                test_rmse=10.0,
            ),
            _row(
                step=1,
                train_size=4,
                target_seed=1,
                val_rmse=np.nan,
                test_rmse=11.0,
            ),
            _row(
                step=1,
                train_size=4,
                target_seed=0,
                val_rmse=0.5,
                test_rmse=12.0,
            ),
        ]
    )

    selected = select_checkpoints(raw, [8, 4])

    assert selected[["train_size", "target_seed", "step", "test_rmse"]].to_dict(
        "records"
    ) == [
        {"train_size": 8, "target_seed": 0, "step": 1, "test_rmse": 10.0},
        {"train_size": 8, "target_seed": 1, "step": 1, "test_rmse": 11.0},
        {"train_size": 4, "target_seed": 0, "step": 1, "test_rmse": 10.0},
        {"train_size": 4, "target_seed": 1, "step": 1, "test_rmse": 11.0},
    ]


def test_lr_selection_uses_validation_not_test():
    best = pd.DataFrame(
        [
            _row(step=1, train_size=4, lr=0.01, val_rmse=2.0, test_rmse=0.1),
            _row(step=1, train_size=4, lr=0.1, val_rmse=1.0, test_rmse=10.0),
        ]
    )

    selected = select_evaluations(best)

    assert selected["selected_lr"].unique().tolist() == [0.1]


def test_learning_rate_selection_breaks_validation_ties_toward_lower_rate():
    checkpoints = pd.DataFrame(
        [
            _row(
                step=1,
                train_size=4,
                lr=lr,
                val_rmse=1.0,
                test_rmse=0.0,
            )
            for lr in (0.1, 0.01)
        ]
    )

    selected = select_evaluations(checkpoints)

    assert selected["selected_lr"].unique().tolist() == [0.01]


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
    selected = select_evaluations(best)

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
        selected = select_evaluations(best)

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
