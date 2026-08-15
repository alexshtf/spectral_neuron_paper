import numpy as np
import pandas as pd
import pytest

from paper.tuning import select_learning_rates, select_rows_at_learning_rates


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


def test_learning_rate_is_rejected_if_any_validation_trial_is_nonfinite():
    tuning = pd.DataFrame(
        {
            "curve": ["a"] * 4,
            "lr": [0.01, 0.01, 0.1, 0.1],
            "validation": [0.1, np.nan, 1.0, 1.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="nonfinite"):
        selected = select_learning_rates(
            tuning,
            curve_columns=["curve"],
            validation_metric="validation",
        )

    assert selected["selected_lr"].tolist() == [0.1]
