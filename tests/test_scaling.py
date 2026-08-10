from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from paper.experiments.scaling import (
    SeedGrid,
    selected_runs,
    summarize_scaling,
)


EXPERIMENT_COLUMNS = ["protocol"]
CURVE_COLUMNS = [*EXPERIMENT_COLUMNS, "model", "dim", "train_size"]


def _tuning() -> pd.DataFrame:
    rows = []
    scores = {
        10: {0.01: (0.4, 0.4), 0.1: (0.3, 0.5)},
        20: {0.01: (np.nan, np.inf), 0.1: (0.2, 0.3)},
        30: {0.01: (0.1, 0.2), 0.1: (0.4, 0.5)},
    }
    for train_size, lr_scores in scores.items():
        for lr, values in lr_scores.items():
            for init_seed, score in enumerate(values):
                rows.append(
                    {
                        "protocol": "scaling-v1",
                        "model": "spectral",
                        "dim": 3,
                        "train_size": train_size,
                        "lr": lr,
                        "init_seed": init_seed,
                        "val_loss": score,
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    dim: int


def test_selected_runs_coalesces_checkpoints_with_the_same_lr():
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        runs = selected_runs(
            _tuning(),
            experiment_columns=EXPERIMENT_COLUMNS,
            curve_columns=CURVE_COLUMNS,
            validation_metric="val_loss",
            evaluation_seeds=SeedGrid(
                data_seeds=range(2, 4),
                init_seeds=range(5, 6),
            ),
            make_model_spec=ModelSpec,
        )

    observed = {
        (run.config.data_seed, run.config.lr): run.train_sizes for run in runs
    }
    assert observed == {
        (2, 0.01): (10, 30),
        (2, 0.1): (20,),
        (3, 0.01): (10, 30),
        (3, 0.1): (20,),
    }
    assert {run.config.init_seed for run in runs} == {5}


def test_selected_runs_rejects_mixed_experiments():
    tuning = pd.concat(
        [_tuning(), _tuning().assign(protocol="scaling-v2")],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly one experiment"):
        selected_runs(
            tuning,
            experiment_columns=EXPERIMENT_COLUMNS,
            curve_columns=CURVE_COLUMNS,
            validation_metric="val_loss",
            evaluation_seeds=SeedGrid(),
            make_model_spec=ModelSpec,
        )


def test_summarize_scaling_uses_only_selected_evaluation_rows():
    tuning = pd.DataFrame(
        [
            {
                "phase": "tuning",
                "protocol": "scaling-v1",
                "model": "linear",
                "dim": 0,
                "train_size": 10,
                "lr": lr,
                "val_loss": loss,
            }
            for lr, loss in ((0.01, 0.1), (0.1, 0.2))
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "phase": "evaluation",
                "protocol": "scaling-v1",
                "model": "linear",
                "dim": 0,
                "train_size": 10,
                "lr": lr,
                "test_loss": loss,
            }
            for lr, loss in (
                (0.01, 1.0),
                (0.01, 3.0),
                (0.1, 100.0),
            )
        ]
    )

    summary = summarize_scaling(
        pd.concat((tuning, evaluation), ignore_index=True),
        curve_columns=CURVE_COLUMNS,
        validation_metric="val_loss",
        quantile_metrics=("test_loss",),
    ).iloc[0]

    assert summary[
        [
            "selected_lr",
            "median_test_loss",
            "q25_test_loss",
            "q75_test_loss",
            "n",
        ]
    ].tolist() == [0.01, 2.0, 1.5, 2.5, 2]
