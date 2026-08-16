from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from paper.experiments.scaling import (
    ScalingSchema,
    SeedGrid,
    select_evaluation_runs,
    select_evaluations,
    summarize_evaluations,
    validate_results,
)


RESULT_SCHEMA = ScalingSchema(
    experiment_columns=("protocol",),
    model_columns=("model", "dim"),
    model_spec_columns=("model", "dim"),
    validation_metric="val_loss",
    test_metrics=("test_loss",),
)

TRAIN_SIZES = (10, 20)
LEARNING_RATES = (0.01, 0.1)
TUNING_SEEDS = SeedGrid(init_seeds=range(2))
EVALUATION_SEEDS = SeedGrid(data_seeds=range(1, 2), init_seeds=range(2, 4))
EXPECTED_MODELS = (
    {"model": "linear", "dim": 0},
    {"model": "spectral", "dim": 3},
)


@pytest.fixture
def complete_raw() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(EXPECTED_MODELS):
        for train_size in TRAIN_SIZES:
            for lr in LEARNING_RATES:
                for data_seed, init_seed in TUNING_SEEDS:
                    rows.append(
                        {
                            "protocol": "scaling-v1",
                            "phase": "tuning",
                            "train_size": train_size,
                            "data_seed": data_seed,
                            **model,
                            "lr": lr,
                            "init_seed": init_seed,
                            "val_loss": model_index + train_size / 100 + lr,
                        }
                    )
            for data_seed, init_seed in EVALUATION_SEEDS:
                rows.append(
                    {
                        "protocol": "scaling-v1",
                        "phase": "evaluation",
                        "train_size": train_size,
                        "data_seed": data_seed,
                        **model,
                        "lr": LEARNING_RATES[0],
                        "init_seed": init_seed,
                        "test_loss": model_index + train_size / 100,
                    }
                )
    return pd.DataFrame(rows).reindex(columns=RESULT_SCHEMA.raw_columns)


def _validate(raw: pd.DataFrame) -> dict[str, object]:
    return validate_results(
        raw,
        schema=RESULT_SCHEMA,
        expected_model_rows=EXPECTED_MODELS,
        train_sizes=TRAIN_SIZES,
        learning_rates=LEARNING_RATES,
        tuning_seeds=TUNING_SEEDS,
        evaluation_seeds=EVALUATION_SEEDS,
    )


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


def test_evaluation_runs_coalesce_checkpoints_with_the_same_learning_rate():
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        runs = select_evaluation_runs(
            _tuning(),
            schema=RESULT_SCHEMA,
            evaluation_seeds=SeedGrid(
                data_seeds=range(2, 4),
                init_seeds=range(5, 6),
            ),
            model_specs={("spectral", 3): ModelSpec("spectral", 3)},
        )

    observed = {
        (run.config.data_seed, run.config.lr): run.test_checkpoints for run in runs
    }
    assert observed == {
        (2, 0.01): (10, 30),
        (2, 0.1): (20,),
        (3, 0.01): (10, 30),
        (3, 0.1): (20,),
    }
    assert {run.config.init_seed for run in runs} == {5}


def test_evaluation_run_selection_rejects_mixed_experiments():
    tuning = pd.concat(
        [_tuning(), _tuning().assign(protocol="scaling-v2")],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly one experiment"):
        select_evaluation_runs(
            tuning,
            schema=RESULT_SCHEMA,
            evaluation_seeds=SeedGrid(),
            model_specs={("spectral", 3): ModelSpec("spectral", 3)},
        )


def test_evaluation_summary_uses_only_selected_learning_rate_rows():
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

    selected = select_evaluations(
        pd.concat((tuning, evaluation), ignore_index=True),
        schema=RESULT_SCHEMA,
    )
    summary = summarize_evaluations(
        selected,
        schema=RESULT_SCHEMA,
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


def test_evaluation_selection_keeps_experiments_separate():
    rows = []
    for protocol, selected_lr in (("scaling-v1", 0.01), ("scaling-v2", 0.1)):
        common = {
            "protocol": protocol,
            "model": "linear",
            "dim": 0,
            "train_size": 10,
        }
        for lr in LEARNING_RATES:
            rows.append(
                common
                | {
                    "phase": "tuning",
                    "lr": lr,
                    "val_loss": float(lr != selected_lr),
                }
            )
            rows.append(
                common
                | {
                    "phase": "evaluation",
                    "lr": lr,
                    "test_loss": 100 * selected_lr + lr,
                }
            )

    selected = select_evaluations(pd.DataFrame(rows), schema=RESULT_SCHEMA)

    assert selected[["protocol", "lr"]].to_dict("records") == [
        {"protocol": "scaling-v1", "lr": 0.01},
        {"protocol": "scaling-v2", "lr": 0.1},
    ]


def test_validate_results_accepts_a_complete_experiment(complete_raw):
    assert _validate(complete_raw) == {"protocol": "scaling-v1"}


def test_validate_results_checks_schema_and_run_identity(complete_raw):
    with pytest.raises(ValueError, match="schema"):
        _validate(complete_raw.drop(columns="test_loss"))

    missing_identity = complete_raw.copy()
    missing_identity.loc[0, "model"] = np.nan
    with pytest.raises(ValueError, match="identity"):
        _validate(missing_identity)

    duplicate = pd.concat((complete_raw, complete_raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _validate(duplicate)


def test_validate_results_checks_experiment_model_and_checkpoint_grids(
    complete_raw,
):
    mixed_experiments = complete_raw.copy()
    mixed_experiments.loc[0, "protocol"] = "scaling-v2"
    with pytest.raises(ValueError, match="one experiment"):
        _validate(mixed_experiments)

    with pytest.raises(ValueError, match="tuning and evaluation phases"):
        _validate(complete_raw.loc[complete_raw["phase"] == "tuning"])

    missing_model = complete_raw.loc[complete_raw["model"] != "spectral"]
    with pytest.raises(ValueError, match="model/capacity grid"):
        _validate(missing_model)

    for phase in ("tuning", "evaluation"):
        incomplete = complete_raw.loc[
            ~(
                (complete_raw["phase"] == phase)
                & (complete_raw["model"] == "spectral")
                & (complete_raw["train_size"] == TRAIN_SIZES[0])
            )
        ]
        with pytest.raises(ValueError, match=phase):
            _validate(incomplete)


def test_validate_results_separates_phase_metrics(complete_raw):
    leaked_test = complete_raw.copy()
    leaked_test.loc[leaked_test["phase"] == "tuning", "test_loss"] = 0.5
    with pytest.raises(ValueError, match="test metrics"):
        _validate(leaked_test)

    leaked_validation = complete_raw.copy()
    leaked_validation.loc[
        leaked_validation["phase"] == "evaluation", "val_loss"
    ] = 0.5
    with pytest.raises(ValueError, match="validation metrics"):
        _validate(leaked_validation)

    nonfinite_test = complete_raw.copy()
    index = nonfinite_test.index[nonfinite_test["phase"] == "evaluation"][0]
    nonfinite_test.loc[index, "test_loss"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _validate(nonfinite_test)


def test_validate_results_checks_learning_rate_and_seed_grids(complete_raw):
    curve = (
        (complete_raw["phase"] == "tuning")
        & (complete_raw["model"] == "spectral")
        & (complete_raw["train_size"] == TRAIN_SIZES[0])
    )
    missing_lr = complete_raw.loc[
        ~(curve & (complete_raw["lr"] == LEARNING_RATES[-1]))
    ]
    with pytest.raises(ValueError, match="learning-rate grid"):
        _validate(missing_lr)

    missing_tuning_seed = complete_raw.drop(
        complete_raw.index[curve & (complete_raw["lr"] == LEARNING_RATES[-1])][0]
    )
    with pytest.raises(ValueError, match="tuning seeds"):
        _validate(missing_tuning_seed)

    unexpected_evaluation_seed = complete_raw.copy()
    index = unexpected_evaluation_seed.index[
        unexpected_evaluation_seed["phase"] == "evaluation"
    ][0]
    unexpected_evaluation_seed.loc[index, "init_seed"] = 99
    with pytest.raises(ValueError, match="evaluation seeds"):
        _validate(unexpected_evaluation_seed)


def test_validate_results_checks_selected_and_finite_learning_rates(complete_raw):
    tuning_curve = (
        (complete_raw["phase"] == "tuning")
        & (complete_raw["model"] == "spectral")
        & (complete_raw["train_size"] == TRAIN_SIZES[0])
    )
    nonfinite_candidate = complete_raw.copy()
    nonfinite_candidate.loc[
        tuning_curve & (complete_raw["lr"] == LEARNING_RATES[-1]), "val_loss"
    ] = np.nan
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        _validate(nonfinite_candidate)

    no_finite_candidate = complete_raw.copy()
    no_finite_candidate.loc[tuning_curve, "val_loss"] = np.nan
    with pytest.warns(RuntimeWarning, match="nonfinite"):
        with pytest.raises(ValueError, match="no finite validation"):
            _validate(no_finite_candidate)

    wrong_lr = complete_raw.copy()
    evaluation_curve = (
        (wrong_lr["phase"] == "evaluation")
        & (wrong_lr["model"] == "spectral")
        & (wrong_lr["train_size"] == TRAIN_SIZES[0])
    )
    wrong_lr.loc[evaluation_curve, "lr"] = LEARNING_RATES[-1]
    with pytest.raises(ValueError, match="selected LR"):
        _validate(wrong_lr)
