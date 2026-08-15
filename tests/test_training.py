import fitstream as fts
import pytest
import torch
from torch import nn

from paper.models import SparseLinear
from paper.training import (
    BINARY_OBJECTIVE,
    REGRESSION_OBJECTIVE,
    _adam_optimizers,
    evaluate_binary,
    evaluate_on,
    evaluate_regression,
    fit_and_evaluate,
    fit_test_trajectory,
    fit_validation_trajectory,
    train_events,
)


class ModeRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.modes: list[bool] = []

    def forward(self, x: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        self.modes.append(self.training)
        return torch.zeros(x.shape[:-1])


def test_checkpoint_metrics_describe_post_update_model():
    x = torch.ones(1, 1)
    y = torch.ones(1, 1)

    class Task:
        @staticmethod
        def train_batches(max_examples: int):
            for _ in range(max_examples):
                yield (x,), y

        @staticmethod
        def val_batches():
            yield (x,), y

        @staticmethod
        def test_batches():
            yield (x,), y

    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)

    result = fit_and_evaluate(
        Task(),
        model,
        objective=REGRESSION_OBJECTIVE,
        lr=0.1,
        checkpoints=(1, 2),
    )

    with torch.inference_mode():
        expected_rmse = torch.mean((model(x) - y) ** 2).sqrt().item()

    assert result["val_rmse"].iloc[-1] == pytest.approx(expected_rmse)


@pytest.mark.parametrize("training", [False, True])
def test_evaluation_restores_model_mode(training):
    model = ModeRecorder()
    x = torch.zeros(2, 1)
    y = torch.zeros(2)

    model.train(training)
    evaluate_binary(model, [((x.long(), x), y)])
    assert model.modes.pop() is False
    assert model.training is training

    model.train(training)
    evaluate_regression(model, [((x,), y)])
    assert model.modes.pop() is False
    assert model.training is training


def test_regression_metrics_are_global_over_uneven_batches():
    model = ModeRecorder()
    batches = [
        ((torch.zeros(1, 1),), torch.tensor([3.0])),
        ((torch.zeros(3, 1),), torch.tensor([4.0, 0.0, 0.0])),
    ]

    metrics = evaluate_regression(model, batches)

    assert metrics == {"rmse": 2.5}


def test_binary_models_use_adam_and_sparse_adam():
    optimizers = _adam_optimizers(SparseLinear(3, 1), lr=0.1)

    assert tuple(map(type, optimizers)) == (torch.optim.Adam, torch.optim.SparseAdam)


def test_training_consumes_each_nested_prefix_once():
    seen: list[int] = []
    calls: list[int] = []

    class RecordingTask:
        @staticmethod
        def train_batches(max_examples: int):
            calls.append(max_examples)
            for batch_start in range(0, max_examples, 2):
                rows = list(range(batch_start, min(batch_start + 2, max_examples)))
                seen.extend(rows)
                yield (
                    (
                        torch.zeros(len(rows), 1, dtype=torch.long),
                        torch.ones(len(rows), 1),
                    ),
                    torch.zeros(len(rows)),
                )

    model = SparseLinear(num_features=1, num_fields=1)
    events = list(
        train_events(
            RecordingTask(),
            model,
            lr=0.1,
            checkpoints=(4, 7),
            loss=BINARY_OBJECTIVE.loss,
        )
    )

    assert calls == [7]
    assert seen == list(range(7))
    assert [(event["step"], event["train_size"]) for event in events] == [
        (2, 4),
        (4, 7),
    ]
    assert all(
        "train_seconds" in event and event["model"] is model for event in events
    )
    assert all(isinstance(event, fts.Event) for event in events)


@pytest.mark.parametrize(
    ("batch_sizes", "message"),
    [
        ((2,), "reached 2 examples; expected checkpoint 3"),
        ((4,), "reached 4 examples; expected checkpoint 3"),
        ((3, 1), "yielded more than 3 examples"),
    ],
)
def test_trainer_asserts_actual_example_count(batch_sizes, message):
    class InvalidTask:
        @staticmethod
        def train_batches(max_examples: int):
            for size in batch_sizes:
                yield (torch.zeros(size, 1),), torch.zeros(size, 1)

    with pytest.raises(AssertionError, match=message):
        list(
            train_events(
                InvalidTask(),
                nn.Linear(1, 1),
                lr=0.1,
                checkpoints=(3,),
                loss=nn.functional.mse_loss,
            )
        )


def test_checkpoint_on_minibatch_boundary_does_not_change_later_training():
    rows = torch.arange(16, dtype=torch.float32)
    features = torch.stack((rows, rows.remainder(3)), dim=-1)
    labels = (torch.arange(16) % 2).float()

    class DenseTask:
        @staticmethod
        def train_batches(max_examples: int):
            for batch_start in range(0, max_examples, 4):
                batch_stop = min(batch_start + 4, max_examples)
                yield (
                    (features[batch_start:batch_stop],),
                    labels[batch_start:batch_stop],
                )

    def trained_parameters(
        checkpoints: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        with torch.random.fork_rng():
            torch.manual_seed(11)
            model = nn.Sequential(
                nn.Linear(2, 1),
                nn.Flatten(start_dim=-2, end_dim=-1),
            )
        list(
            train_events(
                DenseTask(),
                model,
                lr=1e-2,
                checkpoints=checkpoints,
                loss=BINARY_OBJECTIVE.loss,
            )
        )
        return tuple(parameter.detach().clone() for parameter in model.parameters())

    without_intermediate = trained_parameters((16,))
    with_intermediate = trained_parameters((8, 16))

    for expected, actual in zip(without_intermediate, with_intermediate, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_time_is_cumulative_but_excludes_suspension(monkeypatch):
    ticks = iter((0.0, 2.0, 10.0, 13.0))
    monkeypatch.setattr("paper.training.perf_counter", lambda: next(ticks))

    class Task:
        @staticmethod
        def train_batches(max_examples: int):
            for _ in range(max_examples):
                yield (
                    (torch.zeros(1, 1, dtype=torch.long),),
                    torch.zeros(1),
                )

    events = list(
        train_events(
            Task(),
            SparseLinear(num_features=1, num_fields=1),
            lr=0.1,
            checkpoints=(1, 2),
            loss=BINARY_OBJECTIVE.loss,
        )
    )

    assert [event["train_seconds"] for event in events] == [2.0, 5.0]


def test_evaluation_time_is_cumulative(monkeypatch):
    ticks = iter((0.0, 2.0, 10.0, 13.0))
    monkeypatch.setattr("paper.training.perf_counter", lambda: next(ticks))
    augment = evaluate_on("val", lambda: (), lambda *_: {"logloss": 0.5})

    first = augment({"model": object()})
    second = augment({"model": object()})

    assert first == {"val_logloss": 0.5, "val_seconds": 2.0}
    assert second == {"val_logloss": 0.5, "val_seconds": 5.0}


def test_binary_scaling_tests_only_selected_checkpoints():
    seen: list[int] = []

    class RecordingTask:
        @staticmethod
        def train_batches(max_examples: int):
            seen.extend(range(max_examples))
            for _ in range(max_examples):
                yield (
                    (
                        torch.zeros(1, 1, dtype=torch.long),
                        torch.ones(1, 1),
                    ),
                    torch.zeros(1),
                )

        @staticmethod
        def test_batches():
            yield (
                (torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)),
                torch.zeros(1),
            )

    result = fit_test_trajectory(
        RecordingTask(),
        SparseLinear(num_features=1, num_fields=1),
        objective=BINARY_OBJECTIVE,
        lr=0.1,
        checkpoints=(3, 5, 7),
        test_checkpoints=(3, 7),
    )

    assert seen == list(range(7))
    assert result["train_size"].tolist() == [3, 7]


def test_binary_tuning_evaluates_every_checkpoint():
    validation_calls = 0

    class Task:
        @staticmethod
        def train_batches(max_examples: int):
            for _ in range(max_examples):
                yield (
                    (torch.zeros(1, 1, dtype=torch.long),),
                    torch.zeros(1),
                )

        @staticmethod
        def val_batches():
            nonlocal validation_calls
            validation_calls += 1
            yield (torch.zeros(1, 1, dtype=torch.long),), torch.zeros(1)

    result = fit_validation_trajectory(
        Task(),
        SparseLinear(num_features=1, num_fields=1),
        objective=BINARY_OBJECTIVE,
        lr=0.1,
        checkpoints=(3, 5, 7),
    )

    assert result["train_size"].tolist() == [3, 5, 7]
    assert validation_calls == 3


def test_regression_scaling_keeps_validation_and_test_separate():
    validation_calls = 0
    test_calls = 0

    class Task:
        @staticmethod
        def train_batches(max_examples: int):
            for _ in range(max_examples):
                yield (torch.ones(1, 1),), torch.ones(1)

        @staticmethod
        def val_batches():
            nonlocal validation_calls
            validation_calls += 1
            yield (torch.ones(1, 1),), torch.ones(1)

        @staticmethod
        def test_batches():
            nonlocal test_calls
            test_calls += 1
            yield (torch.ones(1, 1),), torch.ones(1)

    def model():
        return nn.Sequential(nn.Linear(1, 1), nn.Flatten(start_dim=0))

    tuning = fit_validation_trajectory(
        Task(),
        model(),
        objective=REGRESSION_OBJECTIVE,
        lr=0.1,
        checkpoints=(3, 5, 7),
    )
    testing = fit_test_trajectory(
        Task(),
        model(),
        objective=REGRESSION_OBJECTIVE,
        lr=0.1,
        checkpoints=(3, 5, 7),
        test_checkpoints=(3, 7),
    )

    assert tuning["train_size"].tolist() == [3, 5, 7]
    assert testing["train_size"].tolist() == [3, 7]
    assert validation_calls == 3
    assert test_calls == 2
