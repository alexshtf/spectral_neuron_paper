import numpy as np
import pytest
import torch
from torch import nn

from paper.models import SparseLinear
from paper.tasks import Task
from paper.training import (
    _adam_optimizers,
    binary_metrics_on,
    evaluate_binary,
    evaluate_rmse,
    fit_and_test_binary_scaling,
    run_one_stream,
    train_binary_scaling_events,
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

    def train_batches(_: np.random.Generator):
        while True:
            yield x, y

    task = Task(
        input_dim=1,
        x_val=x,
        y_val=y,
        x_test=x,
        y_test=y,
        train_batches=train_batches,
    )
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)

    result = run_one_stream(
        task,
        model,
        lr=0.1,
        train_seed=0,
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
    evaluate_rmse(model, x, y)
    assert model.modes.pop() is False
    assert model.training is training

    model.train(training)
    evaluate_binary(model, [(x.long(), x, y)])
    assert model.modes.pop() is False
    assert model.training is training


def test_binary_models_use_adam_and_sparse_adam():
    optimizers = _adam_optimizers(SparseLinear(3, 1), lr=0.1)

    assert tuple(map(type, optimizers)) == (torch.optim.Adam, torch.optim.SparseAdam)


def test_binary_scaling_consumes_each_nested_prefix_once():
    seen: list[int] = []

    class RecordingTask:
        @staticmethod
        def train_batches(start: int, stop: int):
            for batch_start in range(start, stop, 2):
                rows = list(range(batch_start, min(batch_start + 2, stop)))
                seen.extend(rows)
                yield (
                    torch.zeros(len(rows), 1, dtype=torch.long),
                    torch.ones(len(rows), 1),
                    torch.zeros(len(rows)),
                )

    model = SparseLinear(num_features=1, num_fields=1)
    events = list(
        train_binary_scaling_events(
            RecordingTask(),
            model,
            lr=0.1,
            checkpoints=(3, 7),
        )
    )

    assert seen == list(range(7))
    assert [event["train_size"] for event in events] == [3, 7]


def test_binary_training_time_is_cumulative_but_excludes_suspension(monkeypatch):
    ticks = iter((0.0, 2.0, 10.0, 13.0))
    monkeypatch.setattr("paper.training.perf_counter", lambda: next(ticks))

    class Task:
        @staticmethod
        def train_batches(start: int, stop: int):
            yield (
                torch.zeros(stop - start, 1, dtype=torch.long),
                None,
                torch.zeros(stop - start),
            )

    events = list(
        train_binary_scaling_events(
            Task(),
            SparseLinear(num_features=1, num_fields=1),
            lr=0.1,
            checkpoints=(1, 2),
        )
    )

    assert [event["train_seconds"] for event in events] == [2.0, 5.0]


def test_binary_evaluation_time_is_cumulative(monkeypatch):
    ticks = iter((0.0, 2.0, 10.0, 13.0))
    monkeypatch.setattr("paper.training.perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        "paper.training.evaluate_binary",
        lambda *_args, **_kwargs: {"logloss": 0.5},
    )
    augment = binary_metrics_on("val", lambda: ())

    first = augment({"model": object()})
    second = augment({"model": object()})

    assert first == {"val_logloss": 0.5, "val_seconds": 2.0}
    assert second == {"val_logloss": 0.5, "val_seconds": 5.0}


def test_binary_scaling_tests_only_selected_checkpoints():
    seen: list[int] = []

    class RecordingTask:
        @staticmethod
        def train_batches(start: int, stop: int):
            seen.extend(range(start, stop))
            yield (
                torch.zeros(stop - start, 1, dtype=torch.long),
                torch.ones(stop - start, 1),
                torch.zeros(stop - start),
            )

        @staticmethod
        def test_batches():
            yield (
                torch.zeros(1, 1, dtype=torch.long),
                torch.ones(1, 1),
                torch.zeros(1),
            )

    result = fit_and_test_binary_scaling(
        RecordingTask(),
        SparseLinear(num_features=1, num_fields=1),
        lr=0.1,
        checkpoints=(3, 5, 7),
        test_checkpoints=(3, 7),
    )

    assert seen == list(range(7))
    assert result["train_size"].tolist() == [3, 7]
