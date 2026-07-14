import numpy as np
import pytest
import torch
from torch import nn

from paper.models import SparseLinear
from paper.tasks import Task
from paper.training import (
    fit_and_test_binary_scaling,
    run_one_stream,
    train_binary_scaling_events,
)


class CountingLinear(nn.Module):
    def __init__(self, clock: list[int]):
        super().__init__()
        self.clock = clock
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.clock[0] += 1
        return self.linear(x).squeeze(-1)


def test_checkpoint_metrics_describe_post_update_model(monkeypatch):
    clock = [0]
    monkeypatch.setattr("paper.training.perf_counter", lambda: float(clock[0]))

    x = torch.ones(1, 1)
    y = torch.ones(1)

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
    model = CountingLinear(clock)

    result = run_one_stream(
        task,
        model,
        lr=0.1,
        train_seed=0,
        steps=2,
        checkpoints=(1, 2),
    )

    with torch.inference_mode():
        expected_rmse = torch.mean((model(x) - y) ** 2).sqrt().item()

    assert result["train_rmse"].iloc[-1] == pytest.approx(expected_rmse)
    assert result["elapsed_seconds"].tolist() == [4.0, 8.0]


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
