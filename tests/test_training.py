import numpy as np
import pytest
import torch
from torch import nn

from paper.tasks import Task
from paper.training import run_one_stream


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
