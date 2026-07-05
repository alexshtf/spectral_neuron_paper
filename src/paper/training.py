from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import Any

import fitstream as fts
import numpy as np
import pandas as pd
import torch
from torch import nn

from paper.tasks import Task

type Event = dict[str, Any]


def train_events(
    task: Task,
    model: nn.Module,
    *,
    lr: float,
    train_seed: int,
    steps: int,
    checkpoints: Iterable[int],
) -> Iterator[Event]:
    torch.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    batches = task.train_batches(rng)
    checkpoint_set = set(checkpoints)
    start_time = perf_counter()

    for step in range(1, steps + 1):
        x, y = next(batches)

        optimizer.zero_grad()
        train_loss = loss_fn(model(x), y)
        train_loss.backward()
        optimizer.step()

        if step in checkpoint_set:
            yield {
                "step": step,
                "model": model,
                "train_rmse": float(train_loss.detach().sqrt()),
                "seconds": perf_counter() - start_time,
            }


def rmse_on(
    name: str, x: torch.Tensor, y: torch.Tensor
) -> Callable[[Event], dict[str, float]]:
    def augment(ev: Event) -> dict[str, float]:
        model = ev["model"]
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                pred = model(x)
                rmse = torch.mean((pred - y) ** 2).sqrt().item()
        finally:
            model.train(was_training)
        return {f"{name}_rmse": rmse}

    return augment


def run_one_stream(
    task: Task,
    model: nn.Module,
    *,
    lr: float,
    train_seed: int,
    steps: int,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    events = fts.pipe(
        train_events(
            task,
            model,
            lr=lr,
            train_seed=train_seed,
            steps=steps,
            checkpoints=checkpoints,
        ),
        fts.augment(rmse_on("val", task.x_val, task.y_val)),
        fts.augment(rmse_on("test", task.x_test, task.y_test)),
    )
    return fts.collect_pd(events).drop(columns=["model"], errors="ignore")
