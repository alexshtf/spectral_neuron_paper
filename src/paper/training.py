from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import fitstream as fts
import numpy as np
import pandas as pd
import torch
from torch import nn

from paper.tasks import Task

type Event = dict[str, Any]
type TensorBatch = tuple[torch.Tensor, torch.Tensor]
type BatchFactory = Callable[[], Iterable[TensorBatch]]


class BinaryTask(Protocol):
    def train_batches(self, epoch: int) -> Iterable[TensorBatch]: ...

    def val_batches(self) -> Iterable[TensorBatch]: ...

    def test_batches(self) -> Iterable[TensorBatch]: ...


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
                "train_rmse": evaluate_rmse(model, x, y),
            }


def evaluate_rmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            pred = model(x)
            return torch.mean((pred - y) ** 2).sqrt().item()
    finally:
        model.train(was_training)


def rmse_on(
    name: str, x: torch.Tensor, y: torch.Tensor
) -> Callable[[Event], dict[str, float]]:
    def augment(ev: Event) -> dict[str, float]:
        return {f"{name}_rmse": evaluate_rmse(ev["model"], x, y)}

    return augment


def elapsed_since(start_time: float) -> Callable[[Event], dict[str, float]]:
    def augment(_: Event) -> dict[str, float]:
        return {"elapsed_seconds": perf_counter() - start_time}

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
    start_time = perf_counter()
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
        fts.augment(elapsed_since(start_time)),
    )
    return fts.collect_pd(events).drop(columns=["model"], errors="ignore")


def train_binary_events(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    max_epochs: int,
) -> Iterator[Event]:
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for feature_ids, labels in task.train_batches(epoch):
            logits = model(feature_ids)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                optimizer.step()

            total_loss += loss.detach().item() * len(labels)
            total_samples += len(labels)

        yield {
            "epoch": epoch,
            "model": model,
            "train_logloss": total_loss / total_samples,
        }


def evaluate_binary(
    model: nn.Module, batches: Iterable[TensorBatch]
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_logloss = 0.0
    total_brier = 0.0
    total_samples = 0
    try:
        with torch.inference_mode():
            for feature_ids, labels in batches:
                logits = model(feature_ids)
                total_logloss += nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                ).item()
                total_brier += (torch.sigmoid(logits) - labels).square().sum().item()
                total_samples += len(labels)
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("evaluation data must not be empty")
    return {
        "logloss": total_logloss / total_samples,
        "brier": total_brier / total_samples,
    }


def binary_metrics_on(name: str, batches: BatchFactory) -> Callable[[Event], Event]:
    def augment(event: Event) -> Event:
        metrics = evaluate_binary(event["model"], batches())
        return {f"{name}_{key}": value for key, value in metrics.items()}

    return augment


@dataclass
class BestEpoch:
    key: str
    value: float = float("inf")
    epoch: int = 0

    def update(self, event: Event) -> None:
        value = float(event[self.key])
        if value < self.value:
            self.value = value
            self.epoch = int(event["epoch"])


def tune_binary_stream(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
) -> pd.DataFrame:
    start_time = perf_counter()
    best = BestEpoch("val_logloss")
    events = fts.pipe(
        train_binary_events(task, model, lr=lr, max_epochs=max_epochs),
        fts.augment(binary_metrics_on("val", task.val_batches)),
        fts.augment(elapsed_since(start_time)),
        fts.tap(best.update),
        fts.early_stop("val_logloss", patience=patience),
    )
    history = fts.collect_pd(events).drop(columns=["model"], errors="ignore")

    selected = history.loc[history["epoch"] == best.epoch].copy()
    selected["epochs_run"] = len(history)
    return selected.reset_index(drop=True)


def fit_and_test_binary(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    epochs: int,
) -> dict[str, float]:
    events = fts.pipe(
        train_binary_events(task, model, lr=lr, max_epochs=epochs),
        fts.take(epochs),
    )
    for _ in events:
        pass
    return evaluate_binary(model, task.test_batches())
