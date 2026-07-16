from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import Any, Protocol

import fitstream as fts
import numpy as np
import pandas as pd
import torch
from torch import nn

from paper.tasks import Task

type Event = dict[str, Any]
type TensorBatch = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
type BatchFactory = Callable[[], Iterable[TensorBatch]]


class BinaryTask(Protocol):
    def train_batches(self, start: int, stop: int) -> Iterable[TensorBatch]: ...

    def val_batches(self) -> Iterable[TensorBatch]: ...

    def test_batches(self) -> Iterable[TensorBatch]: ...


def _checkpoints(values: Iterable[int]) -> tuple[int, ...]:
    checkpoints = tuple(map(int, values))
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("checkpoints must be non-empty, unique, and increasing")
    if checkpoints[0] <= 0:
        raise ValueError("checkpoints must be positive")
    return checkpoints


def train_events(
    task: Task,
    model: nn.Module,
    *,
    lr: float,
    train_seed: int,
    checkpoints: Iterable[int],
) -> Iterator[Event]:
    torch.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    batches = task.train_batches(rng)
    checkpoints = _checkpoints(checkpoints)
    checkpoint_set = set(checkpoints)
    for step in range(1, checkpoints[-1] + 1):
        x, y = next(batches)

        optimizer.zero_grad()
        train_loss = loss_fn(model(x), y)
        train_loss.backward()
        optimizer.step()

        if step in checkpoint_set:
            yield {"step": step, "model": model}


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


def run_one_stream(
    task: Task,
    model: nn.Module,
    *,
    lr: float,
    train_seed: int,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    events = fts.pipe(
        train_events(
            task,
            model,
            lr=lr,
            train_seed=train_seed,
            checkpoints=checkpoints,
        ),
        fts.augment(rmse_on("val", task.x_val, task.y_val)),
        fts.augment(rmse_on("test", task.x_test, task.y_test)),
    )
    return fts.collect_pd(events)


def _adam_optimizers(
    model: nn.Module,
    *,
    lr: float,
) -> list[torch.optim.Optimizer]:
    sparse_parameters = [
        module.weight
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.sparse
    ]
    sparse_ids = {id(parameter) for parameter in sparse_parameters}
    dense_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in sparse_ids
    ]

    optimizers: list[torch.optim.Optimizer] = []
    if dense_parameters:
        optimizers.append(torch.optim.Adam(dense_parameters, lr=lr))
    if sparse_parameters:
        optimizers.append(torch.optim.SparseAdam(sparse_parameters, lr=lr))
    return optimizers


def train_binary_scaling_events(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
) -> Iterator[Event]:
    checkpoints = _checkpoints(checkpoints)

    optimizers = _adam_optimizers(model, lr=lr)
    start = 0
    train_seconds = 0.0

    for train_size in checkpoints:
        model.train()
        segment_started = perf_counter()
        for feature_ids, feature_values, labels in task.train_batches(
            start, train_size
        ):
            logits = model(feature_ids, feature_values)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

            model.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        train_seconds += perf_counter() - segment_started
        start = train_size
        yield {
            "train_size": train_size,
            "train_seconds": train_seconds,
            "model": model,
        }


def evaluate_binary(
    model: nn.Module,
    batches: Iterable[TensorBatch],
    *,
    include_brier: bool = True,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_logloss = 0.0
    total_brier = 0.0
    total_samples = 0
    try:
        with torch.inference_mode():
            for feature_ids, feature_values, labels in batches:
                logits = model(feature_ids, feature_values)
                total_logloss += nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                ).item()
                if include_brier:
                    total_brier += (
                        (torch.sigmoid(logits) - labels).square().sum().item()
                    )
                total_samples += len(labels)
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("evaluation data must not be empty")
    metrics = {"logloss": total_logloss / total_samples}
    if include_brier:
        metrics["brier"] = total_brier / total_samples
    return metrics


def binary_metrics_on(
    name: str,
    batches: BatchFactory,
    *,
    include_brier: bool = True,
) -> Callable[[Event], Event]:
    elapsed_seconds = 0.0

    def augment(event: Event) -> Event:
        nonlocal elapsed_seconds
        started = perf_counter()
        metrics = evaluate_binary(
            event["model"], batches(), include_brier=include_brier
        )
        elapsed_seconds += perf_counter() - started
        return {
            **{f"{name}_{key}": value for key, value in metrics.items()},
            f"{name}_seconds": elapsed_seconds,
        }

    return augment


def tune_binary_scaling_stream(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    events = fts.pipe(
        train_binary_scaling_events(task, model, lr=lr, checkpoints=checkpoints),
        fts.augment(binary_metrics_on("val", task.val_batches, include_brier=False)),
    )
    return fts.collect_pd(events)


def fit_and_test_binary_scaling(
    task: BinaryTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
    test_checkpoints: Iterable[int],
) -> pd.DataFrame:
    checkpoints = tuple(map(int, checkpoints))
    test_checkpoints = set(map(int, test_checkpoints))
    if not test_checkpoints <= set(checkpoints):
        raise ValueError("test_checkpoints must be drawn from checkpoints")
    trained = train_binary_scaling_events(
        task,
        model,
        lr=lr,
        checkpoints=checkpoints,
    )
    events = fts.pipe(
        (event for event in trained if event["train_size"] in test_checkpoints),
        fts.augment(binary_metrics_on("test", task.test_batches)),
    )
    return fts.collect_pd(events)
