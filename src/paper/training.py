from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any, Protocol

import fitstream as fts
import numpy as np
import pandas as pd
import torch
from torch import nn

from paper.tasks import Task

type Event = dict[str, Any]
type ModelInputs = tuple[torch.Tensor, ...]
type Batch = tuple[ModelInputs, torch.Tensor]
type BatchFactory = Callable[[], Iterable[Batch]]
type Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
type Evaluator = Callable[[nn.Module, Iterable[Batch]], dict[str, float]]


@dataclass(frozen=True)
class Objective:
    loss: Loss
    validation_metrics: Evaluator
    test_metrics: Evaluator


class ScalingTask(Protocol):
    def train_batches(self, max_examples: int) -> Iterable[Batch]: ...

    def val_batches(self) -> Iterable[Batch]: ...

    def test_batches(self) -> Iterable[Batch]: ...


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


def _predict(
    model: nn.Module, model_inputs: ModelInputs, labels: torch.Tensor
) -> torch.Tensor:
    predictions = model(*model_inputs)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"predictions have shape {tuple(predictions.shape)}; "
            f"labels have shape {tuple(labels.shape)}"
        )
    return predictions


def train_scaling_events(
    task: ScalingTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
    loss: Loss,
) -> Iterator[Event]:
    checkpoints = _checkpoints(checkpoints)

    optimizers = _adam_optimizers(model, lr=lr)
    batches = iter(task.train_batches(checkpoints[-1]))
    examples_seen = 0
    train_seconds = 0.0

    for checkpoint in checkpoints:
        model.train()
        segment_started = perf_counter()
        for model_inputs, labels in batches:
            batch_loss = loss(_predict(model, model_inputs, labels), labels)

            model.zero_grad(set_to_none=True)
            batch_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            examples_seen += len(labels)
            if examples_seen >= checkpoint:
                break

        train_seconds += perf_counter() - segment_started
        assert examples_seen == checkpoint, (
            f"training stream reached {examples_seen} examples; "
            f"expected checkpoint {checkpoint}"
        )
        yield {
            "train_size": examples_seen,
            "train_seconds": train_seconds,
            "model": model,
        }

    sentinel = object()
    assert next(batches, sentinel) is sentinel, (
        f"training stream yielded more than {checkpoints[-1]} examples"
    )


def evaluate_binary(
    model: nn.Module,
    batches: Iterable[Batch],
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
            for model_inputs, labels in batches:
                logits = _predict(model, model_inputs, labels)
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


def evaluate_regression(
    model: nn.Module,
    batches: Iterable[Batch],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    squared_error = 0.0
    samples = 0
    try:
        with torch.inference_mode():
            for model_inputs, labels in batches:
                errors = _predict(model, model_inputs, labels) - labels
                squared_error += errors.square().sum().item()
                samples += labels.numel()
    finally:
        model.train(was_training)

    if samples == 0:
        raise ValueError("evaluation data must not be empty")
    return {"rmse": (squared_error / samples) ** 0.5}


BINARY_OBJECTIVE = Objective(
    loss=nn.functional.binary_cross_entropy_with_logits,
    validation_metrics=partial(evaluate_binary, include_brier=False),
    test_metrics=evaluate_binary,
)
REGRESSION_OBJECTIVE = Objective(
    loss=nn.functional.mse_loss,
    validation_metrics=evaluate_regression,
    test_metrics=evaluate_regression,
)


def metrics_on(
    name: str,
    batches: BatchFactory,
    evaluate: Evaluator,
) -> Callable[[Event], Event]:
    elapsed_seconds = 0.0

    def augment(event: Event) -> Event:
        nonlocal elapsed_seconds
        started = perf_counter()
        metrics = evaluate(event["model"], batches())
        elapsed_seconds += perf_counter() - started
        return {
            **{f"{name}_{key}": value for key, value in metrics.items()},
            f"{name}_seconds": elapsed_seconds,
        }

    return augment


def _events_at_train_sizes(
    events: Iterable[Event], train_sizes: Iterable[int]
) -> Iterator[Event]:
    selected = frozenset(train_sizes)
    for event in events:
        if event["train_size"] in selected:
            yield event


def _scaling_results(
    task: ScalingTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
    selected_checkpoints: Iterable[int] | None,
    loss: Loss,
    metric_name: str,
    batches: BatchFactory,
    evaluate: Evaluator,
) -> pd.DataFrame:
    checkpoints = _checkpoints(checkpoints)
    selected = (
        checkpoints
        if selected_checkpoints is None
        else tuple(map(int, selected_checkpoints))
    )
    if not set(selected) <= set(checkpoints):
        raise ValueError("metric checkpoints must be drawn from checkpoints")

    events = fts.pipe(
        _events_at_train_sizes(
            train_scaling_events(
                task,
                model,
                lr=lr,
                checkpoints=checkpoints,
                loss=loss,
            ),
            selected,
        ),
        fts.augment(metrics_on(metric_name, batches, evaluate)),
    )
    return fts.collect_pd(events)


def tune_scaling_stream(
    task: ScalingTask,
    model: nn.Module,
    *,
    objective: Objective,
    lr: float,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    return _scaling_results(
        task,
        model,
        lr=lr,
        checkpoints=checkpoints,
        selected_checkpoints=None,
        loss=objective.loss,
        metric_name="val",
        batches=task.val_batches,
        evaluate=objective.validation_metrics,
    )


def fit_and_test_scaling(
    task: ScalingTask,
    model: nn.Module,
    *,
    objective: Objective,
    lr: float,
    checkpoints: Iterable[int],
    test_checkpoints: Iterable[int],
) -> pd.DataFrame:
    return _scaling_results(
        task,
        model,
        lr=lr,
        checkpoints=checkpoints,
        selected_checkpoints=test_checkpoints,
        loss=objective.loss,
        metric_name="test",
        batches=task.test_batches,
        evaluate=objective.test_metrics,
    )
