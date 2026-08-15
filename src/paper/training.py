from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any

import fitstream as fts
import pandas as pd
import torch
from torch import nn

from paper.tasks import Batch, ModelInputs, Task, TrainTask


type Event = dict[str, Any]
type BatchFactory = Callable[[], Iterable[Batch]]
type Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
type Evaluator = Callable[[nn.Module, Iterable[Batch]], dict[str, float]]


@dataclass(frozen=True)
class Objective:
    loss: Loss
    validation_metrics: Evaluator
    test_metrics: Evaluator


def _checkpoints(values: Iterable[int]) -> tuple[int, ...]:
    checkpoints = tuple(map(int, values))
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("checkpoints must be non-empty, unique, and increasing")
    if checkpoints[0] <= 0:
        raise ValueError("checkpoints must be positive")
    return checkpoints


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


def train_events(
    task: TrainTask,
    model: nn.Module,
    *,
    lr: float,
    checkpoints: Iterable[int],
    loss: Loss,
) -> Iterator[fts.Event]:
    checkpoints = _checkpoints(checkpoints)

    optimizers = _adam_optimizers(model, lr=lr)
    batches = iter(task.train_batches(checkpoints[-1]))
    step = 0
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

            step += 1
            examples_seen += len(labels)
            if examples_seen >= checkpoint:
                break

        train_seconds += perf_counter() - segment_started
        assert examples_seen == checkpoint, (
            f"training stream reached {examples_seen} examples; "
            f"expected checkpoint {checkpoint}"
        )
        yield fts.Event(
            step=step,
            train_size=examples_seen,
            train_seconds=train_seconds,
            model=model,
        )

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
    squared_error: torch.Tensor | None = None
    samples = 0
    try:
        with torch.inference_mode():
            for model_inputs, labels in batches:
                errors = _predict(model, model_inputs, labels) - labels
                batch_error = errors.square().sum()
                squared_error = (
                    batch_error
                    if squared_error is None
                    else squared_error + batch_error
                )
                samples += labels.numel()
    finally:
        model.train(was_training)

    if squared_error is None:
        raise ValueError("evaluation data must not be empty")
    return {"rmse": (squared_error / samples).sqrt().item()}


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


def evaluate_on(
    name: str,
    batches: BatchFactory,
    evaluate: Evaluator,
) -> Callable[[Event], dict[str, float]]:
    elapsed_seconds = 0.0

    def augment(event: Event) -> dict[str, float]:
        nonlocal elapsed_seconds
        started = perf_counter()
        metrics = evaluate(event["model"], batches())
        elapsed_seconds += perf_counter() - started
        return {
            **{f"{name}_{key}": value for key, value in metrics.items()},
            f"{name}_seconds": elapsed_seconds,
        }

    return augment


def fit_and_evaluate(
    task: Task,
    model: nn.Module,
    *,
    objective: Objective,
    lr: float,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    events = fts.pipe(
        train_events(
            task,
            model,
            lr=lr,
            checkpoints=checkpoints,
            loss=objective.loss,
        ),
        fts.augment(
            evaluate_on("val", task.val_batches, objective.validation_metrics)
        ),
        fts.augment(evaluate_on("test", task.test_batches, objective.test_metrics)),
    )
    return fts.collect_pd(events)


def at_train_sizes(
    train_sizes: Iterable[int],
) -> Callable[[Iterable[Event]], Iterator[Event]]:
    selected = frozenset(train_sizes)

    def transform(events: Iterable[Event]) -> Iterator[Event]:
        for event in events:
            if event["train_size"] in selected:
                yield event

    return transform


def _fit_metric_trajectory(
    task: Task,
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
        train_events(
            task,
            model,
            lr=lr,
            checkpoints=checkpoints,
            loss=loss,
        ),
        at_train_sizes(selected),
        fts.augment(evaluate_on(metric_name, batches, evaluate)),
    )
    return fts.collect_pd(events)


def fit_validation_trajectory(
    task: Task,
    model: nn.Module,
    *,
    objective: Objective,
    lr: float,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    return _fit_metric_trajectory(
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


def fit_test_trajectory(
    task: Task,
    model: nn.Module,
    *,
    objective: Objective,
    lr: float,
    checkpoints: Iterable[int],
    test_checkpoints: Iterable[int],
) -> pd.DataFrame:
    return _fit_metric_trajectory(
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
