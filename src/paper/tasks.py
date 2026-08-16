from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from paper.targets import ArrayTarget


type ModelInputs = tuple[torch.Tensor, ...]
type Batch = tuple[ModelInputs, torch.Tensor]


class TrainTask(Protocol):
    def train_batches(self, max_examples: int) -> Iterator[Batch]: ...


class Task(TrainTask, Protocol):
    def val_batches(self) -> Iterator[Batch]: ...

    def test_batches(self) -> Iterator[Batch]: ...


@dataclass(frozen=True)
class SyntheticTask:
    target: ArrayTarget
    input_dim: int
    lower: float
    upper: float
    batch_size: int
    train_seed: int
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    noise_std: float = 0.0

    def train_batches(self, max_examples: int) -> Iterator[Batch]:
        rng = np.random.default_rng(self.train_seed)
        for batch_start in range(0, max_examples, self.batch_size):
            size = min(self.batch_size, max_examples - batch_start)
            x_np = rng.uniform(
                self.lower,
                self.upper,
                size=(size, self.input_dim),
            )
            y_np = np.asarray(self.target(x_np), dtype=float)
            y_np = y_np + rng.normal(0.0, self.noise_std, size=y_np.shape)
            yield ((_to_x_tensor(x_np),), _to_y_tensor(y_np))

    def val_batches(self) -> Iterator[Batch]:
        yield ((self.x_val,), self.y_val)

    def test_batches(self) -> Iterator[Batch]:
        yield ((self.x_test,), self.y_test)


def _to_x_tensor(x_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x_np, dtype=torch.get_default_dtype())


def _to_y_tensor(y_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(y_np, dtype=torch.get_default_dtype()).reshape(-1)


def _make_task(
    target: ArrayTarget,
    *,
    input_dim: int,
    x_test_np: np.ndarray,
    lower: float,
    upper: float,
    batch_size: int,
    val_size: int,
    validation_seed: int,
    train_seed: int,
    noise_std: float = 0.0,
) -> SyntheticTask:
    rng = np.random.default_rng(validation_seed)
    x_val_np = rng.uniform(lower, upper, size=(val_size, input_dim))

    x_val = _to_x_tensor(x_val_np)
    y_val = _to_y_tensor(target(x_val_np))
    x_test = _to_x_tensor(x_test_np)
    y_test = _to_y_tensor(target(x_test_np))

    return SyntheticTask(
        target=target,
        input_dim=input_dim,
        lower=lower,
        upper=upper,
        batch_size=batch_size,
        train_seed=train_seed,
        noise_std=noise_std,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def make_univariate_task(
    target: ArrayTarget,
    *,
    lower: float,
    upper: float,
    batch_size: int,
    val_size: int,
    test_size: int,
    validation_seed: int,
    train_seed: int,
    noise_std: float = 0.0,
) -> SyntheticTask:
    return _make_task(
        target,
        input_dim=1,
        x_test_np=np.linspace(lower, upper, test_size).reshape(-1, 1),
        lower=lower,
        upper=upper,
        batch_size=batch_size,
        val_size=val_size,
        validation_seed=validation_seed,
        train_seed=train_seed,
        noise_std=noise_std,
    )


def make_bivariate_task(
    target: ArrayTarget,
    *,
    lower: float,
    upper: float,
    batch_size: int,
    val_size: int,
    test_size: int,
    validation_seed: int,
    train_seed: int,
    noise_std: float = 0.0,
) -> SyntheticTask:
    if test_size < 1:
        raise ValueError(f"test_size must be positive; got {test_size}")

    grid_size = round(np.sqrt(test_size))
    if grid_size * grid_size != test_size:
        raise ValueError(f"test_size must be a perfect square; got {test_size}")

    grid = np.linspace(lower, upper, grid_size)
    x1, x2 = np.meshgrid(grid, grid, indexing="ij")
    return _make_task(
        target,
        input_dim=2,
        x_test_np=np.column_stack((x1.ravel(), x2.ravel())),
        lower=lower,
        upper=upper,
        batch_size=batch_size,
        val_size=val_size,
        validation_seed=validation_seed,
        train_seed=train_seed,
        noise_std=noise_std,
    )
