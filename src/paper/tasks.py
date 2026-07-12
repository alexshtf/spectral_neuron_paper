from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from paper.targets import ArrayTarget


BatchIterator = Iterator[tuple[torch.Tensor, torch.Tensor]]
BatchStream = Callable[[np.random.Generator], BatchIterator]


@dataclass
class Task:
    input_dim: int
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    train_batches: BatchStream


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
    seed: int,
    noise_std: float = 0.0,
) -> Task:
    rng = np.random.default_rng(seed)

    x_val_np = rng.uniform(lower, upper, size=(val_size, input_dim))

    x_val = _to_x_tensor(x_val_np)
    y_val = _to_y_tensor(target(x_val_np))
    x_test = _to_x_tensor(x_test_np)
    y_test = _to_y_tensor(target(x_test_np))

    def train_batches(batch_rng: np.random.Generator) -> BatchIterator:
        while True:
            x_np = batch_rng.uniform(lower, upper, size=(batch_size, input_dim))
            y_np = np.asarray(target(x_np), dtype=float)
            if noise_std > 0:
                y_np = y_np + batch_rng.normal(0.0, noise_std, size=y_np.shape)
            yield _to_x_tensor(x_np), _to_y_tensor(y_np)

    return Task(
        input_dim=input_dim,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_batches=train_batches,
    )


def make_univariate_task(
    target: ArrayTarget,
    *,
    lower: float,
    upper: float,
    batch_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    noise_std: float = 0.0,
) -> Task:
    return _make_task(
        target,
        input_dim=1,
        x_test_np=np.linspace(lower, upper, test_size).reshape(-1, 1),
        lower=lower,
        upper=upper,
        batch_size=batch_size,
        val_size=val_size,
        seed=seed,
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
    seed: int,
    noise_std: float = 0.0,
) -> Task:
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
        seed=seed,
        noise_std=noise_std,
    )
