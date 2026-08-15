import numpy as np
import pytest
import torch

from paper.targets import TargetSpec, make_bivariate_target
from paper.tasks import make_bivariate_task, make_univariate_task


def test_training_inputs_match_across_noise_levels():
    def make_task(noise_std):
        return make_univariate_task(
            lambda x: x[..., 0],
            lower=-1.0,
            upper=1.0,
            batch_size=4,
            val_size=3,
            test_size=5,
            seed=0,
            train_seed=1,
            noise_std=noise_std,
        )

    noiseless = make_task(0.0).train_batches(12)
    noisy = make_task(0.1).train_batches(12)

    for _ in range(3):
        (x_noiseless,), _ = next(noiseless)
        (x_noisy,), _ = next(noisy)
        torch.testing.assert_close(x_noiseless, x_noisy)


def test_bivariate_task_uses_a_square_tensor_product_test_grid():
    target = make_bivariate_target(TargetSpec(kind="general", complexity=5, seed=1))
    task = make_bivariate_task(
        target,
        lower=-1.0,
        upper=1.0,
        batch_size=4,
        val_size=3,
        test_size=9,
        seed=2,
        train_seed=3,
    )

    grid = np.linspace(-1.0, 1.0, 3)
    expected = np.array([(x1, x2) for x1 in grid for x2 in grid])
    assert task.input_dim == 2
    assert task.x_val.shape == (3, 2)
    np.testing.assert_allclose(task.x_test.numpy(), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        task.y_test.numpy(), target(expected), rtol=1e-6, atol=1e-6
    )


def test_bivariate_task_requires_a_perfect_square_test_size():
    with pytest.raises(ValueError, match="perfect square"):
        make_bivariate_task(
            lambda x: x[..., 0],
            lower=-1.0,
            upper=1.0,
            batch_size=4,
            val_size=3,
            test_size=8,
            seed=2,
            train_seed=3,
        )
