import numpy as np
import pytest

from paper.targets import TargetSpec, make_bivariate_target
from paper.tasks import make_bivariate_task


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
        )
