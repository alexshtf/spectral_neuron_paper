import numpy as np
import pytest

from paper.targets import TargetSpec, make_target, random_monotone_2d


def test_target_seed_reproducible():
    spec = TargetSpec(kind="general", complexity=6, seed=123)
    x = np.linspace(spec.lower, spec.upper, 100)

    y1 = make_target(spec)(x)
    y2 = make_target(spec)(x)

    np.testing.assert_allclose(y1, y2)


def test_monotone_target_is_monotone_on_grid():
    spec = TargetSpec(kind="monotone", complexity=8, seed=123)
    x = np.linspace(spec.lower, spec.upper, 500)
    y = make_target(spec)(x)

    assert np.all(np.diff(y) >= -1e-10)


def test_monotone_2d_target_is_reproducible():
    points = np.array([[-2.0, -1.0], [0.0, 0.5], [3.0, 2.0]])
    y1 = random_monotone_2d(5, rng=np.random.default_rng(123))(points)
    y2 = random_monotone_2d(5, rng=np.random.default_rng(123))(points)

    np.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize("complexity", [2, 5, 10])
def test_monotone_2d_target_is_monotone_in_second_coordinate(complexity):
    target = random_monotone_2d(complexity, rng=np.random.default_rng(123))
    x1, x2 = np.meshgrid(
        np.linspace(-4.0, 4.0, 51), np.linspace(-4.0, 4.0, 101), indexing="ij"
    )
    values = target(np.stack((x1, x2), axis=-1))

    assert values.shape == x1.shape
    assert np.all(np.diff(values, axis=1) >= -1e-10)


def test_monotone_2d_target_is_nonlinear_in_second_coordinate():
    target = random_monotone_2d(10, rng=np.random.default_rng(123))
    x2 = np.linspace(-4.0, 4.0, 101)
    values = target(np.column_stack((np.zeros_like(x2), x2)))
    chord = np.linspace(values[0], values[-1], values.size)

    assert np.max(np.abs(values - chord)) > 0.1


def test_convex_target_is_convex_on_grid():
    spec = TargetSpec(kind="convex", complexity=8, seed=123)
    x = np.linspace(spec.lower, spec.upper, 500)
    y = make_target(spec)(x)

    assert np.all(np.diff(y, n=2) >= -1e-10)
