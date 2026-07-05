import numpy as np

from paper.targets import TargetSpec, make_target


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


def test_convex_target_is_convex_on_grid():
    spec = TargetSpec(kind="convex", complexity=8, seed=123)
    x = np.linspace(spec.lower, spec.upper, 500)
    y = make_target(spec)(x)

    assert np.all(np.diff(y, n=2) >= -1e-10)
