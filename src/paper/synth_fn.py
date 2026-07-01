import numpy as np
import scipy.interpolate as interp


def random_func(
    m: int,
    sigma: float = 1.0,
    lower=-4,
    upper=4,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng(42)

    ts = np.linspace(lower, upper, m)
    ys = rng.normal(0.0, sigma, size=m)
    return interp.CubicSpline(ts, ys, bc_type="natural", extrapolate=True)


def random_inc_func(
    m: int,
    sigma: float = 1.0,
    lower=-4,
    upper=4,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng(42)

    ts = np.r_[lower, lower, lower, np.linspace(lower, upper, m), upper, upper, upper]
    cs = np.cumsum(np.exp(rng.normal(0.0, sigma, size=m + 3)))
    return interp.BSpline(ts, cs, k=3, extrapolate=True)
