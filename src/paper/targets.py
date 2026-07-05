from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.interpolate as interp


type ArrayTarget = Callable[[np.ndarray], np.ndarray]
type TargetKind = Literal["general", "monotone", "convex"]


@dataclass(frozen=True)
class TargetSpec:
    kind: TargetKind
    complexity: int
    seed: int
    lower: float = -4.0
    upper: float = 4.0


def _eval_1d(spline: Callable[[np.ndarray], np.ndarray], x_np: np.ndarray) -> np.ndarray:
    x = np.asarray(x_np, dtype=float)
    y = np.asarray(spline(x.reshape(-1)), dtype=float)
    if x.ndim > 0 and x.shape[-1] == 1:
        return y.reshape(x.shape[:-1])
    return y.reshape(x.shape)


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return values - values.mean()
    return (values - values.mean()) / std


def random_general_1d(
    complexity: int,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
    rng: np.random.Generator | None = None,
) -> ArrayTarget:
    if rng is None:
        rng = np.random.default_rng(42)

    knots = np.linspace(lower, upper, complexity)
    values = rng.normal(0.0, 1.0, size=complexity)
    spline = interp.CubicSpline(knots, values, bc_type="natural", extrapolate=True)
    return lambda x_np: _eval_1d(spline, x_np)


def random_monotone_1d(
    complexity: int,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
    rng: np.random.Generator | None = None,
) -> ArrayTarget:
    if rng is None:
        rng = np.random.default_rng(42)

    knots = np.linspace(lower, upper, complexity)
    increments = np.exp(rng.normal(0.0, 1.0, size=complexity))
    values = _standardize(np.cumsum(increments))
    spline = interp.PchipInterpolator(knots, values, extrapolate=True)
    return lambda x_np: _eval_1d(spline, x_np)


def random_convex_1d(
    complexity: int,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
    rng: np.random.Generator | None = None,
) -> ArrayTarget:
    if rng is None:
        rng = np.random.default_rng(42)

    knots = np.linspace(lower, upper, complexity)
    dx = np.diff(knots)
    slopes = np.cumsum(np.exp(rng.normal(0.0, 0.5, size=complexity - 1)))
    values = np.r_[0.0, np.cumsum(slopes * dx)]
    values = _standardize(values)
    spline = interp.make_interp_spline(knots, values, k=1)
    return lambda x_np: _eval_1d(spline, x_np)


def make_target(spec: TargetSpec) -> ArrayTarget:
    rng = np.random.default_rng(spec.seed)

    match spec.kind:
        case "general":
            return random_general_1d(
                spec.complexity, lower=spec.lower, upper=spec.upper, rng=rng
            )
        case "monotone":
            return random_monotone_1d(
                spec.complexity, lower=spec.lower, upper=spec.upper, rng=rng
            )
        case "convex":
            return random_convex_1d(
                spec.complexity, lower=spec.lower, upper=spec.upper, rng=rng
            )
        case _:
            raise ValueError(spec.kind)
