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


def random_general_2d(
    complexity: int,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
    rng: np.random.Generator | None = None,
) -> ArrayTarget:
    if rng is None:
        rng = np.random.default_rng(42)

    knots = np.linspace(lower, upper, complexity)
    values = rng.normal(0.0, 1.0, size=(complexity, complexity))
    return interp.RegularGridInterpolator((knots, knots), values, method='cubic')


def random_monotone_2d(
    complexity: int,
    *,
    lower: float = -4.0,
    upper: float = 4.0,
    rng: np.random.Generator | None = None,
) -> ArrayTarget:
    """Build a random target monotone in its second coordinate.

    Let κ₀ < ⋯ < κₘ₋₁ be the shared grid, and let pᵢ be the PCHIP
    interpolant through a positive affine standardization of row i of

        Yᵢⱼ = Σₗ₌₀ʲ exp(Zᵢₗ),    Zᵢₗ ∼ N(0, 1).

    Each pᵢ is nondecreasing. For x₁ ∈ [κᵢ, κᵢ₊₁], set

        t = (x₁ − κᵢ) / (κᵢ₊₁ − κᵢ),
        w(t) = 6t⁵ − 15t⁴ + 10t³,
        f(x₁, x₂) = (1 − w(t))pᵢ(x₂) + w(t)pᵢ₊₁(x₂).

    The quintic is the unique lowest-degree polynomial satisfying

        w(0) = 0,  w(1) = 1,
        w′(0) = w′(1) = w″(0) = w″(1) = 0.

    Its endpoint values select the two rows, while its vanishing first and
    second derivatives make adjacent x₁ cells meet C²-smoothly. Its range is
    also contained in [0, 1], which makes the blend convex.

    Since 0 ≤ w(t) ≤ 1, for x₂′ ≥ x₂,

        f(x₁, x₂′) − f(x₁, x₂)
        = (1 − w)[pᵢ(x₂′) − pᵢ(x₂)]
          + w[pᵢ₊₁(x₂′) − pᵢ₊₁(x₂)] ≥ 0.

    Thus f is monotone in x₂ on the target domain. This resembles a
    tensor-product spline: PCHIP supplies the x₂ curves, and the same local
    x₁ blending polynomial is used in every cell. It is deliberately not a
    tensor-product spline interpolant: no spline is fit across x₁ and no
    bivariate spline coefficients are solved for. The local convex blend
    preserves the row-wise PCHIP monotonicity; PCHIP makes f C¹ in x₂.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if complexity < 2:
        raise ValueError("complexity must be at least 2")
    if not np.isfinite((lower, upper)).all() or lower >= upper:
        raise ValueError("expected finite lower < upper")

    knots = np.linspace(lower, upper, complexity)
    values = np.cumsum(
        np.exp(rng.normal(0.0, 1.0, size=(complexity, complexity))), axis=1
    )
    rows = interp.PchipInterpolator(
        knots, _standardize(values).T, axis=0, extrapolate=False
    )

    def target(x_np: np.ndarray) -> np.ndarray:
        x = np.asarray(x_np, dtype=float)
        if x.ndim == 0 or x.shape[-1] != 2:
            raise ValueError("expected input shape (..., 2)")

        if not np.isfinite(x).all() or np.any((x < lower) | (x > upper)):
            raise ValueError(f"expected inputs in [{lower}, {upper}]")

        x1 = x[..., 0]
        left = np.clip(
            np.searchsorted(knots, x1, side="right") - 1,
            0,
            complexity - 2,
        )
        t = (x1 - knots[left]) / (knots[left + 1] - knots[left])
        # Quintic smootherstep: a C2 convex blend between the adjacent rows.
        weight = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
        row_values = rows(x[..., 1])
        left_values = np.take_along_axis(
            row_values, left[..., None], axis=-1
        )[..., 0]
        right_values = np.take_along_axis(
            row_values, (left + 1)[..., None], axis=-1
        )[..., 0]
        return np.asarray((1.0 - weight) * left_values + weight * right_values)

    return target


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


def make_bivariate_target(spec: TargetSpec) -> ArrayTarget:
    rng = np.random.default_rng(spec.seed)

    match spec.kind:
        case "general":
            return random_general_2d(
                spec.complexity, lower=spec.lower, upper=spec.upper, rng=rng
            )
        case "monotone":
            return random_monotone_2d(
                spec.complexity, lower=spec.lower, upper=spec.upper, rng=rng
            )
        case _:
            raise ValueError(f"unsupported bivariate target kind: {spec.kind}")
