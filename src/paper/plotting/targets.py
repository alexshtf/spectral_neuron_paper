import numpy as np
from matplotlib.figure import Figure

from paper.targets import TargetSpec, make_bivariate_target, make_target

from ._common import _subplot_grid


def plot_target_gallery(specs: list[TargetSpec]) -> Figure:
    fig, axes = _subplot_grid(len(specs), cell_width=4, cell_height=3)

    for spec, ax in zip(specs, axes):
        target = make_target(spec)
        xs = np.linspace(spec.lower, spec.upper, 1000)
        ax.plot(xs, target(xs))
        ax.set_title(f"{spec.kind}, complexity={spec.complexity}, seed={spec.seed}")

    return fig


def plot_bivariate_target_gallery(
    specs: list[TargetSpec], *, resolution: int = 200
) -> Figure:
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2; got {resolution}")

    fig, axes = _subplot_grid(len(specs), cell_width=4.5, cell_height=4)

    for spec, ax in zip(specs, axes):
        target = make_bivariate_target(spec)
        grid = np.linspace(spec.lower, spec.upper, resolution)
        x1, x2 = np.meshgrid(grid, grid, indexing="ij")
        values = target(np.stack((x1, x2), axis=-1))
        contour = ax.contourf(x1, x2, values, levels=20)
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"{spec.kind}, complexity={spec.complexity}, seed={spec.seed}")
        ax.set(xlabel="$x_1$", ylabel="$x_2$")
        ax.set_aspect("equal")

    return fig
