import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.targets import TargetSpec, make_target


def _subplot_grid(n_items: int, *, cell_width: float, cell_height: float):
    if n_items == 0:
        raise ValueError("at least one item is required")

    n_cols = int(math.ceil(math.sqrt(n_items)))
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_width * n_cols, cell_height * n_rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axs.ravel()

    for ax in axes[n_items:]:
        ax.set_visible(False)

    return fig, axes


def plot_scaling(summary: pd.DataFrame):
    complexities = sorted(summary["complexity"].unique())
    fig, axes = _subplot_grid(
        len(complexities), cell_width=4.5, cell_height=3.5
    )

    for complexity, ax in zip(complexities, axes):
        sub = summary.loc[summary["complexity"] == complexity]
        for (model, dim), group in sub.groupby(["model", "dim"], sort=True):
            group = group.sort_values("budget")
            label = f"{model}, dim={dim}"
            ax.plot(group["budget"], group["median_test_rmse"], marker="o", label=label)
            ax.fill_between(
                group["budget"],
                group["q25_test_rmse"],
                group["q75_test_rmse"],
                alpha=0.15,
            )
        ax.set_title(f"complexity={complexity}")
        ax.set_xlabel("budget")
        ax.set_ylabel("test RMSE")
        ax.set_xscale("log")
        if (sub["median_test_rmse"] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")

    return fig


def plot_target_gallery(specs: list[TargetSpec]):
    fig, axes = _subplot_grid(len(specs), cell_width=4, cell_height=3)

    for spec, ax in zip(specs, axes):
        target = make_target(spec)
        xs = np.linspace(spec.lower, spec.upper, 1000)
        ax.plot(xs, target(xs))
        ax.set_title(f"{spec.kind}, complexity={spec.complexity}, seed={spec.seed}")

    return fig
