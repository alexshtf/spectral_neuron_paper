import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure

BINARY_METRIC_LABELS = {
    "logloss": "test log loss",
    "brier": "test Brier score",
}
TRAIN_SIZE_LABEL = "training samples processed by optimizer"

type FigureContainer = Figure | SubFigure


def _subplot_matrix(
    n_rows: int,
    n_cols: int,
    *,
    cell_width: float,
    cell_height: float,
    container: FigureContainer | None = None,
):
    if n_rows == 0 or n_cols == 0:
        raise ValueError("at least one item is required")

    if container is None:
        container = plt.figure(
            figsize=(cell_width * n_cols, cell_height * n_rows),
            layout="constrained",
        )
    axs = container.subplots(n_rows, n_cols, squeeze=False)
    return container, axs


def _subplot_grid(
    n_items: int,
    *,
    cell_width: float,
    cell_height: float,
    container: FigureContainer | None = None,
):
    if n_items == 0:
        raise ValueError("at least one item is required")

    n_cols = int(math.ceil(math.sqrt(n_items)))
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axs = _subplot_matrix(
        n_rows,
        n_cols,
        cell_width=cell_width,
        cell_height=cell_height,
        container=container,
    )
    axes = axs.ravel()

    for ax in axes[n_items:]:
        ax.set_visible(False)

    return fig, axes


_DIMENSION_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


def _dimension_styles(
    dimensions: list[int],
) -> tuple[dict[int, tuple[float, float, float]], dict[int, str]]:
    palette = dict(
        zip(
            dimensions,
            sns.color_palette("colorblind", len(dimensions)),
            strict=True,
        )
    )
    markers = {
        dim: _DIMENSION_MARKERS[i % len(_DIMENSION_MARKERS)]
        for i, dim in enumerate(dimensions)
    }
    return palette, markers


def _finish_scaling_grid(
    grid: sns.FacetGrid,
    *,
    title: str,
    x_label: str,
    y_label: str,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    grid.set_axis_labels(x_label, y_label)
    for ax in grid.axes.flat:
        ax.set_xscale("log", base=2)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)
    grid.figure.suptitle(title, y=1.02)
    return grid.figure


def _binary_relplot(
    results: pd.DataFrame,
    *,
    value: str,
    metric: str,
    title: str,
    x_label: str,
    by: str,
    hue_order: list,
    palette,
    markers,
    dashes,
    legend_title: str,
    col: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    facet = (
        {"col": col, "col_wrap": min(2, results[col].nunique())}
        if col is not None
        else {}
    )
    grid = sns.relplot(
        data=results,
        x="train_size",
        y=value,
        hue=by,
        style=by,
        hue_order=hue_order,
        style_order=hue_order,
        palette=palette,
        markers=markers,
        dashes=dashes,
        kind="line",
        estimator=np.median,
        errorbar=("pi", 50),
        err_kws={"alpha": 0.15},
        linewidth=2,
        height=3.5,
        aspect=1.25,
        facet_kws={"sharex": True, "sharey": True},
        **facet,
    )
    if col is not None:
        grid.set_titles("dim={col_name}")
    if grid.legend is not None:
        grid.legend.set_title(legend_title)
    return _finish_scaling_grid(
        grid,
        title=title,
        x_label=x_label,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        xlim=xlim,
    )
