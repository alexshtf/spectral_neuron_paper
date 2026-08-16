import math
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

BINARY_METRIC_LABELS = {
    "logloss": "test log loss",
    "brier": "test Brier score",
}
TRAIN_SIZE_LABEL = "training samples processed by optimizer"

type FigureContainer = Figure | SubFigure


@dataclass(frozen=True)
class CurveStyle:
    label: str | int
    color: str | tuple[float, float, float]
    marker: str
    dashes: str | tuple[int, ...] = ""


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


def _dimension_curve_styles(dimensions: list[int]) -> tuple[CurveStyle, ...]:
    return tuple(
        CurveStyle(
            dimension,
            color,
            _DIMENSION_MARKERS[index % len(_DIMENSION_MARKERS)],
        )
        for index, (dimension, color) in enumerate(
            zip(
                dimensions,
                sns.color_palette("colorblind", len(dimensions)),
                strict=True,
            )
        )
    )


def _finish_scaling_grid(
    grid: sns.FacetGrid,
    *,
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    facet_title: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    if facet_title is not None:
        grid.set_titles(facet_title)
    grid.set_axis_labels(x_label, y_label)
    for ax in grid.axes.flat:
        ax.set_xscale("log", base=2)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)
    if grid.legend is not None:
        grid.legend.set_title(legend_title)
    grid.figure.suptitle(title, y=1.02)
    return grid.figure


def _summary_metric_columns(
    summary: pd.DataFrame, metric: str
) -> tuple[str, str, str]:
    if metric not in BINARY_METRIC_LABELS:
        raise ValueError(
            f"metric must be one of {sorted(BINARY_METRIC_LABELS)}; got {metric!r}"
        )

    columns = tuple(
        f"{statistic}_test_{metric}" for statistic in ("median", "q25", "q75")
    )
    missing = {"train_size", "n", *columns}.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    return columns


def _draw_summary_curves(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    by: str,
    styles: Sequence[CurveStyle],
    q25: str,
    q75: str,
    **_: object,
) -> None:
    labels = [style.label for style in styles]
    colors = {style.label: style.color for style in styles}
    ax = plt.gca()
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=by,
        style=by,
        hue_order=labels,
        style_order=labels,
        palette=colors,
        markers={style.label: style.marker for style in styles},
        dashes={style.label: style.dashes for style in styles},
        estimator=None,
        errorbar=None,
        legend=False,
        linewidth=2,
        ax=ax,
    )
    for style in styles:
        curve = data.loc[data[by] == style.label].sort_values(x)
        if curve.empty:
            continue
        repeated = curve["n"] > 1
        ax.fill_between(
            curve[x],
            curve[q25].where(repeated),
            curve[q75].where(repeated),
            color=style.color,
            alpha=0.15,
        )


def _legend_handle(style: CurveStyle) -> Line2D:
    handle = Line2D(
        [],
        [],
        color=style.color,
        marker=style.marker,
        linewidth=2,
    )
    handle.set_dashes(style.dashes)
    return handle


def _summary_curve_grid(
    summary: pd.DataFrame,
    *,
    metric: str,
    by: str,
    styles: Sequence[CurveStyle],
    col: str | None = None,
    col_order: Sequence[object] | None = None,
    height: float = 3.5,
) -> sns.FacetGrid:
    median, q25, q75 = _summary_metric_columns(summary, metric)
    required = {by}
    if col is not None:
        required.add(col)
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    labels = [style.label for style in styles]
    facet = {}
    if col is not None:
        facets = (
            list(col_order)
            if col_order is not None
            else summary[col].drop_duplicates().tolist()
        )
        facet = {
            "col": col,
            "col_order": facets,
            "col_wrap": min(2, len(facets)),
        }

    # Keep FacetGrid's early layout independent of internal column-name lengths.
    plot_data = summary.rename(columns={"train_size": "_x", median: "_y"})
    grid = sns.FacetGrid(
        data=plot_data,
        height=height,
        aspect=1.25,
        sharex=True,
        sharey=True,
        dropna=False,
        **facet,
    )
    grid.map_dataframe(
        _draw_summary_curves,
        x="_x",
        y="_y",
        by=by,
        styles=styles,
        q25=q25,
        q75=q75,
    )
    grid.add_legend(
        legend_data={style.label: _legend_handle(style) for style in styles},
        label_order=labels,
        title=by,
        adjust_subtitles=True,
    )
    return grid
