from collections.abc import Callable, Iterable
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure, SubFigure

from ._common import (
    FigureContainer,
    _grid_shape,
    _subplot_grid,
    _subplot_matrix,
)

SCALING_COLUMNS = {
    "target_kind",
    "complexity",
    "noise_std",
    "model",
    "dim",
    "train_size",
    "median_test_rmse",
    "q25_test_rmse",
    "q75_test_rmse",
}

MODEL_LINESTYLES = {
    "unconstrained": "-",
    "monotone": "--",
}
SYNTHETIC_TRAIN_SIZE_LABEL = "training-sample budget"

_MONOTONE_MODEL_PAIR = ("unconstrained", "monotone")


def _scaling_dim_colors(dims: list[int]) -> dict[int, str]:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {dim: color_cycle[i % len(color_cycle)] for i, dim in enumerate(dims)}


def _scaling_model_linestyles(models: list[str]) -> dict[str, str]:
    fallback_styles = ("-", "--", ":", "-.")
    return {
        model: MODEL_LINESTYLES.get(model, fallback_styles[i % len(fallback_styles)])
        for i, model in enumerate(models)
    }


def _ordered_models(models: Iterable[str]) -> list[str]:
    model_set = set(models)
    preferred = [model for model in MODEL_LINESTYLES if model in model_set]
    remaining = sorted(model for model in model_set if model not in MODEL_LINESTYLES)
    return preferred + remaining


def _scaling_label(model: str, dim: int) -> str:
    return f"dim={dim}, {model}"


def _plot_curve(
    ax,
    group: pd.DataFrame,
    *,
    label: str,
    color: str | None = None,
    linestyle: str | None = None,
) -> None:
    group = group.sort_values("train_size")
    x = group["train_size"].to_numpy()
    median = group["median_test_rmse"].to_numpy()
    q25 = group["q25_test_rmse"].to_numpy()
    q75 = group["q75_test_rmse"].to_numpy()
    plot_kwargs = {"marker": "o", "label": label}
    fill_kwargs = {"alpha": 0.15}
    if color is not None:
        plot_kwargs["color"] = color
        fill_kwargs["color"] = color
        fill_kwargs["linewidth"] = 0
    if linestyle is not None:
        plot_kwargs["linestyle"] = linestyle
    ax.plot(x, median, **plot_kwargs)
    ax.fill_between(x, q25, q75, **fill_kwargs)


def _finish_scaling_axis(ax, data: pd.DataFrame) -> None:
    ax.set_xlabel(SYNTHETIC_TRAIN_SIZE_LABEL)
    ax.set_ylabel("test RMSE")
    ax.set_xscale("log")
    if (data["median_test_rmse"] > 0).all():
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)


def _add_shared_legend(fig, axes, *, borderaxespad: float = 0.5) -> None:
    handles = []
    labels = []

    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
            borderaxespad=borderaxespad,
        )


def _check_scaling_columns(summary: pd.DataFrame) -> None:
    missing = SCALING_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")


def _check_target_kind(summary: pd.DataFrame, expected: str) -> None:
    target_kinds = sorted(summary["target_kind"].dropna().unique())
    if target_kinds != [expected]:
        raise ValueError(f"expected target_kind={expected!r}, got {target_kinds}")


def _noise_stds(summary: pd.DataFrame) -> list[float]:
    if summary["noise_std"].isna().any():
        raise ValueError("noise_std must not contain missing values")
    return sorted(summary["noise_std"].unique().tolist())


def _noise_title(noise_std: float) -> str:
    if noise_std == 0:
        return "Noiseless training (σ = 0)"
    return f"Noisy training (σ = {noise_std:g})"


def _grid_figure_size(summary: pd.DataFrame) -> tuple[float, float]:
    n_rows, n_cols = _grid_shape(summary["complexity"].nunique())
    return 4.5 * n_cols, 3.5 * n_rows


def _pairwise_figure_size(summary: pd.DataFrame) -> tuple[float, float]:
    return (
        4.0 * summary["dim"].nunique(),
        3.0 * summary["complexity"].nunique(),
    )


def _plot_scaling_grid(
    summary: pd.DataFrame, *, container: FigureContainer | None = None
) -> FigureContainer:
    complexities = sorted(summary["complexity"].unique())
    dims = sorted(summary["dim"].unique())
    models = _ordered_models(summary["model"].unique())
    dim_colors = _scaling_dim_colors(dims)
    model_linestyles = _scaling_model_linestyles(models)
    fig, axes = _subplot_grid(
        len(complexities),
        cell_width=4.5,
        cell_height=3.5,
        container=container,
    )

    for complexity, ax in zip(complexities, axes):
        sub = summary.loc[summary["complexity"] == complexity]
        for dim in dims:
            for model in models:
                group = sub.loc[(sub["dim"] == dim) & (sub["model"] == model)]
                if group.empty:
                    continue

                color = dim_colors[dim]
                linestyle = model_linestyles[model]
                label = _scaling_label(model, dim)
                _plot_curve(
                    ax,
                    group,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )
        ax.set_title(f"complexity={complexity}")
        _finish_scaling_axis(ax, sub)

    _add_shared_legend(
        fig,
        axes,
        borderaxespad=2 if isinstance(fig, SubFigure) else 0.5,
    )

    return fig


def _plot_pairwise_scaling(
    summary: pd.DataFrame,
    *,
    container: FigureContainer | None = None,
) -> FigureContainer:
    paired = summary.loc[summary["model"].isin(_MONOTONE_MODEL_PAIR)].copy()
    complexities = sorted(paired["complexity"].unique())
    dims = sorted(paired["dim"].unique())
    fig, axs = _subplot_matrix(
        len(complexities),
        len(dims),
        cell_width=4.0,
        cell_height=3.0,
        container=container,
    )

    for row, complexity in enumerate(complexities):
        for col, dim in enumerate(dims):
            ax = axs[row, col]
            sub = paired.loc[
                (paired["complexity"] == complexity) & (paired["dim"] == dim)
            ]
            if sub.empty:
                ax.set_visible(False)
                continue

            for model in _MONOTONE_MODEL_PAIR:
                group = sub.loc[sub["model"] == model]
                if not group.empty:
                    _plot_curve(
                        ax,
                        group,
                        label=model,
                        linestyle=MODEL_LINESTYLES[model],
                    )

            ax.set_title(f"complexity={complexity}, dim={dim}")
            _finish_scaling_axis(ax, sub)

    _add_shared_legend(
        fig,
        axs.ravel(),
        borderaxespad=2 if isinstance(fig, SubFigure) else 0.5,
    )

    return fig


def _plot_noise_subfigures(
    summary: pd.DataFrame,
    plotter: Callable[..., FigureContainer],
    figure_size: tuple[float, float],
) -> Figure:
    noise_stds = _noise_stds(summary)
    width, height = figure_size
    fig = plt.figure(
        figsize=(width, height * len(noise_stds)),
        layout="constrained",
    )
    fig.get_layout_engine().set(h_pad=0.12)
    subfigures = fig.subfigures(len(noise_stds), 1, squeeze=False).ravel()

    for noise_std, subfigure in zip(noise_stds, subfigures):
        noise_summary = summary.loc[summary["noise_std"] == noise_std]
        subfigure.suptitle(_noise_title(noise_std), y=1, fontsize="x-large")
        plotter(noise_summary, container=subfigure)

    return fig


def plot_general_scaling(summary: pd.DataFrame) -> Figure:
    _check_scaling_columns(summary)
    _check_target_kind(summary, "general")
    if len(_noise_stds(summary)) > 1:
        return _plot_noise_subfigures(
            summary,
            _plot_scaling_grid,
            _grid_figure_size(summary),
        )
    return cast(Figure, _plot_scaling_grid(summary))


def plot_monotone_scaling(summary: pd.DataFrame) -> Figure:
    _check_scaling_columns(summary)
    _check_target_kind(summary, "monotone")
    if set(summary["model"].unique()) != set(_MONOTONE_MODEL_PAIR):
        raise ValueError(
            "monotone scaling expects unconstrained and monotone models"
        )
    if len(_noise_stds(summary)) > 1:
        return _plot_noise_subfigures(
            summary,
            _plot_pairwise_scaling,
            _pairwise_figure_size(summary),
        )
    return cast(Figure, _plot_pairwise_scaling(summary))
