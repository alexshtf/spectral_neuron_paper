import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.targets import TargetSpec, make_target

SCALING_COLUMNS = {
    "complexity",
    "model",
    "dim",
    "budget",
    "median_test_rmse",
    "q25_test_rmse",
    "q75_test_rmse",
}

MODEL_LINESTYLES = {
    "unconstrained": "-",
    "monotone": "--",
}


def _subplot_matrix(
    n_rows: int, n_cols: int, *, cell_width: float, cell_height: float
):
    if n_rows == 0 or n_cols == 0:
        raise ValueError("at least one item is required")

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_width * n_cols, cell_height * n_rows),
        squeeze=False,
        layout="constrained",
    )
    return fig, axs


def _subplot_grid(n_items: int, *, cell_width: float, cell_height: float):
    if n_items == 0:
        raise ValueError("at least one item is required")

    n_cols = int(math.ceil(math.sqrt(n_items)))
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axs = _subplot_matrix(
        n_rows, n_cols, cell_width=cell_width, cell_height=cell_height
    )
    axes = axs.ravel()

    for ax in axes[n_items:]:
        ax.set_visible(False)

    return fig, axes


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
    group = group.sort_values("budget")
    x = group["budget"].to_numpy()
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


def _use_pairwise_scaling(
    summary: pd.DataFrame, model_pair: tuple[str, str]
) -> bool:
    if "target_kind" not in summary or "model" not in summary:
        return False
    target_kinds = set(summary["target_kind"].dropna().unique())
    models = set(summary["model"].dropna().unique())
    return target_kinds == {"monotone"} and set(model_pair).issubset(models)


def _add_shared_legend(fig, axes) -> None:
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
        )


def _check_scaling_columns(summary: pd.DataFrame) -> None:
    missing = SCALING_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")


def _check_single_target_kind(summary: pd.DataFrame) -> None:
    if "target_kind" not in summary:
        return

    target_kinds = sorted(summary["target_kind"].dropna().unique())
    if len(target_kinds) > 1:
        raise ValueError(
            "plot_scaling expects a single target_kind; "
            f"filter summary first, got {target_kinds}"
        )


def plot_scaling(
    summary: pd.DataFrame,
    *,
    pair_by_dim: bool | None = None,
    model_pair: tuple[str, str] = ("unconstrained", "monotone"),
):
    _check_scaling_columns(summary)
    _check_single_target_kind(summary)
    if pair_by_dim is None:
        pair_by_dim = _use_pairwise_scaling(summary, model_pair)
    if pair_by_dim:
        return plot_pairwise_scaling(summary, model_pair=model_pair)

    complexities = sorted(summary["complexity"].unique())
    dims = sorted(summary["dim"].unique())
    models = _ordered_models(summary["model"].unique())
    dim_colors = _scaling_dim_colors(dims)
    model_linestyles = _scaling_model_linestyles(models)
    fig, axes = _subplot_grid(
        len(complexities), cell_width=4.5, cell_height=3.5
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
        ax.set_xlabel("budget")
        ax.set_ylabel("test RMSE")
        ax.set_xscale("log")
        if (sub["median_test_rmse"] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")

    return fig


def plot_pairwise_scaling(
    summary: pd.DataFrame,
    *,
    model_pair: tuple[str, str] = ("unconstrained", "monotone"),
):
    _check_scaling_columns(summary)
    _check_single_target_kind(summary)
    paired = summary.loc[summary["model"].isin(model_pair)].copy()
    complexities = sorted(paired["complexity"].unique())
    dims = sorted(paired["dim"].unique())
    fig, axs = _subplot_matrix(
        len(complexities), len(dims), cell_width=4.0, cell_height=3.0
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

            for model in model_pair:
                group = sub.loc[sub["model"] == model]
                if not group.empty:
                    _plot_curve(ax, group, label=model)

            ax.set_title(f"complexity={complexity}, dim={dim}")
            ax.set_xlabel("budget")
            ax.set_ylabel("test RMSE")
            ax.set_xscale("log")
            if (sub["median_test_rmse"] > 0).all():
                ax.set_yscale("log")
            ax.grid(True, alpha=0.25)

    _add_shared_legend(fig, axs.ravel())

    return fig


def plot_target_gallery(specs: list[TargetSpec]):
    fig, axes = _subplot_grid(len(specs), cell_width=4, cell_height=3)

    for spec, ax in zip(specs, axes):
        target = make_target(spec)
        xs = np.linspace(spec.lower, spec.upper, 1000)
        ax.plot(xs, target(xs))
        ax.set_title(f"{spec.kind}, complexity={spec.complexity}, seed={spec.seed}")

    return fig
