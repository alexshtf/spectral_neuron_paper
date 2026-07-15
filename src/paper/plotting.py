import math
from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure

from paper.targets import TargetSpec, make_bivariate_target, make_target

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

CRITEO_METRIC_LABELS = {
    "logloss": "test log loss",
    "brier": "test Brier score",
}

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


def _noise_stds(summary: pd.DataFrame) -> list[float]:
    if "noise_std" not in summary:
        return []
    return sorted(summary["noise_std"].dropna().unique().tolist())


def _noise_title(noise_std: float) -> str:
    if noise_std == 0:
        return "Noiseless training (σ = 0)"
    return f"Noisy training (σ = {noise_std:g})"


def _scaling_figure_size(
    summary: pd.DataFrame, *, pair_by_dim: bool
) -> tuple[float, float]:
    complexities = summary["complexity"].nunique()
    if pair_by_dim:
        return 4.0 * summary["dim"].nunique(), 3.0 * complexities

    n_cols = int(math.ceil(math.sqrt(complexities)))
    n_rows = int(math.ceil(complexities / n_cols))
    return 4.5 * n_cols, 3.5 * n_rows


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
        ax.set_xlabel("budget")
        ax.set_ylabel("test RMSE")
        ax.set_xscale("log")
        if (sub["median_test_rmse"] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")

    return fig


def _plot_pairwise_scaling(
    summary: pd.DataFrame,
    *,
    model_pair: tuple[str, str],
    container: FigureContainer | None = None,
) -> FigureContainer:
    paired = summary.loc[summary["model"].isin(model_pair)].copy()
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


def _plot_noise_subfigures(
    summary: pd.DataFrame,
    *,
    pair_by_dim: bool,
    model_pair: tuple[str, str],
) -> Figure:
    noise_stds = _noise_stds(summary)
    width, height = _scaling_figure_size(summary, pair_by_dim=pair_by_dim)
    fig = plt.figure(
        figsize=(width, height * len(noise_stds)),
        layout="constrained",
    )
    subfigures = fig.subfigures(len(noise_stds), 1, squeeze=False).ravel()

    for noise_std, subfigure in zip(noise_stds, subfigures):
        noise_summary = summary.loc[summary["noise_std"] == noise_std]
        subfigure.suptitle(_noise_title(noise_std), x=0.01, ha="left")
        if pair_by_dim:
            _plot_pairwise_scaling(
                noise_summary,
                model_pair=model_pair,
                container=subfigure,
            )
        else:
            _plot_scaling_grid(noise_summary, container=subfigure)

    return fig


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
    if len(_noise_stds(summary)) > 1:
        return _plot_noise_subfigures(
            summary,
            pair_by_dim=pair_by_dim,
            model_pair=model_pair,
        )
    if pair_by_dim:
        return _plot_pairwise_scaling(summary, model_pair=model_pair)
    return _plot_scaling_grid(summary)


CRITEO_MODEL_LABELS = {
    "linear": "Linear",
    "linear-new": "Linear-new",
    "fm": "FM",
    "spectral-old": "Spectral-old",
    "spectral-new": "Spectral-new",
}

CRITEO_MODEL_COLORS = {
    "Linear": "#555555",
    "Linear-new": "#CC78BC",
    "FM": "#0173B2",
    "Spectral-old": "#DE8F05",
    "Spectral-new": "#029E73",
}

CRITEO_MODEL_MARKERS = {
    "Linear": "o",
    "Linear-new": "P",
    "FM": "s",
    "Spectral-old": "^",
    "Spectral-new": "D",
}

CRITEO_MODEL_DASHES = {
    "Linear": "",
    "Linear-new": (2, 2),
    "FM": "",
    "Spectral-old": (4, 2),
    "Spectral-new": "",
}

type CriteoSpectralVariant = Literal["spectral-old", "spectral-new"]


def _check_criteo_results(results: pd.DataFrame, metric: str) -> str:
    if metric not in CRITEO_METRIC_LABELS:
        raise ValueError(
            f"metric must be one of {sorted(CRITEO_METRIC_LABELS)}; got {metric!r}"
        )

    value = f"test_{metric}"
    required = {
        "train_size",
        "model",
        "dim",
        value,
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    return value


def _spectral_dimensions(results: pd.DataFrame) -> list[int]:
    values = results.loc[
        results["model"].isin(("spectral-old", "spectral-new")),
        "dim",
    ].unique()
    dimensions = sorted(map(int, values))
    if not dimensions:
        raise ValueError("results contain no spectral models")
    return dimensions


def _finish_criteo_grid(
    grid: sns.FacetGrid,
    *,
    metric: str,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    label = CRITEO_METRIC_LABELS[metric]
    grid.set_axis_labels("training impressions", f"{label} ↓")
    for ax in grid.axes.flat:
        ax.set_xscale("log", base=2)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)
    grid.figure.suptitle(title, y=1.02)
    return grid.figure


def _criteo_relplot(
    results: pd.DataFrame,
    *,
    metric: str,
    title: str,
    by: str,
    hue_order: list,
    palette,
    markers,
    dashes,
    col: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    value = _check_criteo_results(results, metric)
    facet = {"col": col, "col_wrap": 2} if col is not None else {}
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
        grid.legend.set_title("model" if by == "model_label" else "dimension")
    return _finish_criteo_grid(grid, metric=metric, title=title, xlim=xlim)


def _label_criteo_models(results: pd.DataFrame) -> pd.DataFrame:
    labeled = results.copy()
    labeled["model_label"] = labeled["model"].map(CRITEO_MODEL_LABELS)
    return labeled


def plot_criteo_models_by_dimension(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare parameter-matched models in one facet per spectral dimension."""
    _check_criteo_results(results, metric)
    dimensions = _spectral_dimensions(results)

    nonlinear = results.loc[
        results["model"].isin(("fm", "spectral-old", "spectral-new"))
    ].copy()
    if not set(nonlinear["dim"]) <= set(dimensions):
        raise ValueError("some nonlinear models have no matched spectral dimension")
    nonlinear["dimension"] = nonlinear["dim"]

    linears = results.loc[
        results["model"].isin(("linear", "linear-new"))
    ].merge(
        pd.DataFrame({"dimension": dimensions}), how="cross"
    )
    faceted = _label_criteo_models(pd.concat((linears, nonlinear)))
    model_order = list(CRITEO_MODEL_LABELS.values())
    return _criteo_relplot(
        faceted,
        metric=metric,
        title=f"Criteo {CRITEO_METRIC_LABELS[metric]}: matched models",
        by="model_label",
        hue_order=model_order,
        palette=CRITEO_MODEL_COLORS,
        markers=CRITEO_MODEL_MARKERS,
        dashes=CRITEO_MODEL_DASHES,
        col="dimension",
    )


def plot_criteo_spectral_comparison(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare old and new spectral preprocessing in one facet per dimension."""
    spectral = _label_criteo_models(
        results.loc[results["model"].isin(("spectral-old", "spectral-new"))]
    )
    model_order = ["Spectral-old", "Spectral-new"]
    return _criteo_relplot(
        spectral,
        metric=metric,
        title=f"Criteo {CRITEO_METRIC_LABELS[metric]}: spectral preprocessing",
        by="model_label",
        hue_order=model_order,
        palette=CRITEO_MODEL_COLORS,
        markers=CRITEO_MODEL_MARKERS,
        dashes=CRITEO_MODEL_DASHES,
        col="dim",
    )


def plot_criteo_spectral_dimensions(
    results: pd.DataFrame,
    variant: CriteoSpectralVariant,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare all dimensions of one spectral preprocessing variant."""
    if variant not in ("spectral-old", "spectral-new"):
        raise ValueError(f"unknown spectral variant {variant!r}")
    spectral = results.loc[results["model"] == variant]
    dimensions = _spectral_dimensions(spectral)
    palette = dict(zip(dimensions, sns.color_palette("colorblind", len(dimensions))))
    markers = dict(zip(dimensions, ("o", "s", "^", "D")))
    return _criteo_relplot(
        spectral,
        metric=metric,
        title=(
            f"Criteo {CRITEO_METRIC_LABELS[metric]}: "
            f"{CRITEO_MODEL_LABELS[variant]} across dimensions"
        ),
        by="dim",
        hue_order=dimensions,
        palette=palette,
        markers=markers,
        dashes=False,
        xlim=xlim,
    )


def plot_criteo_fm_dimensions(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare FM embedding dimensions."""
    fm = results.loc[results["model"] == "fm"].copy()
    fm["rank"] = fm["dim"].map(lambda dim: dim * (dim + 1) // 2 - 1)
    ranks = sorted(map(int, fm["rank"].unique()))
    if not ranks:
        raise ValueError("results contain no FM models")
    palette = dict(zip(ranks, sns.color_palette("colorblind", len(ranks))))
    markers = dict(zip(ranks, ("o", "s", "^", "D")))
    return _criteo_relplot(
        fm,
        metric=metric,
        title=f"Criteo {CRITEO_METRIC_LABELS[metric]}: FM across embedding dimensions",
        by="rank",
        hue_order=ranks,
        palette=palette,
        markers=markers,
        dashes=False,
        xlim=xlim,
    )


def plot_target_gallery(specs: list[TargetSpec]):
    fig, axes = _subplot_grid(len(specs), cell_width=4, cell_height=3)

    for spec, ax in zip(specs, axes):
        target = make_target(spec)
        xs = np.linspace(spec.lower, spec.upper, 1000)
        ax.plot(xs, target(xs))
        ax.set_title(f"{spec.kind}, complexity={spec.complexity}, seed={spec.seed}")

    return fig


def plot_bivariate_target_gallery(
    specs: list[TargetSpec], *, resolution: int = 200
):
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
