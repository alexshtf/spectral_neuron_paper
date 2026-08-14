import math
from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

from paper.targets import TargetSpec, make_bivariate_target, make_target

SCALING_COLUMNS = {
    "complexity",
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

BINARY_METRIC_LABELS = {
    "logloss": "test log loss",
    "brier": "test Brier score",
}
TRAIN_SIZE_LABEL = "training samples processed by optimizer"
SYNTHETIC_TRAIN_SIZE_LABEL = "training-sample budget"

HIGGS_MODEL_LABELS = {
    "linear": "Linear",
    "spectral": "Spectral",
    "mlp-1": "MLP-1",
    "mlp-2": "MLP-2",
    "mlp-3": "MLP-3",
}

HIGGS_MODEL_COLORS = {
    "Linear": "#555555",
    "Spectral": "#029E73",
    "MLP-1": "#0173B2",
    "MLP-2": "#DE8F05",
    "MLP-3": "#CC78BC",
}

HIGGS_MODEL_MARKERS = {
    "Linear": "o",
    "Spectral": "^",
    "MLP-1": "s",
    "MLP-2": "D",
    "MLP-3": "P",
}

HIGGS_MODEL_DASHES = {
    "Linear": "",
    "Spectral": "",
    "MLP-1": (4, 2),
    "MLP-2": (2, 2),
    "MLP-3": (4, 2, 1, 2),
}

HIGGS_DIMENSION_LINESTYLES = ("-", "--", ":", "-.")
_MONOTONE_MODEL_PAIR = ("unconstrained", "monotone")

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


def _use_pairwise_scaling(summary: pd.DataFrame) -> bool:
    if "target_kind" not in summary or "model" not in summary:
        return False
    target_kinds = set(summary["target_kind"].dropna().unique())
    models = set(summary["model"].dropna().unique())
    return target_kinds == {"monotone"} and set(_MONOTONE_MODEL_PAIR) <= models


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
    if summary["noise_std"].isna().any():
        raise ValueError("noise_std must not contain missing values")
    return sorted(summary["noise_std"].unique().tolist())


def _noise_title(noise_std: float) -> str:
    if noise_std == 0:
        return "Noiseless training (σ = 0)"
    return f"Noisy training (σ = {noise_std:g})"


def _scaling_figure_size(summary: pd.DataFrame) -> tuple[float, float]:
    complexities = summary["complexity"].nunique()
    if _use_pairwise_scaling(summary):
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
        ax.set_xlabel(SYNTHETIC_TRAIN_SIZE_LABEL)
        ax.set_ylabel("test RMSE")
        ax.set_xscale("log")
        if (sub["median_test_rmse"] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

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
            ax.set_xlabel(SYNTHETIC_TRAIN_SIZE_LABEL)
            ax.set_ylabel("test RMSE")
            ax.set_xscale("log")
            if (sub["median_test_rmse"] > 0).all():
                ax.set_yscale("log")
            ax.grid(True, alpha=0.25)

    _add_shared_legend(
        fig,
        axs.ravel(),
        borderaxespad=2 if isinstance(fig, SubFigure) else 0.5,
    )

    return fig


def _plot_noise_subfigures(summary: pd.DataFrame) -> Figure:
    noise_stds = _noise_stds(summary)
    pairwise = _use_pairwise_scaling(summary)
    width, height = _scaling_figure_size(summary)
    fig = plt.figure(
        figsize=(width, height * len(noise_stds)),
        layout="constrained",
    )
    fig.get_layout_engine().set(h_pad=0.12)
    subfigures = fig.subfigures(len(noise_stds), 1, squeeze=False).ravel()

    for noise_std, subfigure in zip(noise_stds, subfigures):
        noise_summary = summary.loc[summary["noise_std"] == noise_std]
        subfigure.suptitle(_noise_title(noise_std), y=1, fontsize="x-large")
        if pairwise:
            _plot_pairwise_scaling(
                noise_summary,
                container=subfigure,
            )
        else:
            _plot_scaling_grid(noise_summary, container=subfigure)

    return fig


def plot_scaling(summary: pd.DataFrame):
    _check_scaling_columns(summary)
    _check_single_target_kind(summary)
    if len(_noise_stds(summary)) > 1:
        return _plot_noise_subfigures(summary)
    if _use_pairwise_scaling(summary):
        return _plot_pairwise_scaling(summary)
    return _plot_scaling_grid(summary)


CRITEO_MODEL_LABELS = {
    "linear-bucketed": "Linear (bucketed)",
    "linear-continuous": "Linear (continuous)",
    "fm": "FM",
    "spectral-bucketed": "Spectral (bucketed)",
    "spectral-continuous": "Spectral (continuous)",
}

CRITEO_MODEL_COLORS = {
    "Linear (bucketed)": "#555555",
    "Linear (continuous)": "#CC78BC",
    "FM": "#0173B2",
    "Spectral (bucketed)": "#DE8F05",
    "Spectral (continuous)": "#029E73",
}

CRITEO_MODEL_MARKERS = {
    "Linear (bucketed)": "o",
    "Linear (continuous)": "P",
    "FM": "s",
    "Spectral (bucketed)": "^",
    "Spectral (continuous)": "D",
}

CRITEO_MODEL_DASHES = {
    "Linear (bucketed)": "",
    "Linear (continuous)": (2, 2),
    "FM": "",
    "Spectral (bucketed)": (4, 2),
    "Spectral (continuous)": "",
}

type CriteoSpectralVariant = Literal[
    "spectral-bucketed",
    "spectral-continuous",
]

_DIMENSION_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


def _check_criteo_results(results: pd.DataFrame, metric: str) -> str:
    if metric not in BINARY_METRIC_LABELS:
        raise ValueError(
            f"metric must be one of {sorted(BINARY_METRIC_LABELS)}; got {metric!r}"
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
        results["model"].isin(("spectral-bucketed", "spectral-continuous")),
        "dim",
    ].unique()
    dimensions = sorted(map(int, values))
    if not dimensions:
        raise ValueError("results contain no spectral models")
    return dimensions


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
    return _binary_relplot(
        results,
        value=_check_criteo_results(results, metric),
        metric=metric,
        title=title,
        x_label=TRAIN_SIZE_LABEL,
        by=by,
        hue_order=hue_order,
        palette=palette,
        markers=markers,
        dashes=dashes,
        legend_title="model" if by == "model_label" else "dimension",
        col=col,
        xlim=xlim,
    )


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
        results["model"].isin(("fm", "spectral-bucketed", "spectral-continuous"))
    ].copy()
    if not set(nonlinear["dim"]) <= set(dimensions):
        raise ValueError("some nonlinear models have no matched spectral dimension")
    nonlinear["dimension"] = nonlinear["dim"]

    linears = results.loc[
        results["model"].isin(("linear-bucketed", "linear-continuous"))
    ].merge(
        pd.DataFrame({"dimension": dimensions}), how="cross"
    )
    faceted = _label_criteo_models(pd.concat((linears, nonlinear)))
    model_order = list(CRITEO_MODEL_LABELS.values())
    return _criteo_relplot(
        faceted,
        metric=metric,
        title=f"Criteo {BINARY_METRIC_LABELS[metric]}: matched models",
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
    """Compare bucketed and continuous spectral preprocessing by dimension."""
    spectral = _label_criteo_models(
        results.loc[
            results["model"].isin(
                ("spectral-bucketed", "spectral-continuous")
            )
        ]
    )
    model_order = ["Spectral (bucketed)", "Spectral (continuous)"]
    return _criteo_relplot(
        spectral,
        metric=metric,
        title=f"Criteo {BINARY_METRIC_LABELS[metric]}: spectral preprocessing",
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
    spectral = results.loc[results["model"] == variant]
    dimensions = _spectral_dimensions(spectral)
    palette, markers = _dimension_styles(dimensions)
    return _criteo_relplot(
        spectral,
        metric=metric,
        title=(
            f"Criteo {BINARY_METRIC_LABELS[metric]}: "
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
    palette, markers = _dimension_styles(ranks)
    return _criteo_relplot(
        fm,
        metric=metric,
        title=f"Criteo {BINARY_METRIC_LABELS[metric]}: FM across embedding dimensions",
        by="rank",
        hue_order=ranks,
        palette=palette,
        markers=markers,
        dashes=False,
        xlim=xlim,
    )


def _higgs_ratio_count_columns(results: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in results
        if column.startswith("ratio_bin_") and column.endswith("_count")
    ]
    expected = [
        f"ratio_bin_{bin_index:03d}_count"
        for bin_index in range(len(columns))
    ]
    if not columns or columns != expected:
        raise ValueError("ratio count columns must be contiguous and ordered")
    return columns


def _mean_higgs_deviation_shells(
    results: pd.DataFrame,
    *,
    shell_count: int,
) -> tuple[float, list[str], pd.DataFrame]:
    ratio_columns = _higgs_ratio_count_columns(results)
    noise_levels = results["noise_level"].unique()
    if len(noise_levels) != 1:
        raise ValueError(
            "plot_higgs_deviation_shell_grid expects one noise_level; "
            f"got {sorted(map(float, noise_levels))}"
        )
    magnitude_bins = sorted(map(int, results["magnitude_bin_index"].unique()))
    if magnitude_bins != list(range(len(magnitude_bins))):
        raise ValueError("magnitude bins must be contiguous and zero-based")
    if shell_count <= 0 or len(magnitude_bins) % shell_count:
        raise ValueError(
            f"shell_count must be a positive divisor of {len(magnitude_bins)}"
        )

    bins_per_shell = len(magnitude_bins) // shell_count
    shelled = results.copy()
    shelled["shell_index"] = shelled["magnitude_bin_index"] // bins_per_shell
    run_columns = [
        "dim",
        "data_seed",
        "init_seed",
        "feature_index",
        "feature_name",
        "shell_index",
    ]
    sums = (
        shelled.groupby(run_columns, as_index=False, sort=True)[
            ratio_columns + ["total_count", "zero_bound_count"]
        ]
        .sum()
    )
    defined_count = sums["total_count"] - sums["zero_bound_count"]
    probabilities = sums[ratio_columns].div(
        defined_count.where(defined_count > 0), axis=0
    )
    probabilities[run_columns] = sums[run_columns]
    average_columns = ["dim", "feature_index", "feature_name", "shell_index"]
    averaged = (
        probabilities.groupby(average_columns, as_index=False, sort=True)[
            ratio_columns
        ]
        .mean()
    )
    return float(noise_levels[0]), ratio_columns, averaged


def plot_higgs_deviation_shell_grid(
    results: pd.DataFrame,
    *,
    shell_count: int = 4,
    feature_row_height_mm: float = 12.0,
) -> Figure:
    """Plot raw histogram probabilities with one y-scale per grid cell."""
    noise_level, ratio_columns, averaged = _mean_higgs_deviation_shells(
        results, shell_count=shell_count
    )

    cell_peaks = (
        averaged.groupby(["feature_index", "shell_index"], sort=False)[
            ratio_columns
        ]
        .max()
        .max(axis="columns")
    )

    dimensions = sorted(map(int, averaged["dim"].unique()))
    colors, _ = _dimension_styles(dimensions)
    linestyles = {
        dim: HIGGS_DIMENSION_LINESTYLES[
            index % len(HIGGS_DIMENSION_LINESTYLES)
        ]
        for index, dim in enumerate(dimensions)
    }
    features = (
        averaged[["feature_index", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_index", kind="stable")
    )
    ratio_edges = np.linspace(0, 1, len(ratio_columns) + 1)
    shell_edges = np.linspace(0, noise_level, shell_count + 1)

    fig, axes = plt.subplots(
        len(features),
        shell_count,
        figsize=(
            max(90, 45 * shell_count) / 25.4,
            feature_row_height_mm * len(features) / 25.4,
        ),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for feature_row, feature in enumerate(features.itertuples(index=False)):
        feature_index = int(feature.feature_index)
        for shell_index, ax in enumerate(axes[feature_row]):
            cell = averaged.loc[
                (averaged["feature_index"] == feature_index)
                & (averaged["shell_index"] == shell_index)
            ]
            for dim in dimensions:
                histogram = cell.loc[cell["dim"] == dim, ratio_columns]
                if histogram.empty or histogram.iloc[0].isna().all():
                    continue

                heights = histogram.iloc[0].fillna(0).to_numpy()
                heights = np.r_[heights, heights[-1]]
                ax.fill_between(
                    ratio_edges,
                    0,
                    heights,
                    step="post",
                    color=colors[dim],
                    alpha=0.07,
                    linewidth=0,
                )
                ax.step(
                    ratio_edges,
                    heights,
                    where="post",
                    color=colors[dim],
                    linestyle=linestyles[dim],
                    linewidth=1.1,
                )

            peak = cell_peaks.at[(feature_index, shell_index)]
            ax.axhline(0, color="#d9d9d9", linewidth=0.4, zorder=0)
            ax.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
            ax.set(xlim=(0, 1), ylim=(0, peak / 0.85))
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks([])
            ax.tick_params(
                axis="x",
                labelsize=7,
                labelbottom=feature_row == len(features) - 1,
            )
            if shell_index == 0:
                ax.set_ylabel(
                    feature.feature_name,
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=3,
                )
            if feature_row == 0:
                bracket = "]" if shell_index == shell_count - 1 else ")"
                ax.set_title(
                    f"|δ| ∈ [{shell_edges[shell_index]:g}, "
                    f"{shell_edges[shell_index + 1]:g}{bracket}",
                    fontsize=8,
                )
            ax.grid(axis="x", alpha=0.2, linewidth=0.5)
            sns.despine(ax=ax, left=True)

    fig.suptitle(
        (
            "HIGGS deviation ratios under "
            f"δ ∼ Uniform(−ε, ε), ε = {noise_level:g}"
        ),
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.supxlabel("Deviation ratio  |Δf| / (|δ| ‖Aⱼ‖₂)", fontsize=8)
    fig.supylabel("Feature", fontsize=8)

    handles = [
        Line2D(
            [],
            [],
            color=colors[dim],
            linestyle=linestyles[dim],
            linewidth=1.2,
            label=str(dim),
        )
        for dim in dimensions
    ]
    fig.legend(
        handles=handles,
        title="Matrix dimension",
        loc="outside upper right",
        ncols=len(handles),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    return fig


def _check_higgs_metric(results: pd.DataFrame, metric: str) -> str:
    if metric not in BINARY_METRIC_LABELS:
        raise ValueError(
            f"metric must be one of {sorted(BINARY_METRIC_LABELS)}; got {metric!r}"
        )

    value = f"test_{metric}"
    required = {"train_size", "model", "dim", value}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results are empty")
    if results[list(required)].isna().any().any():
        raise ValueError("HIGGS plotting columns must not contain missing values")
    return value


def _check_higgs_results(results: pd.DataFrame, metric: str) -> tuple[str, list[int]]:
    value = _check_higgs_metric(results, metric)
    capacity_columns = {"width", "num_parameters"}
    missing = capacity_columns.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    dimensions = sorted(
        map(int, results.loc[results["model"] == "spectral", "dim"].unique())
    )
    if not dimensions:
        raise ValueError("results contain no spectral models")
    return value, dimensions


def _higgs_capacity_title(results: pd.DataFrame, dim: int) -> str:
    capacity = (
        results.loc[
            (results["dim"] == dim) & (results["model"] != "linear"),
            ["model", "width", "num_parameters"],
        ]
        .drop_duplicates()
        .set_index("model")
    )
    spectral_parameters = int(capacity.at["spectral", "num_parameters"])
    mlp_models = ("mlp-1", "mlp-2", "mlp-3")
    entries = tuple(
        f"{model.removeprefix('mlp-')}×{int(capacity.at[model, 'width'])} "
        f"({int(capacity.at[model, 'num_parameters']):,}p)"
        for model in mlp_models
    )
    return (
        f"dim={dim} · Spectral {spectral_parameters:,}p\n"
        f"MLP {entries[0]} · {entries[1]}\n"
        f"MLP {entries[2]}"
    )


def plot_higgs_models_by_dimension(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare HIGGS model families in one facet per matched dimension."""
    value, dimensions = _check_higgs_results(results, metric)

    nonlinear = results.loc[results["model"] != "linear"].copy()
    nonlinear["dimension"] = nonlinear["dim"].astype(int)
    linears = results.loc[results["model"] == "linear"].merge(
        pd.DataFrame({"dimension": dimensions}), how="cross"
    )
    faceted = pd.concat((linears, nonlinear), ignore_index=True)
    faceted["model_label"] = faceted["model"].map(HIGGS_MODEL_LABELS)
    model_order = list(HIGGS_MODEL_LABELS.values())

    grid = sns.relplot(
        data=faceted,
        x="train_size",
        y=value,
        hue="model_label",
        style="model_label",
        hue_order=model_order,
        style_order=model_order,
        palette=HIGGS_MODEL_COLORS,
        markers=HIGGS_MODEL_MARKERS,
        dashes=HIGGS_MODEL_DASHES,
        col="dimension",
        col_order=dimensions,
        col_wrap=2,
        kind="line",
        estimator=np.median,
        errorbar=("pi", 50),
        err_kws={"alpha": 0.15},
        linewidth=2,
        height=3.8,
        aspect=1.25,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels(
        TRAIN_SIZE_LABEL,
        f"{BINARY_METRIC_LABELS[metric]} ↓",
    )
    for dim, ax in zip(dimensions, grid.axes.flat):
        ax.set_title(_higgs_capacity_title(results, dim), fontsize="small")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
    if grid.legend is not None:
        grid.legend.set_title("model")
    grid.figure.suptitle(
        f"HIGGS {BINARY_METRIC_LABELS[metric]}: matched model families",
        y=0.99,
    )
    top = 0.72 if len(dimensions) <= 2 else 0.88
    grid.figure.subplots_adjust(top=top, hspace=0.55)
    return grid.figure


def plot_higgs_spectral_dimensions(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare all HIGGS spectral dimensions on one scaling axis."""
    value = _check_higgs_metric(results, metric)
    spectral = results.loc[results["model"] == "spectral"]
    dimensions = sorted(map(int, spectral["dim"].unique()))
    if not dimensions:
        raise ValueError("results contain no spectral models")

    palette, markers = _dimension_styles(dimensions)
    return _binary_relplot(
        spectral,
        value=value,
        metric=metric,
        title=(
            f"HIGGS {BINARY_METRIC_LABELS[metric]}: "
            "spectral neurons across dimensions"
        ),
        x_label=TRAIN_SIZE_LABEL,
        by="dim",
        hue_order=dimensions,
        palette=palette,
        markers=markers,
        dashes=False,
        legend_title="dimension",
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
