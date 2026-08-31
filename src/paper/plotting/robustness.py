from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

_SHELL_LINESTYLES = ("-", "--", ":", "-.")


@dataclass(frozen=True)
class _DeviationCurveStyle:
    shell_index: int
    label: str
    color: tuple[float, float, float]
    linestyle: str


def _deviation_curve_styles(
    shell_labels: list[str],
) -> tuple[_DeviationCurveStyle, ...]:
    return tuple(
        _DeviationCurveStyle(
            shell_index,
            label,
            color,
            _SHELL_LINESTYLES[shell_index % len(_SHELL_LINESTYLES)],
        )
        for shell_index, (label, color) in enumerate(
            zip(
                shell_labels,
                sns.color_palette("colorblind", len(shell_labels)),
                strict=True,
            )
        )
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
    feature_column_width_mm: float = 24.0,
    dimension_row_height_mm: float = 18.0,
) -> Figure:
    """Plot shell probabilities in two dimension-by-feature subfigures."""
    noise_level, ratio_columns, averaged = _mean_higgs_deviation_shells(
        results, shell_count=shell_count
    )

    cell_peaks = (
        averaged.groupby("feature_index", sort=False)[ratio_columns]
        .max()
        .max(axis="columns")
    )

    dimensions = sorted(map(int, averaged["dim"].unique()))
    features = (
        averaged[["feature_index", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_index", kind="stable")
    )
    ratio_edges = np.linspace(0, 1, len(ratio_columns) + 1)
    shell_edges = np.linspace(0, noise_level, shell_count + 1)
    shell_labels = [
        (
            f"|δ| ∈ [{shell_edges[index]:g}, {shell_edges[index + 1]:g}"
            f"{']' if index == shell_count - 1 else ')'}"
        )
        for index in range(shell_count)
    ]
    styles = _deviation_curve_styles(shell_labels)
    feature_split = (len(features) + 1) // 2
    feature_groups = (
        features.iloc[:feature_split],
        features.iloc[feature_split:],
    )

    fig = plt.figure(
        figsize=(
            feature_column_width_mm * max(map(len, feature_groups)) / 25.4,
            2 * (dimension_row_height_mm * len(dimensions) + 40) / 25.4,
        ),
        layout="constrained",
    )
    subfigures = fig.subfigures(2, 1)
    for subfigure, feature_group in zip(
        subfigures, feature_groups, strict=True
    ):
        axes = subfigure.subplots(
            len(dimensions),
            len(feature_group),
            sharex=True,
            squeeze=False,
        )
        subfigure.supylabel("Matrix dimension", fontsize=8)
        for dimension_row, (dimension, row_axes) in enumerate(
            zip(dimensions, axes, strict=True)
        ):
            for feature_column, (feature, ax) in enumerate(
                zip(feature_group.itertuples(index=False), row_axes, strict=True)
            ):
                feature_index = int(feature.feature_index)
                cell = averaged.loc[
                    (averaged["dim"] == dimension)
                    & (averaged["feature_index"] == feature_index)
                ]
                for style in styles:
                    histogram = cell.loc[
                        cell["shell_index"] == style.shell_index, ratio_columns
                    ]
                    if histogram.empty or histogram.iloc[0].isna().all():
                        continue

                    heights = histogram.iloc[0].fillna(0).to_numpy()
                    heights = np.r_[heights, heights[-1]]
                    ax.fill_between(
                        ratio_edges,
                        0,
                        heights,
                        step="post",
                        color=style.color,
                        alpha=0.05,
                        linewidth=0,
                    )
                    ax.step(
                        ratio_edges,
                        heights,
                        where="post",
                        color=style.color,
                        linestyle=style.linestyle,
                        linewidth=1.1,
                    )

                peak = cell_peaks.at[feature_index]
                ax.axhline(0, color="#d9d9d9", linewidth=0.4, zorder=0)
                ax.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
                ax.set(xlim=(0, 1), ylim=(0, peak / 0.85))
                ax.set_box_aspect(
                    dimension_row_height_mm / feature_column_width_mm
                )
                ax.set_xticks(np.linspace(0, 1, 3))
                ax.set_yticks([])
                ax.tick_params(
                    axis="x",
                    labelsize=7,
                    labelbottom=dimension_row == len(dimensions) - 1,
                )
                if feature_column == 0:
                    ax.set_ylabel(
                        str(dimension),
                        fontsize=8,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=3,
                    )
                if dimension_row == 0:
                    ax.set_title(
                        feature.feature_name,
                        fontsize=8,
                        rotation=45,
                        ha="left",
                        va="bottom",
                        pad=4,
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

    handles = [
        Line2D(
            [],
            [],
            color=style.color,
            linestyle=style.linestyle,
            linewidth=1.2,
            label=style.label,
        )
        for style in styles
    ]
    fig.legend(
        handles=handles,
        title="Perturbation-magnitude shell",
        loc="outside upper right",
        ncols=len(handles),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    return fig
