from itertools import product

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from paper.plotting import (
    plot_bivariate_target_gallery,
    plot_criteo_models_by_dimension,
    plot_criteo_fm_dimensions,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
    plot_higgs_deviation_shell_grid,
    plot_higgs_models_by_dimension,
    plot_higgs_spectral_dimensions,
    plot_scaling,
)
from paper.targets import TargetSpec


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _row(
    *,
    complexity: int,
    dim: int,
    model: str,
    train_size: int,
    target_kind: str = "monotone",
    noise_std: float = 0.0,
) -> dict:
    return {
        "target_kind": target_kind,
        "complexity": complexity,
        "noise_std": noise_std,
        "model": model,
        "dim": dim,
        "train_size": train_size,
        "median_test_rmse": 1.0 / train_size,
        "q25_test_rmse": 0.8 / train_size,
        "q75_test_rmse": 1.2 / train_size,
    }


def test_plot_scaling_pairs_monotone_models_by_dimension():
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity, dim=dim, model=model, train_size=train_size
            )
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in ("unconstrained", "monotone")
            for train_size in (32, 64)
        ]
    )

    fig = plot_scaling(summary)
    axes = [ax for ax in fig.axes if ax.get_visible()]

    assert len(axes) == 4
    assert {ax.get_title() for ax in axes} == {
        "complexity=5, dim=3",
        "complexity=5, dim=5",
        "complexity=10, dim=3",
        "complexity=10, dim=5",
    }
    assert all(len(ax.lines) == 2 for ax in axes)
    assert all(
        ax.get_xlabel() == "training-sample budget" for ax in axes
    )


def test_plot_scaling_styles_models_and_dimensions():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=dim,
                model=model,
                train_size=train_size,
                target_kind="general",
            )
            for dim in (5, 9)
            for model in ("unconstrained", "monotone")
            for train_size in (32, 64)
        ]
    )

    fig = plot_scaling(summary)
    lines = {line.get_label(): line for line in fig.axes[0].lines}

    assert (
        lines["dim=5, unconstrained"].get_color()
        == lines["dim=5, monotone"].get_color()
    )
    assert (
        lines["dim=9, unconstrained"].get_color()
        == lines["dim=9, monotone"].get_color()
    )
    assert (
        lines["dim=5, unconstrained"].get_color()
        != lines["dim=9, unconstrained"].get_color()
    )
    assert lines["dim=5, unconstrained"].get_linestyle() == "-"
    assert lines["dim=5, monotone"].get_linestyle() == "--"


def test_plot_scaling_rejects_mixed_target_kinds():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=5,
                model="unconstrained",
                train_size=train_size,
                target_kind=target_kind,
            )
            for target_kind in ("general", "monotone")
            for train_size in (32, 64)
        ]
    )

    with pytest.raises(ValueError, match="single target_kind"):
        plot_scaling(summary)


@pytest.mark.parametrize(
    ("target_kind", "models", "expected_axes", "lines_per_axis"),
    [
        ("monotone", ("unconstrained", "monotone"), 4, 2),
        ("general", ("unconstrained",), 2, 2),
    ],
)
def test_plot_scaling_separates_noise_before_choosing_target_layout(
    target_kind, models, expected_axes, lines_per_axis
):
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity,
                dim=dim,
                model=model,
                train_size=train_size,
                target_kind=target_kind,
                noise_std=noise_std,
            )
            for noise_std in (0.0, 0.1)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in models
            for train_size in (32, 64)
        ]
    )

    fig = plot_scaling(summary)
    assert [subfigure._suptitle.get_text() for subfigure in fig.subfigs] == [
        "Noiseless training (σ = 0)",
        "Noisy training (σ = 0.1)",
    ]
    for subfigure in fig.subfigs:
        axes = [ax for ax in subfigure.axes if ax.get_visible()]
        assert len(axes) == expected_axes
        assert all(len(ax.lines) == lines_per_axis for ax in axes)
        assert all(len(line.get_xdata()) == 2 for ax in axes for line in ax.lines)


@pytest.mark.parametrize(("figsize", "dpi"), [((16, 6), 72), ((24, 9), 200)])
def test_plot_scaling_separates_noise_titles_legends_and_axes(figsize, dpi):
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity,
                dim=dim,
                model=model,
                train_size=train_size,
                noise_std=noise_std,
            )
            for noise_std in (0.0, 0.1)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in ("unconstrained", "monotone")
            for train_size in (32, 64)
        ]
    )

    fig = plot_scaling(summary)
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    fig.canvas.draw()

    for subfigure in fig.subfigs:
        title = subfigure._suptitle.get_window_extent()
        legend = subfigure.legends[0].get_window_extent()
        axes_titles = [ax.title.get_window_extent() for ax in subfigure.axes[:2]]

        assert title.y0 > legend.y1
        assert legend.y0 > max(axis_title.y1 for axis_title in axes_titles)


def test_plot_bivariate_target_gallery_draws_contours():
    specs = [
        TargetSpec(kind="general", complexity=5, seed=0),
        TargetSpec(kind="monotone", complexity=5, seed=0),
    ]

    fig = plot_bivariate_target_gallery(specs, resolution=20)
    axes = [ax for ax in fig.axes if ax.get_title()]

    assert [ax.get_title() for ax in axes] == [
        "general, complexity=5, seed=0",
        "monotone, complexity=5, seed=0",
    ]
    assert all(ax.collections for ax in axes)
    assert all(ax.get_xlabel() == "$x_1$" for ax in axes)
    assert all(ax.get_ylabel() == "$x_2$" for ax in axes)


def _criteo_results() -> pd.DataFrame:
    models = (
        ("linear-bucketed", 0),
        ("linear-continuous", 0),
        *(
            (model, dim)
            for dim in (3, 5)
            for model in ("fm", "spectral-bucketed", "spectral-continuous")
        ),
    )
    return pd.DataFrame(
        {
            "train_size": train_size,
            "model": model,
            "dim": dim,
            "test_logloss": 0.5 + dim / 100 + seed / 1000,
        }
        for train_size, seed, (model, dim) in product(
            (2**14, 2**18), range(3), models
        )
    )


def _legend_labels(fig) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_plot_criteo_models_by_dimension_facets_matched_models():
    fig = plot_criteo_models_by_dimension(_criteo_results())
    assert [ax.get_title() for ax in fig.axes] == ["dim=3", "dim=5"]
    assert _legend_labels(fig) == [
        "Linear (bucketed)",
        "Linear (continuous)",
        "FM",
        "Spectral (bucketed)",
        "Spectral (continuous)",
    ]
    assert fig.legends[0].get_title().get_text() == "model"
    assert len(fig.axes[1].lines) == 5
    assert len(fig.axes[1].collections) == 5
    assert all(ax.get_xscale() == "log" for ax in fig.axes)


def test_plot_criteo_spectral_comparison_facets_dimensions():
    fig = plot_criteo_spectral_comparison(_criteo_results())
    assert [ax.get_title() for ax in fig.axes] == ["dim=3", "dim=5"]
    assert _legend_labels(fig) == [
        "Spectral (bucketed)",
        "Spectral (continuous)",
    ]
    assert fig.legends[0].get_title().get_text() == "model"
    assert len(fig.axes[1].lines) == 2
    assert len(fig.axes[1].collections) == 2


@pytest.mark.parametrize(
    "variant", ["spectral-bucketed", "spectral-continuous"]
)
def test_plot_criteo_spectral_dimensions_uses_one_axis(variant):
    fig = plot_criteo_spectral_dimensions(_criteo_results(), variant)
    assert len(fig.axes) == 1
    assert _legend_labels(fig) == ["3", "5"]
    assert fig.legends[0].get_title().get_text() == "dimension"
    assert len(fig.axes[0].collections) == 2


def test_plot_criteo_fm_dimensions_supports_zoom():
    xlim = (2**14, 2**18)
    fig = plot_criteo_fm_dimensions(_criteo_results(), xlim=xlim)
    assert len(fig.axes) == 1
    assert _legend_labels(fig) == ["5", "14"]
    assert fig.legends[0].get_title().get_text() == "dimension"
    assert len(fig.axes[0].collections) == 2
    assert fig.axes[0].get_xlim() == pytest.approx(xlim)


def _higgs_results() -> pd.DataFrame:
    capacities = {
        3: {
            "spectral": (0, 174),
            "mlp-1": (6, 181),
            "mlp-2": (5, 181),
            "mlp-3": (4, 161),
        },
        5: {
            "spectral": (0, 435),
            "mlp-1": (14, 421),
            "mlp-2": (10, 411),
            "mlp-3": (9, 451),
        },
    }
    rows = []
    for train_size, seed, dim in product((2**14, 2**18), range(3), capacities):
        rows.append(
            {
                "train_size": train_size,
                "model": "linear",
                "dim": 0,
                "width": 0,
                "num_parameters": 29,
                "test_logloss": 0.6 - train_size / 10**7 + seed / 1000,
                "test_brier": 0.2 - train_size / 10**8 + seed / 1000,
            }
        )
        for model, (width, num_parameters) in capacities[dim].items():
            rows.append(
                {
                    "train_size": train_size,
                    "model": model,
                    "dim": dim,
                    "width": width,
                    "num_parameters": num_parameters,
                    "test_logloss": 0.5 + dim / 100 + seed / 1000,
                    "test_brier": 0.15 + dim / 1000 + seed / 1000,
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def _higgs_deviation_results(
    *,
    noise_levels: tuple[float, ...] = (0.5,),
    dimensions: tuple[int, ...] = (3, 7),
    features: tuple[tuple[int, str], ...] = ((0, "lepton pT"), (1, "lepton η")),
    magnitude_bins: int = 4,
    ratio_bins: int = 4,
) -> pd.DataFrame:
    rows = []
    for (
        noise_level,
        dim,
        init_seed,
        (feature_index, feature_name),
        magnitude_bin_index,
    ) in product(
        noise_levels,
        dimensions,
        range(2),
        features,
        range(magnitude_bins),
    ):
        counts = np.arange(1, ratio_bins + 1) * (magnitude_bin_index + 1)
        row = {
            "dim": dim,
            "data_seed": 0,
            "init_seed": init_seed,
            "noise_level": noise_level,
            "feature_index": feature_index,
            "feature_name": feature_name,
            "magnitude_bin_index": magnitude_bin_index,
            "magnitude_left": noise_level * magnitude_bin_index / magnitude_bins,
            "magnitude_right": noise_level
            * (magnitude_bin_index + 1)
            / magnitude_bins,
            "total_count": int(counts.sum() + 1),
            "zero_bound_count": 1,
        }
        row.update(
            {
                f"ratio_bin_{bin_index:03d}_count": int(count)
                for bin_index, count in enumerate(counts)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_plot_higgs_deviation_shell_grid_uses_disjoint_ranges_and_feature_order():
    fig = plot_higgs_deviation_shell_grid(
        _higgs_deviation_results(), shell_count=4
    )
    ax = fig.axes[0]

    assert len(fig.axes) == 8
    assert [axis.get_title() for axis in fig.axes[:4]] == [
        "|δ| ∈ [0, 0.125)",
        "|δ| ∈ [0.125, 0.25)",
        "|δ| ∈ [0.25, 0.375)",
        "|δ| ∈ [0.375, 0.5]",
    ]
    assert ax.get_xlim() == pytest.approx((0, 1))
    assert fig._supxlabel.get_text() == "Deviation ratio  |Δf| / (|δ| ‖Aⱼ‖₂)"
    assert fig._supylabel.get_text() == "Feature"
    assert len(ax.lines) == 4  # Two dimensions, y=0, and x=1.
    assert _legend_labels(fig) == ["3", "7"]
    assert fig.legends[0].get_title().get_text() == "Matrix dimension"
    assert fig.get_figheight() == pytest.approx(24 / 25.4)
    assert fig.axes[0].get_ylabel() == "lepton pT"
    assert fig.axes[4].get_ylabel() == "lepton η"
    assert fig.axes[0].yaxis.label.get_fontsize() == 8
    for cell in fig.axes:
        for histogram in cell.lines[:2]:
            assert histogram.get_ydata()[:-1].sum() == pytest.approx(1)


def test_plot_higgs_deviation_shell_grid_aggregates_shells_without_cumulation():
    rows = []
    for magnitude_bin_index in range(4):
        row = {
            "dim": 3,
            "data_seed": 0,
            "init_seed": 0,
            "noise_level": 0.5,
            "feature_index": 0,
            "feature_name": "lepton pT",
            "magnitude_bin_index": magnitude_bin_index,
            "magnitude_left": magnitude_bin_index / 8,
            "magnitude_right": (magnitude_bin_index + 1) / 8,
            "total_count": 1,
            "zero_bound_count": 0,
        }
        row.update(
            {
                f"ratio_bin_{bin_index:03d}_count": int(
                    bin_index == magnitude_bin_index
                )
                for bin_index in range(4)
            }
        )
        rows.append(row)

    fig = plot_higgs_deviation_shell_grid(pd.DataFrame(rows), shell_count=2)
    first_shell = fig.axes[0].lines[0].get_ydata()[:-1]
    second_shell = fig.axes[1].lines[0].get_ydata()[:-1]

    assert first_shell.tolist() == pytest.approx([0.5, 0.5, 0, 0])
    assert second_shell.tolist() == pytest.approx([0, 0, 0.5, 0.5])
    assert first_shell.sum() == pytest.approx(1)
    assert second_shell.sum() == pytest.approx(1)


def test_plot_higgs_deviation_shell_grid_uses_raw_probabilities_and_cell_limits():
    rows = []
    feature_counts = {
        (0, "sharp"): ((9, 1), (1, 1)),
        (1, "flat"): ((1, 1), (1, 1)),
    }
    for (feature_index, feature_name), shell_counts in feature_counts.items():
        for magnitude_bin_index, counts in enumerate(shell_counts):
            rows.append(
                {
                    "dim": 3,
                    "data_seed": 0,
                    "init_seed": 0,
                    "noise_level": 0.5,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "magnitude_bin_index": magnitude_bin_index,
                    "magnitude_left": magnitude_bin_index / 4,
                    "magnitude_right": (magnitude_bin_index + 1) / 4,
                    "total_count": sum(counts),
                    "zero_bound_count": 0,
                    "ratio_bin_000_count": counts[0],
                    "ratio_bin_001_count": counts[1],
                }
            )

    fig = plot_higgs_deviation_shell_grid(pd.DataFrame(rows), shell_count=2)
    sharp_first, sharp_second, flat_first, flat_second = fig.axes
    assert sharp_first.lines[0].get_ydata()[:-1] == pytest.approx([0.9, 0.1])
    assert sharp_second.lines[0].get_ydata()[:-1] == pytest.approx([0.5, 0.5])
    assert flat_first.lines[0].get_ydata()[:-1] == pytest.approx([0.5, 0.5])
    assert flat_second.lines[0].get_ydata()[:-1] == pytest.approx([0.5, 0.5])
    assert sharp_first.get_ylim()[1] == pytest.approx(0.9 / 0.85)
    assert sharp_second.get_ylim()[1] == pytest.approx(0.5 / 0.85)


def test_plot_higgs_deviation_shell_grid_weights_runs_equally():
    rows = []
    for init_seed, (counts, total_count) in enumerate(
        (((8, 2), 10), ((0, 100), 100))
    ):
        rows.append(
            {
                "dim": 3,
                "data_seed": 0,
                "init_seed": init_seed,
                "noise_level": 0.5,
                "feature_index": 0,
                "feature_name": "lepton pT",
                "magnitude_bin_index": 0,
                "magnitude_left": 0,
                "magnitude_right": 0.5,
                "total_count": total_count,
                "zero_bound_count": 0,
                "ratio_bin_000_count": counts[0],
                "ratio_bin_001_count": counts[1],
            }
        )

    fig = plot_higgs_deviation_shell_grid(pd.DataFrame(rows), shell_count=1)
    heights = fig.axes[0].lines[0].get_ydata()[:-1]

    # Mean seed probabilities are (0.4, 0.6), not pooled counts (8, 102).
    assert heights[0] / heights[1] == pytest.approx(2 / 3)


def test_plot_higgs_deviation_shell_grid_rejects_mixed_noise_and_bad_shell_count():
    with pytest.raises(ValueError, match="one noise_level"):
        plot_higgs_deviation_shell_grid(
            _higgs_deviation_results(noise_levels=(0.25, 0.5))
        )
    with pytest.raises(ValueError, match="divisor"):
        plot_higgs_deviation_shell_grid(
            _higgs_deviation_results(), shell_count=3
        )
    with pytest.raises(ValueError, match="feature_row_height_mm"):
        plot_higgs_deviation_shell_grid(
            _higgs_deviation_results(), feature_row_height_mm=0
        )


def test_plot_higgs_models_by_dimension_facets_and_annotates_capacity():
    fig = plot_higgs_models_by_dimension(_higgs_results())

    assert [ax.get_title() for ax in fig.axes] == [
        "dim=3 · Spectral 174p\nMLP 1×6 (181p) · 2×5 (181p)\nMLP 3×4 (161p)",
        "dim=5 · Spectral 435p\nMLP 1×14 (421p) · 2×10 (411p)\nMLP 3×9 (451p)",
    ]
    assert _legend_labels(fig) == [
        "Linear",
        "Spectral",
        "MLP-1",
        "MLP-2",
        "MLP-3",
    ]
    assert fig.legends[0].get_title().get_text() == "model"
    assert all(
        sum(line.get_label().startswith("_child") for line in ax.lines) == 5
        for ax in fig.axes
    )
    assert all(len(ax.collections) == 5 for ax in fig.axes)
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert all(
        ax.get_xlabel() == "training samples processed by optimizer"
        for ax in fig.axes
    )
    assert fig.axes[0].get_ylabel() == "test log loss ↓"


def test_plot_higgs_capacity_annotation_uses_recorded_values():
    results = _higgs_results()
    results.loc[
        (results["model"] == "mlp-1") & (results["dim"] == 3),
        ["width", "num_parameters"],
    ] = (7, 211)

    fig = plot_higgs_models_by_dimension(results)

    title = fig.axes[0].get_title()
    assert "1×7 (211p)" in title


def test_plot_higgs_models_supports_brier_score():
    fig = plot_higgs_models_by_dimension(_higgs_results(), metric="brier")

    assert fig.axes[0].get_ylabel() == "test Brier score ↓"
    assert fig._suptitle.get_text() == (
        "HIGGS test Brier score: matched model families"
    )


def test_plot_higgs_spectral_dimensions_uses_one_axis():
    fig = plot_higgs_spectral_dimensions(_higgs_results())

    assert len(fig.axes) == 1
    assert _legend_labels(fig) == ["3", "5"]
    assert fig.legends[0].get_title().get_text() == "dimension"
    assert len(fig.axes[0].collections) == 2
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_xlabel() == "training samples processed by optimizer"
    assert fig.axes[0].get_ylabel() == "test log loss ↓"
    assert fig._suptitle.get_text() == (
        "HIGGS test log loss: spectral neurons across dimensions"
    )


def test_plot_higgs_spectral_dimensions_supports_brier_and_zoom():
    xlim = (2**14, 2**18)
    fig = plot_higgs_spectral_dimensions(
        _higgs_results(), metric="brier", xlim=xlim
    )

    assert fig.axes[0].get_ylabel() == "test Brier score ↓"
    assert fig.axes[0].get_xlim() == pytest.approx(xlim)


def test_plot_higgs_spectral_dimensions_supports_more_than_four_dimensions():
    dimensions = [3, 5, 7, 9, 11]
    base = _higgs_results().loc[
        lambda df: (df["model"] == "spectral") & (df["dim"] == 3)
    ]
    results = pd.concat(
        (base.assign(dim=dim) for dim in dimensions),
        ignore_index=True,
    )

    fig = plot_higgs_spectral_dimensions(results)

    assert _legend_labels(fig) == list(map(str, dimensions))


def test_plot_higgs_models_rejects_inconsistent_capacity():
    results = _higgs_results()
    row = results.loc[(results["model"] == "mlp-1") & (results["dim"] == 3)].iloc[0]
    results = pd.concat(
        (results, pd.DataFrame([row.to_dict() | {"num_parameters": 999}])),
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="one recorded capacity"):
        plot_higgs_models_by_dimension(results)
