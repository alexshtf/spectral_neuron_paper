from itertools import product

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paper.plotting import (
    plot_bivariate_target_gallery,
    plot_criteo_models_by_dimension,
    plot_criteo_fm_dimensions,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
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
    budget: int,
    target_kind: str = "monotone",
    noise_std: float = 0.0,
) -> dict:
    return {
        "target_kind": target_kind,
        "complexity": complexity,
        "noise_std": noise_std,
        "model": model,
        "dim": dim,
        "budget": budget,
        "median_test_rmse": 1.0 / budget,
        "q25_test_rmse": 0.8 / budget,
        "q75_test_rmse": 1.2 / budget,
    }


def test_plot_scaling_pairs_monotone_models_by_dimension():
    summary = pd.DataFrame(
        [
            _row(complexity=complexity, dim=dim, model=model, budget=budget)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in ("unconstrained", "monotone")
            for budget in (1, 2)
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


def test_plot_scaling_styles_models_and_dimensions():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=dim,
                model=model,
                budget=budget,
                target_kind="general",
            )
            for dim in (5, 9)
            for model in ("unconstrained", "monotone")
            for budget in (1, 2)
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
                budget=budget,
                target_kind=target_kind,
            )
            for target_kind in ("general", "monotone")
            for budget in (1, 2)
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
                budget=budget,
                target_kind=target_kind,
                noise_std=noise_std,
            )
            for noise_std in (0.0, 0.1)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in models
            for budget in (1, 2)
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
        ("linear", 0),
        ("linear-new", 0),
        *(
            (model, dim)
            for dim in (3, 5)
            for model in ("fm", "spectral-old", "spectral-new")
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
        "Linear",
        "Linear-new",
        "FM",
        "Spectral-old",
        "Spectral-new",
    ]
    assert fig.legends[0].get_title().get_text() == "model"
    assert len(fig.axes[1].lines) == 5
    assert len(fig.axes[1].collections) == 5
    assert all(ax.get_xscale() == "log" for ax in fig.axes)


def test_plot_criteo_spectral_comparison_facets_dimensions():
    fig = plot_criteo_spectral_comparison(_criteo_results())
    assert [ax.get_title() for ax in fig.axes] == ["dim=3", "dim=5"]
    assert _legend_labels(fig) == ["Spectral-old", "Spectral-new"]
    assert fig.legends[0].get_title().get_text() == "model"
    assert len(fig.axes[1].lines) == 2
    assert len(fig.axes[1].collections) == 2


@pytest.mark.parametrize("variant", ["spectral-old", "spectral-new"])
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
