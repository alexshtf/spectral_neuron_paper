from itertools import product

import pandas as pd
import pytest

from paper.plotting import (
    plot_criteo_fm_dimensions,
    plot_criteo_models_by_dimension,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
)


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
