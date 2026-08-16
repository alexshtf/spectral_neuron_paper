from itertools import product

import pandas as pd
import pytest

from paper.plotting import (
    plot_criteo_fm_dimensions,
    plot_criteo_models_by_dimension,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
)
from paper.plotting.criteo import (
    _fm_dimension_curves,
    _matched_model_curves,
    _spectral_comparison_curves,
)


def _criteo_summary() -> pd.DataFrame:
    models = (
        ("linear-bucketed", 0),
        ("linear-continuous", 0),
        *(
            (model, dim)
            for dim in (3, 5)
            for model in ("fm", "spectral-bucketed", "spectral-continuous")
        ),
    )
    rows = []
    for train_size, (model, dim) in product((2**14, 2**18), models):
        median = 0.5 + dim / 100 - train_size / 10**8
        rows.append(
            {
                "train_size": train_size,
                "model": model,
                "dim": dim,
                "median_test_logloss": median,
                "q25_test_logloss": median - 0.01,
                "q75_test_logloss": median + 0.01,
                "n": 3,
            }
        )
    return pd.DataFrame(rows)


def _legend_labels(fig) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_matched_model_curves_replicate_only_linear_baselines():
    faceted, dimensions = _matched_model_curves(_criteo_summary())

    assert dimensions == [3, 5]
    linears = faceted.loc[faceted["model"].str.startswith("linear")]
    assert set(linears["dimension"]) == {3, 5}
    assert len(linears) == 8
    assert len(faceted) == 20

    with pytest.raises(ValueError, match="missing models"):
        _matched_model_curves(
            _criteo_summary().loc[lambda frame: frame["model"] != "fm"]
        )


def test_plot_criteo_models_by_dimension_preserves_facets_and_styles():
    fig = plot_criteo_models_by_dimension(_criteo_summary())

    assert [ax.get_title() for ax in fig.axes] == ["dim=3", "dim=5"]
    assert _legend_labels(fig) == [
        "Linear (bucketed)",
        "Linear (continuous)",
        "FM",
        "Spectral (bucketed)",
        "Spectral (continuous)",
    ]
    assert fig.legends[0].get_title().get_text() == "model"
    assert all(ax.get_xscale() == "log" for ax in fig.axes)


def test_spectral_comparison_filters_and_facets_by_dimension():
    spectral, dimensions = _spectral_comparison_curves(_criteo_summary())
    assert dimensions == [3, 5]
    assert set(spectral["model"]) == {
        "spectral-bucketed",
        "spectral-continuous",
    }

    with pytest.raises(ValueError, match="missing models"):
        _spectral_comparison_curves(
            _criteo_summary().loc[
                lambda frame: frame["model"] != "spectral-continuous"
            ]
        )

    fig = plot_criteo_spectral_comparison(_criteo_summary())
    assert [ax.get_title() for ax in fig.axes] == ["dim=3", "dim=5"]
    assert _legend_labels(fig) == [
        "Spectral (bucketed)",
        "Spectral (continuous)",
    ]


@pytest.mark.parametrize(
    "variant", ["spectral-bucketed", "spectral-continuous"]
)
def test_plot_criteo_spectral_dimensions_supports_zoom(variant):
    xlim = (2**14, 2**18)
    fig = plot_criteo_spectral_dimensions(
        _criteo_summary(), variant, xlim=xlim
    )

    assert len(fig.axes) == 1
    assert _legend_labels(fig) == ["3", "5"]
    assert fig.legends[0].get_title().get_text() == "dimension"
    assert fig.axes[0].get_xlim() == pytest.approx(xlim)


def test_fm_curves_use_parameter_matched_ranks():
    fm, ranks = _fm_dimension_curves(_criteo_summary())
    assert ranks == [5, 14]
    assert set(fm["rank"]) == {5, 14}

    fig = plot_criteo_fm_dimensions(_criteo_summary())
    assert _legend_labels(fig) == ["5", "14"]
