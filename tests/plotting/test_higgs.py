import pandas as pd
import pytest

from paper.plotting import (
    plot_higgs_models_by_dimension,
    plot_higgs_spectral_dimensions,
)
from paper.plotting.higgs import _higgs_capacity_title, _matched_model_curves


def _higgs_summary() -> pd.DataFrame:
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
    for train_size in (2**14, 2**18):
        logloss = 0.6 - train_size / 10**7
        brier = 0.2 - train_size / 10**8
        rows.append(
            {
                "train_size": train_size,
                "model": "linear",
                "dim": 0,
                "width": 0,
                "num_parameters": 29,
                "median_test_logloss": logloss,
                "q25_test_logloss": logloss - 0.01,
                "q75_test_logloss": logloss + 0.01,
                "median_test_brier": brier,
                "q25_test_brier": brier - 0.01,
                "q75_test_brier": brier + 0.01,
                "n": 3,
            }
        )
        for dim, models in capacities.items():
            for model, (width, num_parameters) in models.items():
                logloss = 0.5 + dim / 100 - train_size / 10**8
                brier = 0.15 + dim / 1000 - train_size / 10**9
                rows.append(
                    {
                        "train_size": train_size,
                        "model": model,
                        "dim": dim,
                        "width": width,
                        "num_parameters": num_parameters,
                        "median_test_logloss": logloss,
                        "q25_test_logloss": logloss - 0.01,
                        "q75_test_logloss": logloss + 0.01,
                        "median_test_brier": brier,
                        "q25_test_brier": brier - 0.01,
                        "q75_test_brier": brier + 0.01,
                        "n": 3,
                    }
                )
    return pd.DataFrame(rows)


def _legend_labels(fig) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_matched_model_curves_replicate_only_the_linear_baseline():
    faceted = _matched_model_curves(_higgs_summary(), [3, 5])

    linears = faceted.loc[faceted["model"] == "linear"]
    assert set(linears["dimension"]) == {3, 5}
    assert len(linears) == 4
    assert len(faceted) == 20


def test_capacity_title_uses_recorded_summary_metadata():
    summary = _higgs_summary()
    summary.loc[
        (summary["model"] == "mlp-1") & (summary["dim"] == 3),
        ["width", "num_parameters"],
    ] = (7, 211)

    assert _higgs_capacity_title(summary, 3) == (
        "dim=3 · Spectral 174p\n"
        "MLP 1×7 (211p) · 2×5 (181p)\n"
        "MLP 3×4 (161p)"
    )


def test_plot_higgs_models_preserves_facets_capacity_and_styles():
    fig = plot_higgs_models_by_dimension(_higgs_summary(), metric="brier")

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
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert fig.axes[0].get_ylabel() == "test Brier score ↓"
    assert fig.get_suptitle() == "HIGGS test Brier score: matched model families"


def test_plot_higgs_spectral_dimensions_supports_brier_and_zoom():
    xlim = (2**14, 2**18)
    fig = plot_higgs_spectral_dimensions(
        _higgs_summary(), metric="brier", xlim=xlim
    )

    assert len(fig.axes) == 1
    assert _legend_labels(fig) == ["3", "5"]
    assert fig.legends[0].get_title().get_text() == "dimension"
    assert fig.axes[0].get_ylabel() == "test Brier score ↓"
    assert fig.axes[0].get_xlim() == pytest.approx(xlim)
    assert fig.get_suptitle() == (
        "HIGGS test Brier score: spectral neurons across dimensions"
    )


def test_plot_higgs_spectral_dimensions_scales_style_sequence():
    dimensions = [3, 5, 7, 9, 11]
    base = _higgs_summary().loc[
        lambda df: (df["model"] == "spectral") & (df["dim"] == 3)
    ]
    summary = pd.concat(
        (base.assign(dim=dim) for dim in dimensions),
        ignore_index=True,
    )

    fig = plot_higgs_spectral_dimensions(summary)

    assert _legend_labels(fig) == list(map(str, dimensions))
