from itertools import product

import pandas as pd
import pytest

from paper.plotting import (
    plot_higgs_models_by_dimension,
    plot_higgs_spectral_dimensions,
)


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


def _legend_labels(fig) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


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
