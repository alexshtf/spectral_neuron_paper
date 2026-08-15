from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from ._common import (
    BINARY_METRIC_LABELS,
    TRAIN_SIZE_LABEL,
    _binary_relplot,
    _dimension_styles,
)

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
