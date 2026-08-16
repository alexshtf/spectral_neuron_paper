from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from paper.models import matched_fm_rank

from ._common import (
    BINARY_METRIC_LABELS,
    TRAIN_SIZE_LABEL,
    CurveStyle,
    _dimension_curve_styles,
    _finish_scaling_grid,
    _summary_curve_grid,
    _summary_metric_columns,
)

_CRITEO_MODEL_STYLES = {
    "linear-bucketed": CurveStyle("Linear (bucketed)", "#555555", "o"),
    "linear-continuous": CurveStyle(
        "Linear (continuous)", "#CC78BC", "P", (2, 2)
    ),
    "fm": CurveStyle("FM", "#0173B2", "s"),
    "spectral-bucketed": CurveStyle(
        "Spectral (bucketed)", "#DE8F05", "^", (4, 2)
    ),
    "spectral-continuous": CurveStyle(
        "Spectral (continuous)", "#029E73", "D"
    ),
}

type CriteoSpectralVariant = Literal[
    "spectral-bucketed",
    "spectral-continuous",
]


def _check_criteo_summary(summary: pd.DataFrame, metric: str) -> None:
    _summary_metric_columns(summary, metric)
    missing = {"model", "dim"}.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")


def _spectral_dimensions(summary: pd.DataFrame) -> list[int]:
    values = summary.loc[
        summary["model"].isin(("spectral-bucketed", "spectral-continuous")),
        "dim",
    ].unique()
    dimensions = sorted(map(int, values))
    if not dimensions:
        raise ValueError("summary contains no spectral models")
    return dimensions


def _label_models(summary: pd.DataFrame) -> pd.DataFrame:
    labeled = summary.copy()
    labeled["model_label"] = labeled["model"].map(
        {model: style.label for model, style in _CRITEO_MODEL_STYLES.items()}
    )
    return labeled


def _require_models(summary: pd.DataFrame, models: set[str]) -> None:
    missing = models.difference(summary["model"].unique())
    if missing:
        raise ValueError(f"summary is missing models: {sorted(missing)}")


def _require_dimension_grid(
    summary: pd.DataFrame,
    models: set[str],
    dimensions: list[int],
) -> None:
    observed = set(
        summary[["model", "dim"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected = {
        (model, dimension) for model in models for dimension in dimensions
    }
    if observed != expected:
        raise ValueError("summary has an incomplete model/dimension grid")


def _matched_model_curves(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    models = set(_CRITEO_MODEL_STYLES)
    _require_models(summary, models)
    dimensions = _spectral_dimensions(summary)
    nonlinear_models = {
        "fm",
        "spectral-bucketed",
        "spectral-continuous",
    }
    nonlinear = summary.loc[
        summary["model"].isin(nonlinear_models)
    ].copy()
    _require_dimension_grid(nonlinear, nonlinear_models, dimensions)
    nonlinear["dimension"] = nonlinear["dim"]

    linears = summary.loc[
        summary["model"].isin(("linear-bucketed", "linear-continuous"))
    ].merge(pd.DataFrame({"dimension": dimensions}), how="cross")
    return _label_models(pd.concat((linears, nonlinear))), dimensions


def _spectral_comparison_curves(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    models = {"spectral-bucketed", "spectral-continuous"}
    _require_models(summary, models)
    spectral = summary.loc[
        summary["model"].isin(models)
    ]
    dimensions = _spectral_dimensions(spectral)
    _require_dimension_grid(spectral, models, dimensions)
    return _label_models(spectral), dimensions


def _spectral_dimension_curves(
    summary: pd.DataFrame, variant: CriteoSpectralVariant
) -> tuple[pd.DataFrame, list[int]]:
    spectral = summary.loc[summary["model"] == variant]
    return spectral, _spectral_dimensions(spectral)


def _fm_dimension_curves(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    fm = summary.loc[summary["model"] == "fm"].copy()
    fm["rank"] = fm["dim"].map(matched_fm_rank)
    ranks = sorted(map(int, fm["rank"].unique()))
    if not ranks:
        raise ValueError("summary contains no FM models")
    return fm, ranks


def plot_criteo_models_by_dimension(
    summary: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare parameter-matched models in one facet per spectral dimension."""
    _check_criteo_summary(summary, metric)
    faceted, dimensions = _matched_model_curves(summary)
    grid = _summary_curve_grid(
        faceted,
        metric=metric,
        by="model_label",
        styles=tuple(_CRITEO_MODEL_STYLES.values()),
        col="dimension",
        col_order=dimensions,
    )
    return _finish_scaling_grid(
        grid,
        title=f"Criteo {BINARY_METRIC_LABELS[metric]}: matched models",
        x_label=TRAIN_SIZE_LABEL,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        legend_title="model",
        facet_title="dim={col_name}",
    )


def plot_criteo_spectral_comparison(
    summary: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare bucketed and continuous spectral preprocessing by dimension."""
    _check_criteo_summary(summary, metric)
    spectral, dimensions = _spectral_comparison_curves(summary)
    variants = ("spectral-bucketed", "spectral-continuous")
    grid = _summary_curve_grid(
        spectral,
        metric=metric,
        by="model_label",
        styles=tuple(_CRITEO_MODEL_STYLES[variant] for variant in variants),
        col="dim",
        col_order=dimensions,
    )
    return _finish_scaling_grid(
        grid,
        title=f"Criteo {BINARY_METRIC_LABELS[metric]}: spectral preprocessing",
        x_label=TRAIN_SIZE_LABEL,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        legend_title="model",
        facet_title="dim={col_name}",
    )


def plot_criteo_spectral_dimensions(
    summary: pd.DataFrame,
    variant: CriteoSpectralVariant,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare all dimensions of one spectral preprocessing variant."""
    _check_criteo_summary(summary, metric)
    spectral, dimensions = _spectral_dimension_curves(summary, variant)
    grid = _summary_curve_grid(
        spectral,
        metric=metric,
        by="dim",
        styles=_dimension_curve_styles(dimensions),
    )
    return _finish_scaling_grid(
        grid,
        title=(
            f"Criteo {BINARY_METRIC_LABELS[metric]}: "
            f"{_CRITEO_MODEL_STYLES[variant].label} across dimensions"
        ),
        x_label=TRAIN_SIZE_LABEL,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        legend_title="dimension",
        xlim=xlim,
    )


def plot_criteo_fm_dimensions(
    summary: pd.DataFrame,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare FM embedding dimensions."""
    _check_criteo_summary(summary, metric)
    fm, ranks = _fm_dimension_curves(summary)
    grid = _summary_curve_grid(
        fm,
        metric=metric,
        by="rank",
        styles=_dimension_curve_styles(ranks),
    )
    return _finish_scaling_grid(
        grid,
        title=(
            f"Criteo {BINARY_METRIC_LABELS[metric]}: "
            "FM across embedding dimensions"
        ),
        x_label=TRAIN_SIZE_LABEL,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        legend_title="dimension",
        xlim=xlim,
    )
