import pandas as pd
from matplotlib.figure import Figure

from ._common import (
    BINARY_METRIC_LABELS,
    TRAIN_SIZE_LABEL,
    CurveStyle,
    _dimension_curve_styles,
    _finish_scaling_grid,
    _summary_curve_grid,
    _summary_metric_columns,
)

_HIGGS_MODEL_STYLES = {
    "linear": CurveStyle("Linear", "#555555", "o"),
    "spectral": CurveStyle("Spectral", "#029E73", "^"),
    "mlp-1": CurveStyle("MLP-1", "#0173B2", "s", (4, 2)),
    "mlp-2": CurveStyle("MLP-2", "#DE8F05", "D", (2, 2)),
    "mlp-3": CurveStyle("MLP-3", "#CC78BC", "P", (4, 2, 1, 2)),
}


def _check_higgs_summary(summary: pd.DataFrame, metric: str) -> list[int]:
    _summary_metric_columns(summary, metric)
    required = {"model", "dim", "width", "num_parameters"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("summary is empty")
    if summary[list(required)].isna().any().any():
        raise ValueError("HIGGS plotting columns must not contain missing values")

    dimensions = sorted(
        map(int, summary.loc[summary["model"] == "spectral", "dim"].unique())
    )
    if not dimensions:
        raise ValueError("summary contains no spectral models")
    return dimensions


def _higgs_capacity_title(summary: pd.DataFrame, dim: int) -> str:
    capacity = (
        summary.loc[
            (summary["dim"] == dim) & (summary["model"] != "linear"),
            ["model", "width", "num_parameters"],
        ]
        .drop_duplicates()
        .set_index("model")
    )
    spectral_parameters = int(capacity.at["spectral", "num_parameters"])
    entries = tuple(
        f"{model.removeprefix('mlp-')}×{int(capacity.at[model, 'width'])} "
        f"({int(capacity.at[model, 'num_parameters']):,}p)"
        for model in ("mlp-1", "mlp-2", "mlp-3")
    )
    return (
        f"dim={dim} · Spectral {spectral_parameters:,}p\n"
        f"MLP {entries[0]} · {entries[1]}\n"
        f"MLP {entries[2]}"
    )


def _matched_model_curves(
    summary: pd.DataFrame, dimensions: list[int]
) -> pd.DataFrame:
    nonlinear = summary.loc[summary["model"] != "linear"].copy()
    nonlinear["dimension"] = nonlinear["dim"].astype(int)
    linears = summary.loc[summary["model"] == "linear"].merge(
        pd.DataFrame({"dimension": dimensions}), how="cross"
    )
    faceted = pd.concat((linears, nonlinear), ignore_index=True)
    faceted["model_label"] = faceted["model"].map(
        {model: style.label for model, style in _HIGGS_MODEL_STYLES.items()}
    )
    return faceted


def _spectral_dimension_curves(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    spectral = summary.loc[summary["model"] == "spectral"]
    dimensions = sorted(map(int, spectral["dim"].unique()))
    if not dimensions:
        raise ValueError("summary contains no spectral models")
    return spectral, dimensions


def plot_higgs_models_by_dimension(
    summary: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare HIGGS model families in one facet per matched dimension."""
    dimensions = _check_higgs_summary(summary, metric)
    faceted = _matched_model_curves(summary, dimensions)
    grid = _summary_curve_grid(
        faceted,
        metric=metric,
        by="model_label",
        styles=tuple(_HIGGS_MODEL_STYLES.values()),
        col="dimension",
        col_order=dimensions,
        height=3.8,
    )
    grid.set_axis_labels(
        TRAIN_SIZE_LABEL,
        f"{BINARY_METRIC_LABELS[metric]} ↓",
    )
    for dim, ax in zip(dimensions, grid.axes.flat, strict=True):
        ax.set_title(_higgs_capacity_title(summary, dim), fontsize="small")
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
    summary: pd.DataFrame,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare all HIGGS spectral dimensions on one scaling axis."""
    _summary_metric_columns(summary, metric)
    spectral, dimensions = _spectral_dimension_curves(summary)
    grid = _summary_curve_grid(
        spectral,
        metric=metric,
        by="dim",
        styles=_dimension_curve_styles(dimensions),
    )
    return _finish_scaling_grid(
        grid,
        title=(
            f"HIGGS {BINARY_METRIC_LABELS[metric]}: "
            "spectral neurons across dimensions"
        ),
        x_label=TRAIN_SIZE_LABEL,
        y_label=f"{BINARY_METRIC_LABELS[metric]} ↓",
        legend_title="dimension",
        xlim=xlim,
    )
