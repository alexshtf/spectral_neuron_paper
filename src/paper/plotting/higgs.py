import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ._common import (
    BINARY_METRIC_LABELS,
    TRAIN_SIZE_LABEL,
    _binary_relplot,
    _dimension_styles,
)

HIGGS_MODEL_LABELS = {
    "linear": "Linear",
    "spectral": "Spectral",
    "mlp-1": "MLP-1",
    "mlp-2": "MLP-2",
    "mlp-3": "MLP-3",
}

HIGGS_MODEL_COLORS = {
    "Linear": "#555555",
    "Spectral": "#029E73",
    "MLP-1": "#0173B2",
    "MLP-2": "#DE8F05",
    "MLP-3": "#CC78BC",
}

HIGGS_MODEL_MARKERS = {
    "Linear": "o",
    "Spectral": "^",
    "MLP-1": "s",
    "MLP-2": "D",
    "MLP-3": "P",
}

HIGGS_MODEL_DASHES = {
    "Linear": "",
    "Spectral": "",
    "MLP-1": (4, 2),
    "MLP-2": (2, 2),
    "MLP-3": (4, 2, 1, 2),
}


def _check_higgs_metric(results: pd.DataFrame, metric: str) -> str:
    if metric not in BINARY_METRIC_LABELS:
        raise ValueError(
            f"metric must be one of {sorted(BINARY_METRIC_LABELS)}; got {metric!r}"
        )

    value = f"test_{metric}"
    required = {"train_size", "model", "dim", value}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results are empty")
    if results[list(required)].isna().any().any():
        raise ValueError("HIGGS plotting columns must not contain missing values")
    return value


def _check_higgs_results(results: pd.DataFrame, metric: str) -> tuple[str, list[int]]:
    value = _check_higgs_metric(results, metric)
    capacity_columns = {"width", "num_parameters"}
    missing = capacity_columns.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    dimensions = sorted(
        map(int, results.loc[results["model"] == "spectral", "dim"].unique())
    )
    if not dimensions:
        raise ValueError("results contain no spectral models")
    return value, dimensions


def _higgs_capacity_title(results: pd.DataFrame, dim: int) -> str:
    capacity = (
        results.loc[
            (results["dim"] == dim) & (results["model"] != "linear"),
            ["model", "width", "num_parameters"],
        ]
        .drop_duplicates()
        .set_index("model")
    )
    spectral_parameters = int(capacity.at["spectral", "num_parameters"])
    mlp_models = ("mlp-1", "mlp-2", "mlp-3")
    entries = tuple(
        f"{model.removeprefix('mlp-')}×{int(capacity.at[model, 'width'])} "
        f"({int(capacity.at[model, 'num_parameters']):,}p)"
        for model in mlp_models
    )
    return (
        f"dim={dim} · Spectral {spectral_parameters:,}p\n"
        f"MLP {entries[0]} · {entries[1]}\n"
        f"MLP {entries[2]}"
    )


def plot_higgs_models_by_dimension(
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
) -> Figure:
    """Compare HIGGS model families in one facet per matched dimension."""
    value, dimensions = _check_higgs_results(results, metric)

    nonlinear = results.loc[results["model"] != "linear"].copy()
    nonlinear["dimension"] = nonlinear["dim"].astype(int)
    linears = results.loc[results["model"] == "linear"].merge(
        pd.DataFrame({"dimension": dimensions}), how="cross"
    )
    faceted = pd.concat((linears, nonlinear), ignore_index=True)
    faceted["model_label"] = faceted["model"].map(HIGGS_MODEL_LABELS)
    model_order = list(HIGGS_MODEL_LABELS.values())

    grid = sns.relplot(
        data=faceted,
        x="train_size",
        y=value,
        hue="model_label",
        style="model_label",
        hue_order=model_order,
        style_order=model_order,
        palette=HIGGS_MODEL_COLORS,
        markers=HIGGS_MODEL_MARKERS,
        dashes=HIGGS_MODEL_DASHES,
        col="dimension",
        col_order=dimensions,
        col_wrap=2,
        kind="line",
        estimator=np.median,
        errorbar=("pi", 50),
        err_kws={"alpha": 0.15},
        linewidth=2,
        height=3.8,
        aspect=1.25,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels(
        TRAIN_SIZE_LABEL,
        f"{BINARY_METRIC_LABELS[metric]} ↓",
    )
    for dim, ax in zip(dimensions, grid.axes.flat):
        ax.set_title(_higgs_capacity_title(results, dim), fontsize="small")
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
    results: pd.DataFrame,
    *,
    metric: str = "logloss",
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compare all HIGGS spectral dimensions on one scaling axis."""
    value = _check_higgs_metric(results, metric)
    spectral = results.loc[results["model"] == "spectral"]
    dimensions = sorted(map(int, spectral["dim"].unique()))
    if not dimensions:
        raise ValueError("results contain no spectral models")

    palette, markers = _dimension_styles(dimensions)
    return _binary_relplot(
        spectral,
        value=value,
        metric=metric,
        title=(
            f"HIGGS {BINARY_METRIC_LABELS[metric]}: "
            "spectral neurons across dimensions"
        ),
        x_label=TRAIN_SIZE_LABEL,
        by="dim",
        hue_order=dimensions,
        palette=palette,
        markers=markers,
        dashes=False,
        legend_title="dimension",
        xlim=xlim,
    )
