from ._common import (
    BINARY_METRIC_LABELS,
    TRAIN_SIZE_LABEL,
    FigureContainer,
)
from .criteo import (
    CRITEO_MODEL_COLORS,
    CRITEO_MODEL_DASHES,
    CRITEO_MODEL_LABELS,
    CRITEO_MODEL_MARKERS,
    CriteoSpectralVariant,
    plot_criteo_fm_dimensions,
    plot_criteo_models_by_dimension,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
)
from .higgs import (
    HIGGS_MODEL_COLORS,
    HIGGS_MODEL_DASHES,
    HIGGS_MODEL_LABELS,
    HIGGS_MODEL_MARKERS,
    plot_higgs_models_by_dimension,
    plot_higgs_spectral_dimensions,
)
from .robustness import (
    HIGGS_DIMENSION_LINESTYLES,
    plot_higgs_deviation_shell_grid,
)
from .synthetic import (
    MODEL_LINESTYLES,
    SCALING_COLUMNS,
    SYNTHETIC_TRAIN_SIZE_LABEL,
    plot_scaling,
)
from .targets import (
    plot_bivariate_target_gallery,
    plot_target_gallery,
)

__all__ = [
    "BINARY_METRIC_LABELS",
    "CRITEO_MODEL_COLORS",
    "CRITEO_MODEL_DASHES",
    "CRITEO_MODEL_LABELS",
    "CRITEO_MODEL_MARKERS",
    "CriteoSpectralVariant",
    "FigureContainer",
    "HIGGS_DIMENSION_LINESTYLES",
    "HIGGS_MODEL_COLORS",
    "HIGGS_MODEL_DASHES",
    "HIGGS_MODEL_LABELS",
    "HIGGS_MODEL_MARKERS",
    "MODEL_LINESTYLES",
    "SCALING_COLUMNS",
    "SYNTHETIC_TRAIN_SIZE_LABEL",
    "TRAIN_SIZE_LABEL",
    "plot_bivariate_target_gallery",
    "plot_criteo_fm_dimensions",
    "plot_criteo_models_by_dimension",
    "plot_criteo_spectral_comparison",
    "plot_criteo_spectral_dimensions",
    "plot_higgs_deviation_shell_grid",
    "plot_higgs_models_by_dimension",
    "plot_higgs_spectral_dimensions",
    "plot_scaling",
    "plot_target_gallery",
]
