from .criteo import (
    plot_criteo_fm_dimensions,
    plot_criteo_models_by_dimension,
    plot_criteo_spectral_comparison,
    plot_criteo_spectral_dimensions,
)
from .higgs import (
    plot_higgs_models_by_dimension,
    plot_higgs_spectral_dimensions,
)
from .robustness import plot_higgs_deviation_shell_grid
from .synthetic import (
    plot_general_scaling,
    plot_monotone_scaling,
)
from .targets import (
    plot_bivariate_target_gallery,
    plot_target_gallery,
)

__all__ = [
    "plot_bivariate_target_gallery",
    "plot_criteo_fm_dimensions",
    "plot_criteo_models_by_dimension",
    "plot_criteo_spectral_comparison",
    "plot_criteo_spectral_dimensions",
    "plot_general_scaling",
    "plot_higgs_deviation_shell_grid",
    "plot_higgs_models_by_dimension",
    "plot_higgs_spectral_dimensions",
    "plot_monotone_scaling",
    "plot_target_gallery",
]
