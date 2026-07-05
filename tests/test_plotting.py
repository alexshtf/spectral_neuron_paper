import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from paper.plotting import plot_scaling


def _row(
    *,
    complexity: int,
    dim: int,
    model: str,
    budget: int,
    target_kind: str = "monotone",
) -> dict:
    return {
        "target_kind": target_kind,
        "complexity": complexity,
        "noise_std": 0.0,
        "model": model,
        "dim": dim,
        "eig_idx": dim // 2,
        "budget": budget,
        "selected_lr": 0.01,
        "median_test_rmse": 1.0 / budget,
        "q25_test_rmse": 0.8 / budget,
        "q75_test_rmse": 1.2 / budget,
        "mean_test_rmse": 1.0 / budget,
        "n": 3,
    }


def test_plot_scaling_pairs_monotone_models_by_dimension():
    summary = pd.DataFrame(
        [
            _row(complexity=complexity, dim=dim, model=model, budget=budget)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in ("unconstrained", "monotone")
            for budget in (1, 2)
        ]
    )

    fig = plot_scaling(summary)
    try:
        axes = [ax for ax in fig.axes if ax.get_visible()]

        assert len(axes) == 4
        assert {ax.get_title() for ax in axes} == {
            "complexity=5, dim=3",
            "complexity=5, dim=5",
            "complexity=10, dim=3",
            "complexity=10, dim=5",
        }
        assert all(len(ax.lines) == 2 for ax in axes)
    finally:
        plt.close(fig)


def test_plot_scaling_styles_models_and_dimensions():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=dim,
                model=model,
                budget=budget,
                target_kind="general",
            )
            for dim in (5, 9)
            for model in ("unconstrained", "monotone")
            for budget in (1, 2)
        ]
    )

    fig = plot_scaling(summary)
    try:
        ax = fig.axes[0]
        lines = {line.get_label(): line for line in ax.lines}

        assert (
            lines["dim=5, unconstrained"].get_color()
            == lines["dim=5, monotone"].get_color()
        )
        assert (
            lines["dim=9, unconstrained"].get_color()
            == lines["dim=9, monotone"].get_color()
        )
        assert (
            lines["dim=5, unconstrained"].get_color()
            != lines["dim=9, unconstrained"].get_color()
        )
        assert lines["dim=5, unconstrained"].get_linestyle() == "-"
        assert lines["dim=5, monotone"].get_linestyle() == "--"
    finally:
        plt.close(fig)
