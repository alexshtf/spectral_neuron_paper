import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paper.plotting import (
    plot_bivariate_target_gallery,
    plot_criteo_scaling,
    plot_scaling,
)
from paper.targets import TargetSpec


def _row(
    *,
    complexity: int,
    dim: int,
    model: str,
    budget: int,
    target_kind: str = "monotone",
    noise_std: float = 0.0,
) -> dict:
    return {
        "target_kind": target_kind,
        "complexity": complexity,
        "noise_std": noise_std,
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


def test_plot_scaling_rejects_mixed_target_kinds():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=5,
                model="unconstrained",
                budget=budget,
                target_kind=target_kind,
            )
            for target_kind in ("general", "monotone")
            for budget in (1, 2)
        ]
    )

    with pytest.raises(ValueError, match="single target_kind"):
        plot_scaling(summary)


@pytest.mark.parametrize(
    ("target_kind", "models", "expected_axes", "lines_per_axis"),
    [
        ("monotone", ("unconstrained", "monotone"), 4, 2),
        ("general", ("unconstrained",), 2, 2),
    ],
)
def test_plot_scaling_separates_noise_before_choosing_target_layout(
    target_kind, models, expected_axes, lines_per_axis
):
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity,
                dim=dim,
                model=model,
                budget=budget,
                target_kind=target_kind,
                noise_std=noise_std,
            )
            for noise_std in (0.0, 0.1)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in models
            for budget in (1, 2)
        ]
    )

    fig = plot_scaling(summary)
    try:
        assert [subfigure._suptitle.get_text() for subfigure in fig.subfigs] == [
            "Noiseless training (σ = 0)",
            "Noisy training (σ = 0.1)",
        ]
        for subfigure in fig.subfigs:
            axes = [ax for ax in subfigure.axes if ax.get_visible()]
            assert len(axes) == expected_axes
            assert all(len(ax.lines) == lines_per_axis for ax in axes)
            assert all(len(line.get_xdata()) == 2 for ax in axes for line in ax.lines)
    finally:
        plt.close(fig)


def test_plot_bivariate_target_gallery_draws_contours():
    specs = [
        TargetSpec(kind="general", complexity=5, seed=0),
        TargetSpec(kind="monotone", complexity=5, seed=0),
    ]

    fig = plot_bivariate_target_gallery(specs, resolution=20)
    try:
        axes = [ax for ax in fig.axes if ax.get_title()]

        assert [ax.get_title() for ax in axes] == [
            "general, complexity=5, seed=0",
            "monotone, complexity=5, seed=0",
        ]
        assert all(ax.collections for ax in axes)
        assert all(ax.get_xlabel() == "$x_1$" for ax in axes)
        assert all(ax.get_ylabel() == "$x_2$" for ax in axes)
    finally:
        plt.close(fig)


def test_plot_criteo_scaling_pairs_models_by_parameter_count():
    rows = []
    for train_size in (2**14, 2**18):
        for model, dim, rank, parameters in (
            ("linear", 0, 0, 1),
            ("fm", 0, 5, 6),
            ("spectral", 3, 0, 6),
            ("fm", 0, 14, 15),
            ("spectral", 5, 0, 15),
        ):
            rows.append(
                {
                    "train_size": train_size,
                    "model": model,
                    "matrix_dim": dim,
                    "fm_rank": rank,
                    "parameters_per_feature": parameters,
                    "median_test_logloss": 0.5,
                    "q25_test_logloss": 0.49,
                    "q75_test_logloss": 0.51,
                }
            )

    fig = plot_criteo_scaling(pd.DataFrame(rows))
    try:
        ax = fig.axes[0]
        lines = {line.get_label(): line for line in ax.lines}

        assert len(lines) == 5
        assert (
            lines["FM (rank 5, 6/feature)"].get_color()
            == lines["Spectral (dim 3, 6/feature)"].get_color()
        )
        assert (
            lines["FM (rank 14, 15/feature)"].get_color()
            == lines["Spectral (dim 5, 15/feature)"].get_color()
        )
        assert lines["FM (rank 5, 6/feature)"].get_linestyle() == "-"
        assert lines["Spectral (dim 3, 6/feature)"].get_linestyle() == "--"
        assert ax.get_xscale() == "log"
    finally:
        plt.close(fig)
