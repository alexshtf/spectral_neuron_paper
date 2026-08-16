import pandas as pd
import pytest

from paper.plotting import (
    plot_general_scaling,
    plot_monotone_scaling,
)


def _row(
    *,
    complexity: int,
    dim: int,
    model: str,
    train_size: int,
    target_kind: str = "monotone",
    noise_std: float = 0.0,
) -> dict:
    return {
        "target_kind": target_kind,
        "complexity": complexity,
        "noise_std": noise_std,
        "model": model,
        "dim": dim,
        "train_size": train_size,
        "median_test_rmse": 1.0 / train_size,
        "q25_test_rmse": 0.8 / train_size,
        "q75_test_rmse": 1.2 / train_size,
    }


def test_plot_monotone_scaling_pairs_models_by_dimension():
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity, dim=dim, model=model, train_size=train_size
            )
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in ("unconstrained", "monotone")
            for train_size in (32, 64)
        ]
    )

    fig = plot_monotone_scaling(summary)
    axes = [ax for ax in fig.axes if ax.get_visible()]

    assert len(axes) == 4
    assert {ax.get_title() for ax in axes} == {
        "complexity=5, dim=3",
        "complexity=5, dim=5",
        "complexity=10, dim=3",
        "complexity=10, dim=5",
    }
    assert all(
        {line.get_label() for line in ax.lines} == {"unconstrained", "monotone"}
        for ax in axes
    )
    assert {
        line.get_label(): line.get_linestyle() for line in axes[0].lines
    } == {"unconstrained": "-", "monotone": "--"}
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in axes)
    assert all(ax.get_xlabel() == "training-sample budget" for ax in axes)


def test_plot_general_scaling_styles_models_and_dimensions():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=dim,
                model=model,
                train_size=train_size,
                target_kind="general",
            )
            for dim in (5, 9)
            for model in ("unconstrained", "monotone")
            for train_size in (32, 64)
        ]
    )

    fig = plot_general_scaling(summary)
    lines = {line.get_label(): line for line in fig.axes[0].lines}

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
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)


@pytest.mark.parametrize(
    ("plotter", "target_kind"),
    [
        (plot_general_scaling, "monotone"),
        (plot_monotone_scaling, "general"),
    ],
)
def test_scaling_plots_reject_the_wrong_target_kind(plotter, target_kind):
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                dim=5,
                model="unconstrained",
                train_size=train_size,
                target_kind=target_kind,
            )
            for train_size in (32, 64)
        ]
    )

    with pytest.raises(ValueError, match="expected target_kind"):
        plotter(summary)


def test_monotone_scaling_requires_the_intended_model_pair():
    summary = pd.DataFrame(
        [
            _row(
                complexity=5,
                model=model,
                dim=3,
                train_size=train_size,
                target_kind="monotone",
            )
            for model in ("unconstrained", "monotone", "extra")
            for train_size in (32, 64)
        ]
    )

    with pytest.raises(ValueError, match="unconstrained and monotone"):
        plot_monotone_scaling(summary)


@pytest.mark.parametrize(
    ("plotter", "target_kind", "models", "expected_axes"),
    [
        (
            plot_monotone_scaling,
            "monotone",
            ("unconstrained", "monotone"),
            4,
        ),
        (plot_general_scaling, "general", ("unconstrained",), 2),
    ],
)
def test_scaling_plots_separate_noise_levels(
    plotter, target_kind, models, expected_axes
):
    summary = pd.DataFrame(
        [
            _row(
                complexity=complexity,
                dim=dim,
                model=model,
                train_size=train_size,
                target_kind=target_kind,
                noise_std=noise_std,
            )
            for noise_std in (0.0, 0.1, 0.2)
            for complexity in (5, 10)
            for dim in (3, 5)
            for model in models
            for train_size in (32, 64)
        ]
    )

    fig = plotter(summary)
    assert [subfigure.get_suptitle() for subfigure in fig.subfigs] == [
        "Noiseless training (σ = 0)",
        "Noisy training (σ = 0.1)",
        "Noisy training (σ = 0.2)",
    ]
    for subfigure in fig.subfigs:
        axes = [ax for ax in subfigure.axes if ax.get_visible()]
        assert len(axes) == expected_axes
        assert len(subfigure.legends) == 1
        assert all(ax.get_legend() is None for ax in axes)
