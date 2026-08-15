from paper.plotting import plot_bivariate_target_gallery
from paper.targets import TargetSpec


def test_plot_bivariate_target_gallery_draws_contours():
    specs = [
        TargetSpec(kind="general", complexity=5, seed=0),
        TargetSpec(kind="monotone", complexity=5, seed=0),
    ]

    fig = plot_bivariate_target_gallery(specs, resolution=20)
    axes = [ax for ax in fig.axes if ax.get_title()]

    assert [ax.get_title() for ax in axes] == [
        "general, complexity=5, seed=0",
        "monotone, complexity=5, seed=0",
    ]
    assert all(ax.collections for ax in axes)
    assert all(ax.get_xlabel() == "$x_1$" for ax in axes)
    assert all(ax.get_ylabel() == "$x_2$" for ax in axes)
