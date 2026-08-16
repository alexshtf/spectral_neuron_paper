from itertools import product

import numpy as np
import pandas as pd
import pytest

from paper.plotting import plot_higgs_deviation_shell_grid
from paper.plotting.robustness import _mean_higgs_deviation_shells


def _higgs_deviation_results(
    *,
    noise_levels: tuple[float, ...] = (0.5,),
    dimensions: tuple[int, ...] = (3, 7),
    features: tuple[tuple[int, str], ...] = ((0, "lepton pT"), (1, "lepton η")),
    magnitude_bins: int = 4,
    ratio_bins: int = 4,
) -> pd.DataFrame:
    rows = []
    for (
        noise_level,
        dim,
        init_seed,
        (feature_index, feature_name),
        magnitude_bin_index,
    ) in product(
        noise_levels,
        dimensions,
        range(2),
        features,
        range(magnitude_bins),
    ):
        counts = np.arange(1, ratio_bins + 1) * (magnitude_bin_index + 1)
        row = {
            "dim": dim,
            "data_seed": 0,
            "init_seed": init_seed,
            "noise_level": noise_level,
            "feature_index": feature_index,
            "feature_name": feature_name,
            "magnitude_bin_index": magnitude_bin_index,
            "magnitude_left": noise_level * magnitude_bin_index / magnitude_bins,
            "magnitude_right": noise_level
            * (magnitude_bin_index + 1)
            / magnitude_bins,
            "total_count": int(counts.sum() + 1),
            "zero_bound_count": 1,
        }
        row.update(
            {
                f"ratio_bin_{bin_index:03d}_count": int(count)
                for bin_index, count in enumerate(counts)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _legend_labels(fig) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_plot_higgs_deviation_shell_grid_uses_disjoint_ranges_and_feature_order():
    fig = plot_higgs_deviation_shell_grid(
        _higgs_deviation_results(), shell_count=4
    )
    ax = fig.axes[0]

    assert len(fig.axes) == 8
    assert [axis.get_title() for axis in fig.axes[:4]] == [
        "|δ| ∈ [0, 0.125)",
        "|δ| ∈ [0.125, 0.25)",
        "|δ| ∈ [0.25, 0.375)",
        "|δ| ∈ [0.375, 0.5]",
    ]
    assert ax.get_xlim() == pytest.approx((0, 1))
    assert fig.get_supxlabel() == "Deviation ratio  |Δf| / (|δ| ‖Aⱼ‖₂)"
    assert fig.get_supylabel() == "Feature"
    assert _legend_labels(fig) == ["3", "7"]
    assert fig.legends[0].get_title().get_text() == "Matrix dimension"
    assert fig.get_figheight() == pytest.approx(24 / 25.4)
    assert fig.axes[0].get_ylabel() == "lepton pT"
    assert fig.axes[4].get_ylabel() == "lepton η"
    assert fig.axes[0].yaxis.label.get_fontsize() == 8


def test_mean_higgs_deviation_shells_aggregates_disjoint_shells():
    rows = []
    for magnitude_bin_index in range(4):
        row = {
            "dim": 3,
            "data_seed": 0,
            "init_seed": 0,
            "noise_level": 0.5,
            "feature_index": 0,
            "feature_name": "lepton pT",
            "magnitude_bin_index": magnitude_bin_index,
            "magnitude_left": magnitude_bin_index / 8,
            "magnitude_right": (magnitude_bin_index + 1) / 8,
            "total_count": 1,
            "zero_bound_count": 0,
        }
        row.update(
            {
                f"ratio_bin_{bin_index:03d}_count": int(
                    bin_index == magnitude_bin_index
                )
                for bin_index in range(4)
            }
        )
        rows.append(row)

    noise_level, ratio_columns, shells = _mean_higgs_deviation_shells(
        pd.DataFrame(rows), shell_count=2
    )

    assert noise_level == 0.5
    assert ratio_columns == [f"ratio_bin_{index:03d}_count" for index in range(4)]
    np.testing.assert_allclose(
        shells.sort_values("shell_index")[ratio_columns],
        [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]],
    )


def test_higgs_deviation_shells_preserve_probabilities_and_cell_limits():
    rows = []
    feature_counts = {
        (0, "sharp"): ((9, 1), (1, 1)),
        (1, "flat"): ((1, 1), (1, 1)),
    }
    for (feature_index, feature_name), shell_counts in feature_counts.items():
        for magnitude_bin_index, counts in enumerate(shell_counts):
            rows.append(
                {
                    "dim": 3,
                    "data_seed": 0,
                    "init_seed": 0,
                    "noise_level": 0.5,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "magnitude_bin_index": magnitude_bin_index,
                    "magnitude_left": magnitude_bin_index / 4,
                    "magnitude_right": (magnitude_bin_index + 1) / 4,
                    "total_count": sum(counts),
                    "zero_bound_count": 0,
                    "ratio_bin_000_count": counts[0],
                    "ratio_bin_001_count": counts[1],
                }
            )

    results = pd.DataFrame(rows)
    _, ratio_columns, shells = _mean_higgs_deviation_shells(
        results, shell_count=2
    )
    probabilities = shells.set_index(["feature_name", "shell_index"])
    np.testing.assert_allclose(
        probabilities.loc[("sharp", 0), ratio_columns], [0.9, 0.1]
    )
    np.testing.assert_allclose(
        probabilities.loc[("sharp", 1), ratio_columns], [0.5, 0.5]
    )

    fig = plot_higgs_deviation_shell_grid(results, shell_count=2)
    sharp_first, sharp_second, flat_first, flat_second = fig.axes
    assert sharp_first.get_ylim()[1] == pytest.approx(0.9 / 0.85)
    assert sharp_second.get_ylim()[1] == pytest.approx(0.5 / 0.85)
    assert flat_first.get_ylim()[1] == pytest.approx(0.5 / 0.85)
    assert flat_second.get_ylim()[1] == pytest.approx(0.5 / 0.85)


def test_mean_higgs_deviation_shells_weights_runs_equally():
    rows = []
    for init_seed, (counts, total_count) in enumerate(
        (((8, 2), 10), ((0, 100), 100))
    ):
        rows.append(
            {
                "dim": 3,
                "data_seed": 0,
                "init_seed": init_seed,
                "noise_level": 0.5,
                "feature_index": 0,
                "feature_name": "lepton pT",
                "magnitude_bin_index": 0,
                "magnitude_left": 0,
                "magnitude_right": 0.5,
                "total_count": total_count,
                "zero_bound_count": 0,
                "ratio_bin_000_count": counts[0],
                "ratio_bin_001_count": counts[1],
            }
        )

    _, ratio_columns, shells = _mean_higgs_deviation_shells(
        pd.DataFrame(rows), shell_count=1
    )
    probabilities = shells.loc[0, ratio_columns].to_numpy(dtype=float)

    # Mean seed probabilities are (0.4, 0.6), not pooled counts (8, 102).
    np.testing.assert_allclose(probabilities, [0.4, 0.6])


def test_mean_higgs_deviation_shells_rejects_mixed_noise_and_bad_shell_count():
    with pytest.raises(ValueError, match="one noise_level"):
        _mean_higgs_deviation_shells(
            _higgs_deviation_results(noise_levels=(0.25, 0.5)), shell_count=4
        )
    with pytest.raises(ValueError, match="divisor"):
        _mean_higgs_deviation_shells(_higgs_deviation_results(), shell_count=3)
