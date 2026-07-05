from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from paper.models import KthEigval, KthEigval1DMonotone
from paper.plotting import func_gallery
from paper.synth_fn import random_func, random_inc_func
from paper.training import Synthetic1DStream, compute_scaling_law

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

mpl.rcParams["figure.dpi"] = 140

func_gallery(random_func)
plt.savefig(plots_dir / "general_function_gallery.png", bbox_inches="tight")
plt.close()

dims = [3, 5, 10]
complexities = [3, 5, 10, 20]
scaling_laws = []
for complexity in complexities:
    for dim in dims:
        scaling_law = compute_scaling_law(
            Synthetic1DStream(random_func(complexity)),
            lambda: KthEigval(1, dim),
        )
        scaling_law["dim"] = dim
        scaling_law["complexity"] = complexity
        scaling_laws.append(scaling_law.reset_index())
        print(f"Computed dim={dim}, complexity={complexity}")

scaling_laws_df = pd.concat(scaling_laws, axis=0)

sns.relplot(
    scaling_laws_df,
    kind="line",
    x="step",
    y="test_rmse",
    col="complexity",
    hue="dim",
    col_wrap=2,
    height=3,
    facet_kws={"sharey": False, "sharex": True},
)
plt.savefig(plots_dir / "general_scaling_law.png", bbox_inches="tight")
plt.close()

func_gallery(random_inc_func)
plt.savefig(plots_dir / "monotone_function_gallery.png", bbox_inches="tight")
plt.close()

dims = [3, 5, 10]
complexities = [5, 10, 20]
noise_stds = [0, 1e-1]
scaling_laws = []
for complexity in complexities:
    for dim in dims:
        for noise_std in noise_stds:
            scaling_law = compute_scaling_law(
                Synthetic1DStream(random_inc_func(complexity), noise_std=noise_std),
                lambda: KthEigval1DMonotone(dim),
            )
            scaling_law["dim"] = dim
            scaling_law["complexity"] = complexity
            scaling_law["kind"] = "monotone"
            scaling_law["noise_std"] = noise_std
            scaling_laws.append(scaling_law.reset_index())
            print(
                f"Computed dim={dim}, complexity={complexity}, "
                f"noise_std={noise_std}, kind = monotone"
            )

            scaling_law = compute_scaling_law(
                Synthetic1DStream(random_inc_func(complexity), noise_std=noise_std),
                lambda: KthEigval(1, dim),
            )
            scaling_law["dim"] = dim
            scaling_law["complexity"] = complexity
            scaling_law["kind"] = "generic"
            scaling_law["noise_std"] = noise_std
            scaling_laws.append(scaling_law.reset_index())
            print(
                f"Computed dim={dim}, complexity={complexity}, "
                f"noise_std={noise_std}, kind = generic"
            )

scaling_laws_df = pd.concat(scaling_laws, axis=0)

sns.relplot(
    scaling_laws_df[scaling_laws_df["noise_std"] == 0],
    kind="line",
    x="step",
    y="test_rmse",
    row="dim",
    col="complexity",
    hue="kind",
    height=3,
    facet_kws={"sharey": False, "sharex": True},
)
plt.savefig(plots_dir / "monotone_scaling_law_no_noise.png", bbox_inches="tight")
plt.close()

sns.relplot(
    scaling_laws_df[scaling_laws_df["noise_std"] > 0],
    kind="line",
    x="step",
    y="test_rmse",
    row="dim",
    col="complexity",
    hue="kind",
    height=3,
    facet_kws={"sharey": False, "sharex": True},
)
plt.savefig(plots_dir / "monotone_scaling_law_with_noise.png", bbox_inches="tight")
plt.close()
