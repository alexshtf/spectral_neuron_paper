import math

import matplotlib.pyplot as plt
import numpy as np


def func_gallery(random_func_fn, ms=[3, 5, 10, 20]):
    n_cols = int(math.ceil(math.sqrt(len(ms))))
    n_rows = int(math.ceil(len(ms) / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), layout="constrained"
    )
    xs = np.linspace(-4, 4, 1000)

    for m, ax in zip(ms, axs.ravel()):
        func = random_func_fn(m)
        ax.plot(xs, func(xs))
        ax.set_title(f"$f_{{{m}}}(x)$")

    return fig
