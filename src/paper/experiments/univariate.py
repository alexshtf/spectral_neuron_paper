from typing import TextIO

import numpy as np
import pandas as pd

from paper.experiments import synthetic
from paper.experiments.synthetic import Profile
from paper.targets import make_target
from paper.tasks import make_univariate_task

PROFILES: dict[str, Profile] = {
    "sanity": Profile(
        complexities=(5,),
        target_seeds=range(2),
        init_seeds=range(1),
        dims=(5, 9),
        lrs=(1e-3, 1e-2),
        budgets=(1, 2, 5, 10, 30),
        batch_size=32,
    ),
    "small": Profile(
        complexities=(5, 10, 20),
        target_seeds=range(8),
        init_seeds=range(2),
        dims=(5, 9, 15),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 4).tolist()),
        budgets=(1, 2, 5, 10, 20, 50, 100, 200),
        batch_size=32,
    ),
    "full": Profile(
        complexities=(5, 10, 20),
        target_seeds=range(32),
        init_seeds=range(3),
        dims=(5, 9, 15),
        lrs=tuple(np.geomspace(1e-4, 1e-1, 8).tolist()),
        budgets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
        batch_size=32,
        noise_stds=(0.0, 1e-1),
    ),
}


def run_profile(
    profile: Profile,
    *,
    val_size: int = 4096,
    test_size: int = 4096,
    workers: int = 1,
    progress: bool = False,
    progress_file: TextIO | None = None,
) -> pd.DataFrame:
    return synthetic.run_profile(
        profile,
        make_target=make_target,
        make_task=make_univariate_task,
        val_size=val_size,
        test_size=test_size,
        workers=workers,
        progress=progress,
        progress_file=progress_file,
    )


def main(argv: list[str] | None = None) -> None:
    synthetic.run_cli(
        "univariate",
        PROFILES,
        run_profile,
        argv=argv,
    )


if __name__ == "__main__":
    main()
