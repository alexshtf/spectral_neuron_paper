from typing import TextIO

import pandas as pd

from paper.experiments import synthetic
from paper.experiments.synthetic import Profile
from paper.targets import make_target
from paper.tasks import make_univariate_task

PROFILES = synthetic.standard_profiles((5, 10, 20))


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
