from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper.experiments.movielens_scaling import (
    RAW_COLUMNS,
    MovieLensModelSpec,
    Profile,
    SeedGrid,
    _make_seeded_model,
    make_model,
    run_profile,
    summarize_raw,
    validate_raw,
)
from paper.models import FactorizationMachine


def _write_ratings(path: Path, users: int = 6, movies: int = 10) -> None:
    rows = [
        {
            "userId": user + 1,
            "movieId": 10 * (movie + 1),
            "rating": 0.5 + ((3 * user + movie) % 10) / 2,
            "timestamp": 1_000_000 + 100 * user + movie,
        }
        for user in range(users)
        for movie in range(movies)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(8, 16),
        dims=(3,),
        lrs=(1e-2, 1e-1),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(data_seeds=range(1, 2), init_seeds=range(1, 2)),
        batch_size=8,
    )


def test_two_field_fm_is_biased_matrix_factorization():
    model = FactorizationMachine(num_features=4, num_fields=2, rank=2)
    with torch.no_grad():
        model.bias.fill_(0.25)
        model.weight.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model.embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    ids = torch.tensor([[0, 2], [1, 3]])

    expected = torch.tensor(
        [
            0.25 + 1.0 + 3.0 + 1.0 * 5.0 + 2.0 * 6.0,
            0.25 + 2.0 + 4.0 + 3.0 * 7.0 + 4.0 * 8.0,
        ]
    )

    torch.testing.assert_close(model(ids), expected)


def test_parameter_matching_is_per_identity():
    fm_spec = MovieLensModelSpec("fm", 3)
    spectral_spec = MovieLensModelSpec("spectral", 3)
    num_features = 17

    fm = make_model(fm_spec, num_features)
    spectral = make_model(spectral_spec, num_features)

    assert fm_spec.rank == 5
    assert fm_spec.parameters_per_identity == spectral_spec.parameters_per_identity == 6
    assert sum(p.numel() for p in spectral.parameters()) - sum(
        p.numel() for p in fm.parameters()
    ) == 5


def test_seeded_construction_preserves_global_rng():
    torch.manual_seed(17)
    state = torch.random.get_rng_state()

    first = _make_seeded_model(
        MovieLensModelSpec("spectral", 3), num_features=17, init_seed=5
    )
    second = _make_seeded_model(
        MovieLensModelSpec("spectral", 3), num_features=17, init_seed=5
    )

    assert torch.equal(torch.random.get_rng_state(), state)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_tiny_profile_runs_end_to_end(tmp_path):
    ratings = tmp_path / "ratings.csv"
    _write_ratings(ratings)
    profile = _tiny_profile()

    raw = run_profile(
        profile,
        raw_path=ratings,
        cache_dir=tmp_path / "cache",
        chunk_size=13,
    )
    validate_raw(raw, profile)
    summary = summarize_raw(raw)

    assert list(raw.columns) == RAW_COLUMNS
    assert set(raw["model"]) == {"linear", "fm", "spectral"}
    assert set(raw["protocol"]) == {"one_pass_random_prefix"}
    assert set(raw["optimizer"]) == {"adam+sparseadam"}
    assert set(raw["phase"]) == {"tuning", "evaluation"}

    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert set(tuning["train_size"]) == {16}
    assert tuning["val_rmse"].notna().all()
    assert tuning["val_warm_fraction"].between(0, 1).all()
    assert tuning["test_rmse"].isna().all()
    assert tuning["test_warm_fraction"].isna().all()
    assert evaluation["val_rmse"].isna().all()
    assert evaluation["val_warm_fraction"].isna().all()
    assert np.isfinite(evaluation["test_rmse"]).all()
    assert evaluation["test_warm_fraction"].between(0, 1).all()
    assert set(summary["train_size"]) == {8, 16}
    assert len(summary) == 6
