from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from paper.movielens import MovieLensTask, prepare_corpus


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (user, 100 + movie, 0.5 + ((3 * user + movie) % 10) / 2, 1_000 + row)
            for row, (user, movie) in enumerate(
                (user, movie) for user in range(1, 5) for movie in range(10)
            )
        ],
        columns=("userId", "movieId", "rating", "timestamp"),
    )


def _prepare(tmp_path: Path, *, split_seed: int = 7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ratings = _ratings()
    raw_path = tmp_path / "ratings.csv"
    ratings.to_csv(raw_path, index=False)
    corpus = prepare_corpus(
        raw_path,
        tmp_path / "cache",
        split_seed=split_seed,
        chunk_size=7,
    )
    return corpus, ratings


def _parts(corpus):
    feature_ids = np.asarray(corpus.feature_ids())
    return (
        feature_ids[: corpus.train_rows],
        feature_ids[corpus.train_rows : corpus.val_stop],
        feature_ids[corpus.val_stop :],
    )


def test_split_is_deterministic_exhaustive_disjoint_and_warm(tmp_path):
    first, source = _prepare(tmp_path / "first")
    second, _ = _prepare(tmp_path / "second")
    first_parts = _parts(first)
    second_parts = _parts(second)

    assert first.rows == len(source)
    assert (first.train_rows, first.val_rows, first.test_rows) == (32, 4, 4)
    for left, right in zip(first_parts, second_parts):
        np.testing.assert_array_equal(left, right)

    pair_sets = [set(map(tuple, part.tolist())) for part in first_parts]
    assert not pair_sets[0] & pair_sets[1]
    assert not pair_sets[0] & pair_sets[2]
    assert not pair_sets[1] & pair_sets[2]
    assert len(set.union(*pair_sets)) == len(source)

    train_features = np.unique(first_parts[0])
    np.testing.assert_array_equal(train_features, np.arange(first.num_features))


def test_user_and_movie_ids_have_compact_disjoint_namespaces(tmp_path):
    corpus, _ = _prepare(tmp_path)
    feature_ids = corpus.feature_ids()

    np.testing.assert_array_equal(
        np.unique(feature_ids[:, 0]), np.arange(corpus.num_users)
    )
    np.testing.assert_array_equal(
        np.unique(feature_ids[:, 1]),
        np.arange(corpus.num_users, corpus.num_features),
    )


def test_movie_seen_only_in_holdout_is_moved_to_training(tmp_path):
    ratings = _ratings()
    # Both rows are held out for split seed 7; exactly one should be moved.
    ratings.loc[[9, 17], "movieId"] = 999
    raw_path = tmp_path / "ratings.csv"
    ratings.to_csv(raw_path, index=False)

    corpus = prepare_corpus(
        raw_path, tmp_path / "cache", split_seed=7, chunk_size=7
    )
    train, validation, test = _parts(corpus)

    assert corpus.train_rows == 33
    assert np.unique(train[:, 1]).size == corpus.num_movies
    assert set(train[:, 1]) >= set(validation[:, 1]) | set(test[:, 1])
    rare_id = corpus.num_features - 1
    assert np.count_nonzero(train[:, 1] == rare_id) == 1
    holdout = np.concatenate((validation, test))
    assert np.count_nonzero(holdout[:, 1] == rare_id) == 1


def test_training_order_permutes_the_full_pool_deterministically(tmp_path):
    corpus, _ = _prepare(tmp_path)
    first = np.load(corpus.order_path(11))

    np.testing.assert_array_equal(np.sort(first), np.arange(corpus.train_rows))
    np.testing.assert_array_equal(np.load(corpus.order_path(11)), first)
    assert not np.array_equal(first, np.load(corpus.order_path(12)))


def test_task_batches_are_local_centered_and_report_warm_coverage(tmp_path):
    corpus, _ = _prepare(tmp_path)
    task = MovieLensTask(corpus, data_seed=3, batch_size=5)
    feature_ids, ratings = corpus.feature_ids(), corpus.ratings()
    order = np.load(corpus.order_path(3))

    batches = list(task.train_batches(2, 10))
    actual_ids = np.concatenate([inputs[0].numpy() for inputs, _ in batches])
    actual_ratings = np.concatenate([targets.numpy() for _, targets in batches])
    expected_rows = np.concatenate((np.sort(order[2:7]), np.sort(order[7:10])))

    np.testing.assert_array_equal(actual_ids, feature_ids[expected_rows])
    np.testing.assert_allclose(
        actual_ratings,
        ratings[expected_rows] - corpus.rating_center,
    )
    assert corpus.rating_center == 2.75
    assert batches[0][0][0].dtype == torch.int32
    assert batches[0][1].dtype == torch.float32

    coverage = task.warm_coverage((0, 8, corpus.train_rows))
    assert coverage[0] == (0.0, 0.0)
    assert coverage[8][0] <= coverage[corpus.train_rows][0]
    assert coverage[8][1] <= coverage[corpus.train_rows][1]
    assert coverage[corpus.train_rows] == (1.0, 1.0)


def test_accepts_directory_and_official_style_zip(tmp_path):
    ratings = _ratings()
    directory = tmp_path / "ml-20m"
    directory.mkdir()
    ratings.to_csv(directory / "ratings.csv", index=False)

    zip_path = tmp_path / "ml-20m.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.write(directory / "ratings.csv", "ml-20m/ratings.csv")

    from_directory = prepare_corpus(
        directory, tmp_path / "directory-cache", chunk_size=9
    )
    from_zip = prepare_corpus(zip_path, tmp_path / "zip-cache", chunk_size=9)

    np.testing.assert_array_equal(
        from_directory.feature_ids(), from_zip.feature_ids()
    )
    np.testing.assert_array_equal(from_directory.ratings(), from_zip.ratings())
