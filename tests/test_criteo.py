from compression import zstd
from pathlib import Path

import numpy as np
import pytest

from paper.criteo import (
    MISSING_NUMERIC,
    NUM_CATEGORICAL_FIELDS,
    NUM_NUMERIC_FIELDS,
    BucketPreprocessor,
    CriteoTask,
    HybridPreprocessor,
    _bucket_numeric,
    default_cache_dir,
    fit_preprocessors,
    load_encoded,
    load_preprocessor,
    prepare_corpus,
    prepare_encoded_data,
)
from criteo_test_data import write_tiny_criteo


def _compress_zstd(path: Path) -> Path:
    compressed = path.with_name(f"{path.name}.zstd")
    compressed.write_bytes(zstd.compress(path.read_bytes(), level=3))
    path.unlink()
    return compressed


def test_default_cache_dir_tracks_the_corpus_cache_version(monkeypatch):
    monkeypatch.setattr("paper.criteo.CACHE_VERSION", 7)

    assert default_cache_dir(Path("train.txt")).name == ".train.txt.cache-v7"


def _categorical_vocabularies() -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([7, 9], dtype=np.uint32) for _ in range(NUM_CATEGORICAL_FIELDS)
    )


def _assert_arrays(actual, expected):
    for actual_array, expected_array in zip(actual, expected, strict=True):
        if expected_array is None:
            assert actual_array is None
        else:
            np.testing.assert_array_equal(actual_array, expected_array)


def _global_ids(local_ids, field_offsets):
    return np.asarray(local_ids, dtype=np.int32) + np.asarray(
        field_offsets, dtype=np.int32
    )


def test_winner_style_numeric_buckets():
    values = np.array([np.nan, -1, 0, 1, 2, 3, np.e**3])
    buckets = _bucket_numeric(values)

    assert buckets.tolist() == [0, -1, 0, 1, 2, 3, 11]


def test_bucket_preprocessor_uses_exact_field_disjoint_ids():
    minimums = np.zeros(NUM_NUMERIC_FIELDS, dtype=np.int32)
    maximums = np.full(NUM_NUMERIC_FIELDS, 5, dtype=np.int32)
    preprocessor = BucketPreprocessor(
        minimums,
        maximums,
        _categorical_vocabularies(),
    )
    numerics = np.full((4, NUM_NUMERIC_FIELDS), 7, dtype=np.int32)
    numerics[:, 0] = [MISSING_NUMERIC, -1, 3, 7]
    categoricals = np.full((4, NUM_CATEGORICAL_FIELDS), 7, dtype=np.uint32)
    categoricals[:, 0] = [0, 8, 7, 9]

    feature_ids, feature_values = preprocessor.encode(numerics, categoricals)

    offsets = np.concatenate(
        (
            np.arange(NUM_NUMERIC_FIELDS) * 8,
            NUM_NUMERIC_FIELDS * 8 + np.arange(NUM_CATEGORICAL_FIELDS) * 4,
        )
    )
    assert np.all(feature_ids >= offsets)
    assert np.all(feature_ids < offsets + preprocessor.field_sizes)
    assert feature_ids[:, 0].tolist() == [0, 1, 5, 7]
    assert feature_ids[:, NUM_NUMERIC_FIELDS].tolist() == [
        offsets[NUM_NUMERIC_FIELDS],
        offsets[NUM_NUMERIC_FIELDS] + 1,
        offsets[NUM_NUMERIC_FIELDS] + 2,
        offsets[NUM_NUMERIC_FIELDS] + 3,
    ]
    assert preprocessor.field_offsets.dtype == np.int32
    assert feature_values is None


def test_hybrid_preprocessor_separates_special_and_positive_values():
    negatives = (np.array([-2, -1], dtype=np.int32),) + tuple(
        np.array([], dtype=np.int32) for _ in range(NUM_NUMERIC_FIELDS - 1)
    )
    preprocessor = HybridPreprocessor(
        _categorical_vocabularies(),
        negatives,
        positive_mean=np.zeros(NUM_NUMERIC_FIELDS),
        positive_scale=np.ones(NUM_NUMERIC_FIELDS),
    )
    numerics = np.ones((6, NUM_NUMERIC_FIELDS), dtype=np.int32)
    numerics[:, 0] = [MISSING_NUMERIC, -3, -2, -1, 0, 1_000_000]
    categoricals = np.full((6, NUM_CATEGORICAL_FIELDS), 7, dtype=np.uint32)

    feature_ids, feature_values = preprocessor.encode(numerics, categoricals)

    assert feature_ids[:, 0].tolist() == [0, 2, 3, 4, 1, 5]
    assert feature_values[:5, 0].tolist() == [1.0] * 5
    assert feature_values[5, 0] == np.float32(np.log1p(1_000_000))
    assert preprocessor.field_offsets.dtype == np.int32


def test_prepare_corpus_streams_zstd_source(tmp_path):
    raw_path = tmp_path / "train.txt"
    write_tiny_criteo(raw_path, rows=17)
    compressed_path = _compress_zstd(raw_path)

    corpus = prepare_corpus(compressed_path, tmp_path / "cache", chunk_size=5)

    assert corpus.rows == 17
    np.testing.assert_array_equal(
        corpus.labels(),
        [int(row % 4 == 0) for row in range(17)],
    )
    assert not raw_path.exists()


def test_hybrid_preprocessor_fits_sampled_positive_statistics(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path)

    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        chunk_size=23,
    )
    paths = fit_preprocessors(
        corpus,
        ("hybrid",),
        sample_size=8,
        sample_seed=7,
        min_count=2,
    )

    hybrid = load_preprocessor(paths["hybrid"])
    rows = np.random.default_rng(7).choice(
        corpus.train_stop, size=8, replace=False, shuffle=False
    )
    sampled = corpus.numerics()[rows, 0]
    positive = np.log1p(sampled[sampled > 0])
    assert isinstance(hybrid, HybridPreprocessor)
    np.testing.assert_allclose(hybrid.positive_mean[0], positive.mean())
    scale = positive.std()
    np.testing.assert_allclose(hybrid.positive_scale[0], scale if scale > 0 else 1.0)


@pytest.fixture
def encoded_variants(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    write_tiny_criteo(raw_path, rows=103)
    corpus = prepare_corpus(raw_path, cache_dir)
    paths = fit_preprocessors(
        corpus,
        ("bucket", "hybrid"),
        sample_size=8,
        sample_seed=0,
        min_count=2,
    )
    return corpus, {
        kind: (
            load_preprocessor(path),
            prepare_encoded_data(corpus, path, chunk_size=3),
        )
        for kind, path in paths.items()
    }


def test_encoded_cache_matches_preprocessor_with_local_ids(encoded_variants):
    corpus, variants = encoded_variants
    for preprocessor, data in variants.values():
        sources = (
            (data.train_path, slice(0, corpus.train_stop)),
            (data.holdout_path, slice(corpus.train_stop, corpus.rows)),
        )
        for split_path, rows in sources:
            split = load_encoded(split_path)
            assert split.feature_ids.dtype == np.uint16
            _assert_arrays(
                (
                    _global_ids(split.feature_ids, data.field_offsets),
                    split.feature_values,
                    split.labels,
                ),
                (
                    *preprocessor.encode(
                        np.asarray(corpus.numerics()[rows]),
                        np.asarray(corpus.categoricals()[rows]),
                    ),
                    np.asarray(corpus.labels()[rows]),
                ),
            )


def test_criteo_task_gathers_shuffled_training_batches(encoded_variants):
    corpus, variants = encoded_variants
    batch_size = 4
    order = corpus.shuffled_epochs(7)
    order.prepare(2)
    train_size = corpus.train_stop + 3
    expected_rows = np.concatenate(list(order.batches(train_size, batch_size)))

    for _, data in variants.values():
        task = CriteoTask(data, order, batch_size)
        expected = load_encoded(data.train_path)
        batches = list(task.train_batches(train_size))
        actual_ids = np.concatenate(
            [model_inputs[0].numpy() for model_inputs, _ in batches]
        )
        actual_values = (
            None
            if expected.feature_values is None
            else np.concatenate(
                [model_inputs[1].numpy() for model_inputs, _ in batches]
            )
        )
        actual_labels = np.concatenate([labels.numpy() for _, labels in batches])
        assert actual_ids.dtype == np.int32
        _assert_arrays(
            (actual_ids, actual_values, actual_labels),
            (
                _global_ids(expected.feature_ids[expected_rows], data.field_offsets),
                (
                    None
                    if expected.feature_values is None
                    else expected.feature_values[expected_rows]
                ),
                expected.labels[expected_rows],
            ),
        )


def test_criteo_task_respects_holdout_boundaries(encoded_variants):
    corpus, variants = encoded_variants
    _, data = variants["bucket"]
    task = CriteoTask(data, corpus.shuffled_epochs(7), batch_size=4)

    val_labels = np.concatenate([labels.numpy() for _, labels in task.val_batches()])
    test_labels = np.concatenate(
        [labels.numpy() for _, labels in task.test_batches()]
    )
    labels = corpus.labels()
    np.testing.assert_array_equal(
        val_labels, labels[corpus.train_stop : corpus.val_stop]
    )
    np.testing.assert_array_equal(test_labels, labels[corpus.val_stop :])
