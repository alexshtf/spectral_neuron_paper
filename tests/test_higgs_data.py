import pickle
from compression import zstd
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import torch

from paper.higgs import (
    FEATURES_FILE,
    LABELS_FILE,
    METADATA_FILE,
    NUM_FEATURES,
    HiggsCorpus,
    HiggsLayout,
    HiggsTask,
    prepare_corpus,
)


TINY_LAYOUT = HiggsLayout(rows=12, train_stop=8, val_stop=10)


def _tiny_data(rows: int = TINY_LAYOUT.rows) -> np.ndarray:
    row = np.arange(rows, dtype=np.float32)[:, None]
    field = np.arange(NUM_FEATURES, dtype=np.float32)[None, :]
    features = row * 2.0 + field / 10.0
    labels = (np.arange(rows) % 3 == 0).astype(np.float32)
    return np.column_stack((labels, features))


def _write_csv(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",", fmt="%.9g")


def _compress_zstd(path: Path) -> Path:
    compressed = path.with_name(f"{path.name}.zstd")
    compressed.write_bytes(zstd.compress(path.read_bytes(), level=3))
    path.unlink()
    return compressed


def _prepare(tmp_path: Path, values: np.ndarray | None = None):
    raw_path = tmp_path / "HIGGS.csv"
    cache_dir = tmp_path / "cache"
    values = _tiny_data() if values is None else values
    _write_csv(raw_path, values)
    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        layout=TINY_LAYOUT,
        chunk_size=3,
    )
    return raw_path, cache_dir, corpus, values


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 2),
        (3, 3, 4),
        (3, 4, 4),
        (5, 4, 6),
    ],
)
def test_layout_requires_strict_nonempty_boundaries(values):
    with pytest.raises(ValueError, match="train_stop"):
        HiggsLayout(*values)


def test_prepare_corpus_preserves_data_and_training_statistics(tmp_path):
    _, cache_dir, corpus, values = _prepare(tmp_path)

    expected_features = values[:, 1:].astype(np.float32)
    np.testing.assert_array_equal(corpus.features(), expected_features)
    np.testing.assert_array_equal(corpus.labels(), values[:, 0].astype(np.uint8))
    expected_training = expected_features[: TINY_LAYOUT.train_stop].astype(np.float64)
    np.testing.assert_allclose(corpus.feature_mean, expected_training.mean(axis=0))
    np.testing.assert_allclose(corpus.feature_scale, expected_training.std(axis=0))
    assert corpus.layout == TINY_LAYOUT
    assert (cache_dir / FEATURES_FILE).stat().st_size == values.shape[0] * 28 * 4
    assert (cache_dir / LABELS_FILE).stat().st_size == values.shape[0]

    reopened = HiggsCorpus.open(cache_dir, layout=TINY_LAYOUT)
    assert reopened == corpus


def test_prepare_corpus_streams_zstd_source(tmp_path):
    raw_path = tmp_path / "HIGGS.csv"
    values = _tiny_data()
    _write_csv(raw_path, values)
    compressed_path = _compress_zstd(raw_path)

    corpus = prepare_corpus(
        compressed_path,
        tmp_path / "cache",
        layout=TINY_LAYOUT,
        chunk_size=3,
    )

    np.testing.assert_array_equal(corpus.features(), values[:, 1:])
    np.testing.assert_array_equal(corpus.labels(), values[:, 0])
    assert not raw_path.exists()


def test_prepare_corpus_reuses_a_valid_cache_and_reports_progress(tmp_path):
    raw_path, cache_dir, corpus, _ = _prepare(tmp_path)
    output = StringIO()

    cached = prepare_corpus(
        raw_path,
        cache_dir,
        layout=TINY_LAYOUT,
        progress=True,
        progress_file=output,
    )

    assert cached == corpus
    assert "HIGGS corpus: 12 rows (cached)" in output.getvalue()


def test_statistics_ignore_validation_and_test_rows(tmp_path):
    first = _tiny_data()
    second = first.copy()
    second[TINY_LAYOUT.train_stop :, 1:] += 10_000
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()

    *_, first_corpus, _ = _prepare(first_path, first)
    *_, second_corpus, _ = _prepare(second_path, second)

    np.testing.assert_array_equal(first_corpus.feature_mean, second_corpus.feature_mean)
    np.testing.assert_array_equal(
        first_corpus.feature_scale, second_corpus.feature_scale
    )


def test_zero_training_variance_uses_unit_scale(tmp_path):
    values = _tiny_data()
    values[: TINY_LAYOUT.train_stop, 1] = 7.0
    *_, corpus, _ = _prepare(tmp_path, values)

    assert corpus.feature_mean[0] == 7.0
    assert corpus.feature_scale[0] == 1.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (_tiny_data()[:, :-1], "29 columns"),
        (_tiny_data()[:-1], "expected 12 rows"),
        (_tiny_data(13), "input has more"),
    ],
)
def test_prepare_rejects_wrong_shapes_without_publishing_cache(
    tmp_path, values, message
):
    raw_path = tmp_path / "HIGGS.csv"
    cache_dir = tmp_path / "cache"
    _write_csv(raw_path, values)

    with pytest.raises(ValueError, match=message):
        prepare_corpus(raw_path, cache_dir, layout=TINY_LAYOUT, chunk_size=3)

    assert not (cache_dir / METADATA_FILE).exists()
    assert not any(cache_dir.iterdir())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [(0, 2.0, "binary"), (1, np.nan, "finite")],
)
def test_prepare_rejects_invalid_values_without_publishing_cache(
    tmp_path, column, value, message
):
    values = _tiny_data()
    values[4, column] = value
    raw_path = tmp_path / "HIGGS.csv"
    cache_dir = tmp_path / "cache"
    _write_csv(raw_path, values)

    with pytest.raises(ValueError, match=message):
        prepare_corpus(raw_path, cache_dir, layout=TINY_LAYOUT, chunk_size=3)

    assert not (cache_dir / METADATA_FILE).exists()
    assert not any(cache_dir.iterdir())


def test_open_rejects_a_different_layout(tmp_path):
    _, cache_dir, _, _ = _prepare(tmp_path)

    with pytest.raises(ValueError, match="layout"):
        HiggsCorpus.open(cache_dir, layout=HiggsLayout(12, 7, 10))


def test_task_emits_sorted_training_batches_without_mutating_order(tmp_path):
    _, _, corpus, values = _prepare(tmp_path)
    order_path = corpus.shuffled_epochs(7).prepare(1)[0]
    task = HiggsTask(corpus, data_seed=7, batch_size=3)
    order = np.load(order_path, mmap_mode="r")
    original_order = np.array(order)
    mean = np.asarray(corpus.feature_mean, dtype=np.float32)
    scale = np.asarray(corpus.feature_scale, dtype=np.float32)

    batches = list(task.train_batches(5))

    assert all(len(model_inputs) == 1 for model_inputs, _ in batches)
    actual_features = np.concatenate(
        [model_inputs[0].numpy() for model_inputs, _ in batches]
    )
    actual_labels = np.concatenate([labels.numpy() for _, labels in batches])
    expected_rows = np.concatenate(
        [np.sort(original_order[:3]), np.sort(original_order[3:5])]
    )
    expected_features = values[expected_rows, 1:].astype(np.float32)
    expected_features -= mean
    expected_features /= scale
    np.testing.assert_allclose(actual_features, expected_features)
    np.testing.assert_array_equal(actual_labels, values[expected_rows, 0])
    np.testing.assert_array_equal(order, original_order)
    assert all(
        tensor.dtype == torch.float32
        for tensor in (batches[0][0][0], batches[0][1])
    )


def test_task_globally_batches_across_shuffled_pass_boundaries(tmp_path):
    _, cache_dir, corpus, values = _prepare(tmp_path)
    metadata = (cache_dir / METADATA_FILE).read_bytes()
    paths = corpus.shuffled_epochs(7).prepare(2)
    orders = [np.load(path, mmap_mode="r") for path in paths]
    global_order = np.concatenate(orders)
    task = HiggsTask(corpus, data_seed=7, batch_size=3)

    batches = list(task.train_batches(10))

    assert [len(labels) for _, labels in batches] == [3, 3, 3, 1]
    expected_rows = [
        np.sort(global_order[start : min(start + 3, 10)])
        for start in range(0, 10, 3)
    ]
    actual_labels = [labels.numpy() for _, labels in batches]
    for labels, rows in zip(actual_labels, expected_rows, strict=True):
        np.testing.assert_array_equal(labels, values[rows, 0])

    boundary_rows = np.sort(np.concatenate((orders[0][6:8], orders[1][:1])))
    np.testing.assert_array_equal(actual_labels[2], values[boundary_rows, 0])
    assert {path.parent.name for path in paths} == {"shuffle-v1"}
    assert (cache_dir / METADATA_FILE).read_bytes() == metadata


def test_task_uses_exact_sequential_holdout_boundaries(tmp_path):
    _, _, corpus, values = _prepare(tmp_path)
    task = HiggsTask(corpus, data_seed=3, batch_size=3)

    val_labels = np.concatenate([labels.numpy() for _, labels in task.val_batches()])
    test_labels = np.concatenate(
        [labels.numpy() for _, labels in task.test_batches()]
    )

    np.testing.assert_array_equal(
        val_labels, values[TINY_LAYOUT.train_stop : TINY_LAYOUT.val_stop, 0]
    )
    np.testing.assert_array_equal(test_labels, values[TINY_LAYOUT.val_stop :, 0])


def test_task_drops_open_memmaps_when_pickled(tmp_path):
    _, _, corpus, _ = _prepare(tmp_path)
    task = HiggsTask(corpus, data_seed=0, batch_size=3)
    next(task.val_batches())

    restored = pickle.loads(pickle.dumps(task))

    assert restored._arrays_cache is None
    assert next(restored.val_batches())[0][0].shape == (2, NUM_FEATURES)
