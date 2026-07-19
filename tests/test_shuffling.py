from pathlib import Path

import numpy as np
import pytest

from paper.shuffling import ShuffledEpochs, resolve_train_sizes


def _successive_orders(size: int, seed: int, passes: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    order = np.arange(size, dtype=np.uint32)
    result = []
    for _ in range(passes):
        rng.shuffle(order)
        result.append(order.copy())
    return tuple(result)


def test_shuffled_epochs_are_successive_full_permutations(tmp_path: Path):
    shuffled = ShuffledEpochs(tmp_path, size=101, seed=7)
    paths = shuffled.prepare(3)
    orders = tuple(np.load(path, allow_pickle=False) for path in paths)

    legacy = np.random.default_rng(7).permutation(101).astype(np.uint32)
    np.testing.assert_array_equal(orders[0], legacy)
    for actual, expected in zip(
        orders, _successive_orders(101, seed=7, passes=3), strict=True
    ):
        assert actual.dtype == np.uint32
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.sort(actual), np.arange(101))

    assert not np.array_equal(orders[0], orders[1])
    other_seed = np.load(
        ShuffledEpochs(tmp_path, size=101, seed=8).prepare(1)[0],
        allow_pickle=False,
    )
    assert not np.array_equal(orders[0], other_seed)


def test_preparing_more_passes_preserves_cached_prefix(tmp_path: Path):
    shuffled = ShuffledEpochs(tmp_path, size=17, seed=3)
    prefix_paths = shuffled.prepare(2)
    prefix_bytes = tuple(path.read_bytes() for path in prefix_paths)

    paths = shuffled.prepare(4)

    assert paths[:2] == prefix_paths
    assert tuple(path.read_bytes() for path in paths[:2]) == prefix_bytes
    for path, expected in zip(
        paths, _successive_orders(17, seed=3, passes=4), strict=True
    ):
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)


def test_invalid_cached_order_is_replaced_atomically(tmp_path: Path):
    shuffled = ShuffledEpochs(tmp_path, size=11, seed=2)
    path = shuffled.prepare(1)[0]
    with path.open("wb") as file:
        np.save(file, np.arange(10, dtype=np.uint32), allow_pickle=False)

    assert shuffled.prepare(1) == (path,)

    np.testing.assert_array_equal(
        np.load(path, allow_pickle=False),
        _successive_orders(11, seed=2, passes=1)[0],
    )
    assert not tuple(path.parent.glob(".*.tmp"))


def test_batches_are_global_across_pass_boundaries(tmp_path: Path):
    shuffled = ShuffledEpochs(tmp_path, size=5, seed=4)
    stream = np.concatenate(_successive_orders(5, seed=4, passes=3))

    batches = list(shuffled.batches(stop=12, batch_size=4))

    assert list(map(len, batches)) == [4, 4, 4]
    for start, actual in zip(range(0, 12, 4), batches, strict=True):
        np.testing.assert_array_equal(actual, np.sort(stream[start : start + 4]))
    assert stream[4] in batches[1]
    for row in stream[5:8]:
        assert row in batches[1]

    prefix = list(shuffled.batches(stop=8, batch_size=4))
    for actual, expected in zip(prefix, batches[:2], strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_resolve_train_sizes_preserves_explicit_batch_aligned_checkpoints():
    assert resolve_train_sizes((4, 8, 11), batch_size=4) == (4, 8, 11)
    assert resolve_train_sizes((1,), batch_size=8) == (1,)


def test_resolve_train_sizes_rejects_unaligned_nonterminal_checkpoints():
    with pytest.raises(ValueError, match="nonterminal train sizes"):
        resolve_train_sizes((1, 2, 5), batch_size=4)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda path: ShuffledEpochs(path, True, 0), "size"),
        (lambda path: ShuffledEpochs(path, 3, True), "seed"),
        (lambda path: list(ShuffledEpochs(path, 3, 0).batches(True, 1)), "stop"),
        (
            lambda _: resolve_train_sizes((True,), batch_size=1),
            "requested train size",
        ),
    ],
)
def test_shuffle_counts_reject_booleans(tmp_path: Path, factory, match: str):
    with pytest.raises(TypeError, match=match):
        factory(tmp_path)
