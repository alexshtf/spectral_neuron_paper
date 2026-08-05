from compression import zstd
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paper.criteo as criteo
from paper.criteo import (
    MISSING_NUMERIC,
    NUM_CATEGORICAL_FIELDS,
    NUM_NUMERIC_FIELDS,
    BucketPreprocessor,
    CriteoTask,
    HybridPreprocessor,
    _bucket_numeric,
    fit_preprocessors,
    load_encoded,
    load_preprocessor,
    prepare_corpus,
    prepare_encoded_data,
)
from paper.experiments.criteo_scaling import (
    RAW_COLUMNS,
    Profile,
    SeedGrid,
    _best_lrs,
    build_arg_parser,
    default_raw_path,
    run_profile,
    select_lr,
    summarize_raw,
    validate_raw,
)


def _write_tiny_criteo(path: Path, rows: int = 100) -> None:
    lines = []
    for row in range(rows):
        label = int(row % 4 == 0)
        numeric = [str((row + field) % 20) for field in range(NUM_NUMERIC_FIELDS)]
        if row % 11 == 0:
            numeric[0] = "-1"
        if row % 17 == 0:
            numeric[1] = "-2"
        if row % 23 == 0:
            numeric[2] = ""
        categorical = [
            f"{(row + field) % (field + 3):08x}"
            for field in range(NUM_CATEGORICAL_FIELDS)
        ]
        lines.append("\t".join((str(label), *numeric, *categorical)))
    path.write_text("\n".join(lines) + "\n")


def _compress_zstd(path: Path) -> Path:
    compressed = path.with_name(f"{path.name}.zstd")
    compressed.write_bytes(zstd.compress(path.read_bytes(), level=3))
    path.unlink()
    return compressed


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(16, 32),
        dims=(3,),
        lrs=(1e-3, 1e-2),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=16,
        min_count=2,
    )


@pytest.fixture(scope="module")
def complete_raw(tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-criteo-run")
    raw_path = directory / "train.txt"
    _write_tiny_criteo(raw_path)
    return run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=directory / "cache",
    )


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


def test_cli_accepts_append_mode():
    args = build_arg_parser().parse_args(
        ["--data", "train.txt", "--write-mode", "append"]
    )

    assert args.write_mode == "append"


def test_default_path_is_protocol_specific():
    assert default_raw_path("full").name == ("criteo_scaling_full_repeated_shuffle.csv")
    assert default_raw_path("full", "linear-continuous").name == (
        "criteo_scaling_full_repeated_shuffle_linear-continuous.csv"
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
    assert preprocessor.num_numeric_features == 6 + 4 * (NUM_NUMERIC_FIELDS - 1)
    assert preprocessor.field_offsets.dtype == np.int32


def test_prepare_corpus_streams_zstd_source(tmp_path):
    raw_path = tmp_path / "train.txt"
    _write_tiny_criteo(raw_path, rows=17)
    compressed_path = _compress_zstd(raw_path)

    corpus = prepare_corpus(compressed_path, tmp_path / "cache", chunk_size=5)

    assert corpus.rows == 17
    np.testing.assert_array_equal(
        corpus.labels(),
        [int(row % 4 == 0) for row in range(17)],
    )
    assert not raw_path.exists()


def test_preprocessing_reports_progress(tmp_path, monkeypatch):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    write_levels = []
    zstd_open = zstd.open

    def tracked_open(file, mode="rb", **kwargs):
        if "w" in mode:
            write_levels.append(kwargs.get("level"))
        return zstd_open(file, mode, **kwargs)

    monkeypatch.setattr(criteo.zstd, "open", tracked_open)

    output = StringIO()
    corpus = prepare_corpus(
        raw_path,
        cache_dir,
        chunk_size=23,
        progress=True,
        progress_file=output,
    )
    paths = fit_preprocessors(
        corpus,
        ("bucket", "hybrid"),
        sample_size=8,
        sample_seed=7,
        min_count=2,
        progress=True,
        progress_file=output,
    )

    printed = output.getvalue()
    assert "Preparing Criteo corpus" in printed
    assert "Fitting categorical vocabulary on 8 rows" in printed
    assert "Fitting numeric preprocessing on 8 rows" in printed
    assert set(paths) == {"bucket", "hybrid"}
    assert write_levels == [3, 3]
    assert all(path.name.endswith(".pkl.zstd") for path in paths.values())
    assert all(
        path.read_bytes().startswith(b"\x28\xb5\x2f\xfd") for path in paths.values()
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


def test_legacy_preprocessor_cache_is_migrated_to_zstd(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)
    corpus = prepare_corpus(raw_path, cache_dir)
    path = fit_preprocessors(
        corpus,
        ("bucket",),
        sample_size=8,
        sample_seed=7,
        min_count=2,
    )["bucket"]
    with zstd.open(path, "rb") as file:
        pickle_bytes = file.read()
    legacy = path.with_suffix("")
    legacy.write_bytes(pickle_bytes)
    path.unlink()

    migrated = fit_preprocessors(
        corpus,
        ("bucket",),
        sample_size=8,
        sample_seed=7,
        min_count=2,
    )["bucket"]

    assert migrated == path
    assert path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert isinstance(load_preprocessor(path), BucketPreprocessor)


def test_encoded_cache_uses_local_ids_and_tasks_gather_shuffled_batches(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path, rows=103)
    corpus = prepare_corpus(raw_path, cache_dir)
    paths = fit_preprocessors(
        corpus,
        ("bucket", "hybrid"),
        sample_size=8,
        sample_seed=0,
        min_count=2,
    )
    batch_size = 4
    order = corpus.shuffled_epochs(7)
    order.prepare(2)

    for kind, path in paths.items():
        preprocessor = load_preprocessor(path)
        data = prepare_encoded_data(
            corpus,
            path,
            chunk_size=3,
        )
        sources = (
            (data.train, slice(0, corpus.train_stop)),
            (data.holdout, slice(corpus.train_stop, corpus.rows)),
        )
        for split, rows in sources:
            local_ids, feature_values, labels = load_encoded(split)
            assert local_ids.dtype == np.uint16
            _assert_arrays(
                (
                    _global_ids(local_ids, data.field_offsets),
                    feature_values,
                    labels,
                ),
                (
                    *preprocessor.encode(
                        np.asarray(corpus.numerics()[rows]),
                        np.asarray(corpus.categoricals()[rows]),
                    ),
                    np.asarray(corpus.labels()[rows]),
                ),
            )

        task = CriteoTask(data, order, batch_size)
        train_size = corpus.train_stop + 3
        batches = list(task.train_batches(train_size))
        actual_ids = np.concatenate(
            [model_inputs[0].numpy() for model_inputs, _ in batches]
        )
        actual_values = (
            None
            if kind == "bucket"
            else np.concatenate(
                [model_inputs[1].numpy() for model_inputs, _ in batches]
            )
        )
        actual_labels = np.concatenate([labels.numpy() for _, labels in batches])
        assert actual_ids.dtype == np.int32
        expected_rows = np.concatenate(list(order.batches(train_size, batch_size)))
        expected_ids, expected_values, expected_labels = load_encoded(data.train)
        _assert_arrays(
            (actual_ids, actual_values, actual_labels),
            (
                _global_ids(expected_ids[expected_rows], data.field_offsets),
                None if expected_values is None else expected_values[expected_rows],
                expected_labels[expected_rows],
            ),
        )

        if kind == "bucket":
            val_labels = np.concatenate(
                [labels.numpy() for _, labels in task.val_batches()]
            )
            test_labels = np.concatenate(
                [labels.numpy() for _, labels in task.test_batches()]
            )
            labels = corpus.labels()
            np.testing.assert_array_equal(
                val_labels, labels[corpus.train_stop : corpus.val_stop]
            )
            np.testing.assert_array_equal(test_labels, labels[corpus.val_stop :])

            repeated = prepare_encoded_data(corpus, path, chunk_size=3)
            assert repeated.train == data.train
            assert (data.train / "complete").exists()
            assert data.train.with_name(f".{data.train.name}.lock").exists()
            assert not list(data.train.parent.glob(f".{data.train.name}-*"))

            (data.train / "labels.npy").write_bytes(b"corrupt")
            recovered = prepare_encoded_data(corpus, path, chunk_size=3)
            np.testing.assert_array_equal(
                load_encoded(recovered.train)[2],
                corpus.labels()[: corpus.train_stop],
            )
            (recovered.train / "complete").unlink()
            recovered = prepare_encoded_data(corpus, path, chunk_size=3)
            assert (recovered.train / "complete").exists()


def test_profile_evaluates_only_its_requested_train_sizes(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)
    profile = Profile(
        train_sizes=(16, 161),
        dims=(3,),
        lrs=(1e-3,),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=16,
        min_count=2,
    )

    raw = run_profile(
        profile,
        raw_path=raw_path,
        cache_dir=cache_dir,
        variant="linear-bucketed",
    )

    assert raw["train_pool_size"].unique().item() == 80
    assert set(raw["train_size"]) == {16, 161}
    assert raw.groupby("phase")["train_size"].nunique().eq(2).all()


def test_tiny_profile_runs_end_to_end(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    output = StringIO()
    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        chunk_size=23,
        progress=True,
        progress_file=output,
    )
    summary = summarize_raw(raw)

    assert set(raw["model"]) == {
        "linear-bucketed",
        "linear-continuous",
        "fm",
        "spectral-bucketed",
        "spectral-continuous",
    }
    assert set(raw["protocol"]) == {"repeated_shuffle"}
    assert set(raw["optimizer"]) == {"adam+sparseadam"}
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["preprocessor_sample_size"]) == {8}
    assert set(raw["train_pool_size"]) == {80}
    assert set(RAW_COLUMNS) == set(raw.columns)
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert np.isfinite(tuning["val_logloss"]).all()
    assert evaluation["val_logloss"].isna().all()
    assert (
        raw.groupby(["phase", "data_seed", "model", "lr", "init_seed"])["train_size"]
        .nunique()
        .eq(2)
        .all()
    )
    assert raw["test_logloss"].notna().sum() == 10
    assert raw["test_brier"].notna().sum() == 10
    assert {"q25_test_brier", "q75_test_brier"} <= set(summary)
    assert len(summary) == 10
    printed = output.getvalue()
    assert "Encoding bucket train" in printed
    assert "Encoding bucket holdout" in printed
    assert "Encoding hybrid holdout" in printed
    assert "Tuning aggregate trajectory time: training=" in printed
    assert "validation=" in printed
    assert "Evaluation aggregate trajectory time: training=" in printed
    assert "test=" in printed


def test_variant_run_uses_only_its_preprocessor(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)
    legacy = cache_dir / "encoded-v3" / "legacy"
    legacy.mkdir(parents=True)

    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        variant="linear-continuous",
    )

    assert set(raw["model"]) == {"linear-continuous"}
    assert len(list(cache_dir.glob("preprocessor-v*_*.pkl.zstd"))) == 1
    assert legacy.exists()
    assert len(list((cache_dir / "encoded-v4").iterdir())) == 1


def test_validate_raw_accepts_complete_and_variant_sharded_results(complete_raw):
    validate_raw(complete_raw, _tiny_profile())
    linear = complete_raw.loc[
        complete_raw["model"] == "linear-bucketed"
    ].copy()
    validate_raw(linear, _tiny_profile(), variant="linear-bucketed")


def test_validate_raw_rejects_identity_and_incomplete_grids(complete_raw):
    with pytest.raises(ValueError, match="schema"):
        validate_raw(complete_raw.drop(columns="test_brier"), _tiny_profile())

    noninteger_pool = complete_raw.copy()
    noninteger_pool["train_pool_size"] = noninteger_pool["train_pool_size"].astype(
        float
    )
    with pytest.raises(ValueError, match="positive integer train_pool_size"):
        validate_raw(noninteger_pool, _tiny_profile())

    duplicate = pd.concat((complete_raw, complete_raw.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw(duplicate, _tiny_profile())

    incomplete = complete_raw.drop(
        complete_raw.index[complete_raw["phase"] == "tuning"][0]
    )
    with pytest.raises(ValueError, match="tuning"):
        validate_raw(incomplete, _tiny_profile())


def test_validate_raw_checks_metrics_and_checkpoint_selected_lrs(complete_raw):
    nonfinite = complete_raw.copy()
    index = nonfinite.index[nonfinite["phase"] == "evaluation"][0]
    nonfinite.loc[index, "test_logloss"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_raw(nonfinite, _tiny_profile())

    wrong_lr = complete_raw.copy()
    row = wrong_lr.loc[wrong_lr["phase"] == "evaluation"].iloc[0]
    mask = (
        (wrong_lr["phase"] == "evaluation")
        & (wrong_lr["model"] == row["model"])
        & (wrong_lr["dim"] == row["dim"])
        & (wrong_lr["train_size"] == row["train_size"])
    )
    alternate_lr = next(lr for lr in _tiny_profile().lrs if lr != row["lr"])
    wrong_lr.loc[mask, "lr"] = alternate_lr
    with pytest.raises(ValueError, match="selected LR"):
        validate_raw(wrong_lr, _tiny_profile())


def test_best_lrs_filters_nonfinite_and_requires_one_per_checkpoint(complete_raw):
    tuning = complete_raw.loc[complete_raw["phase"] == "tuning"].copy()
    row = tuning.iloc[0]
    curve = (
        (tuning["model"] == row["model"])
        & (tuning["dim"] == row["dim"])
        & (tuning["train_size"] == row["train_size"])
    )
    lr = tuning.loc[curve, "lr"].iloc[0]
    tuning.loc[curve & (tuning["lr"] == lr), "val_logloss"] = np.inf

    with pytest.warns(RuntimeWarning, match="nonfinite"):
        best = _best_lrs(tuning)
    selected = best.loc[
        (best["model"] == row["model"])
        & (best["dim"] == row["dim"])
        & (best["train_size"] == row["train_size"]),
        "selected_lr",
    ]
    assert selected.item() != lr

    tuning.loc[curve, "val_logloss"] = np.nan
    with (
        pytest.warns(RuntimeWarning, match="nonfinite"),
        pytest.raises(ValueError, match="no finite validation"),
    ):
        _best_lrs(tuning)


def test_parallel_profile_matches_serial_results(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    serial = run_profile(
        _tiny_profile(), raw_path=raw_path, cache_dir=cache_dir, workers=1
    )
    parallel = run_profile(
        _tiny_profile(), raw_path=raw_path, cache_dir=cache_dir, workers=2
    )

    pd.testing.assert_frame_equal(serial, parallel)


def test_lr_selection_uses_median_tuning_validation_only():
    common = {
        "protocol": "repeated_shuffle",
        "optimizer": "adam+sparseadam",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_pool_size": 80,
        "train_size": 32,
        "model": "linear-bucketed",
        "dim": 0,
        "data_seed": 0,
    }
    tuning = [
        common
        | {
            "phase": "tuning",
            "lr": lr,
            "init_seed": seed,
            "val_logloss": score,
        }
        for lr, scores in ((0.01, (0.1, 10.0, 10.0)), (0.1, (1.0, 1.0, 100.0)))
        for seed, score in enumerate(scores)
    ]
    evaluation = [
        common
        | {
            "phase": "evaluation",
            "lr": lr,
            "init_seed": 3,
            "val_logloss": val,
            "test_logloss": test,
        }
        for lr, val, test in ((0.01, 0.1, 0.01), (0.1, 9.0, 4.0))
    ]

    selected = select_lr(pd.DataFrame(tuning + evaluation))

    assert selected[["lr", "median_val_logloss", "test_logloss"]].to_dict(
        "records"
    ) == [{"lr": 0.1, "median_val_logloss": 1.0, "test_logloss": 4.0}]


def test_lr_selection_keeps_training_pools_separate():
    rows = []
    for train_pool_size, selected_lr in ((40, 0.01), (80, 0.1)):
        common = {
            "protocol": "repeated_shuffle",
            "optimizer": "adam+sparseadam",
            "preprocessor_sample_size": 8,
            "preprocessor_seed": 0,
            "train_pool_size": train_pool_size,
            "train_size": 32,
            "model": "linear-bucketed",
            "dim": 0,
            "data_seed": 0,
            "init_seed": 0,
        }
        for lr in (0.01, 0.1):
            rows.append(
                common
                | {
                    "phase": "tuning",
                    "lr": lr,
                    "val_logloss": float(lr != selected_lr),
                }
            )
            rows.append(
                common
                | {
                    "phase": "evaluation",
                    "lr": lr,
                    "test_logloss": train_pool_size + lr,
                }
            )

    selected = select_lr(pd.DataFrame(rows))

    assert selected[["train_pool_size", "lr"]].to_dict("records") == [
        {"train_pool_size": 40, "lr": 0.01},
        {"train_pool_size": 80, "lr": 0.1},
    ]
