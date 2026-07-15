from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from paper.criteo import (
    MISSING_NUMERIC,
    NUM_CATEGORICAL_FIELDS,
    NUM_NUMERIC_FIELDS,
    BucketPreprocessor,
    HybridPreprocessor,
    _bucket_numeric,
    fit_preprocessors,
    load_preprocessor,
    prepare_corpus,
)
from paper.experiments.criteo_scaling import (
    RAW_COLUMNS,
    Profile,
    SeedGrid,
    build_arg_parser,
    run_profile,
    select_lr,
    summarize_raw,
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


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(16, 32),
        dims=(3,),
        lrs=(1e-3, 1e-2),
        tuning_seeds=SeedGrid(),
        evaluation_seeds=SeedGrid(),
        batch_size=16,
        min_count=2,
        buckets_per_field=32,
    )


def _frequent_categories() -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([7], dtype=np.uint32) for _ in range(NUM_CATEGORICAL_FIELDS)
    )


def test_cli_accepts_append_mode():
    args = build_arg_parser().parse_args(
        ["--data", "train.txt", "--write-mode", "append"]
    )

    assert args.write_mode == "append"


def test_winner_style_numeric_buckets():
    values = np.array([np.nan, -1, 0, 1, 2, 3, np.e**3])
    buckets = _bucket_numeric(values)

    assert buckets[:6].tolist() == [0, 2, 1, 3, 5, (1 << 31) | 1]
    assert buckets[6] == (1 << 31) | 9


def test_bucket_preprocessor_uses_field_disjoint_hash_ranges():
    preprocessor = BucketPreprocessor(16, _frequent_categories())
    numerics = np.full((2, NUM_NUMERIC_FIELDS), 7, dtype=np.int32)
    categoricals = np.full((2, NUM_CATEGORICAL_FIELDS), 7, dtype=np.uint32)
    categoricals[0, 0] = 8
    categoricals[1, 1] = 0

    feature_ids, feature_values = preprocessor.encode(numerics, categoricals)

    offsets = np.arange(NUM_NUMERIC_FIELDS + NUM_CATEGORICAL_FIELDS) * 16
    assert np.all(feature_ids >= offsets)
    assert np.all(feature_ids < offsets + 16)
    assert feature_ids[0, NUM_NUMERIC_FIELDS] == NUM_NUMERIC_FIELDS * 16 + 1
    assert feature_ids[1, NUM_NUMERIC_FIELDS + 1] == (NUM_NUMERIC_FIELDS + 1) * 16
    assert np.all(feature_values == 1)


def test_hybrid_preprocessor_separates_special_and_positive_values():
    negatives = (np.array([-2, -1], dtype=np.int32),) + tuple(
        np.array([], dtype=np.int32) for _ in range(NUM_NUMERIC_FIELDS - 1)
    )
    preprocessor = HybridPreprocessor(
        16,
        _frequent_categories(),
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


def test_preprocessing_reports_progress(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

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
        buckets_per_field=32,
        progress=True,
        progress_file=output,
    )

    printed = output.getvalue()
    assert "Preparing Criteo corpus" in printed
    assert "Fitting categorical vocabulary on 8 rows" in printed
    assert "Fitting hybrid numerics on 8 rows" in printed
    assert set(paths) == {"bucket", "hybrid"}

    hybrid = load_preprocessor(paths["hybrid"])
    rows = np.random.default_rng(7).choice(
        corpus.train_stop, size=8, replace=False, shuffle=False
    )
    sampled = corpus.numerics()[rows, 0]
    positive = np.log1p(sampled[sampled > 0])
    assert isinstance(hybrid, HybridPreprocessor)
    np.testing.assert_allclose(hybrid.positive_mean[0], positive.mean())
    scale = positive.std()
    np.testing.assert_allclose(
        hybrid.positive_scale[0], scale if scale > 0 else 1.0
    )


def test_training_order_is_a_reproducible_permutation_of_the_full_prefix(tmp_path):
    raw_path = tmp_path / "train.txt"
    _write_tiny_criteo(raw_path)
    corpus = prepare_corpus(raw_path, tmp_path / "cache")

    order = np.load(corpus.order_path(7))

    np.testing.assert_array_equal(np.sort(order), np.arange(corpus.train_stop))
    assert np.any(order[:8] >= 8)
    np.testing.assert_array_equal(order, np.load(corpus.order_path(7)))


def test_tiny_profile_runs_end_to_end(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        chunk_size=23,
    )
    summary = summarize_raw(raw)

    assert set(raw["model"]) == {
        "linear",
        "linear-new",
        "fm",
        "spectral-old",
        "spectral-new",
    }
    assert set(raw["protocol"]) == {"one_pass"}
    assert set(raw["optimizer"]) == {"adam+sparseadam"}
    assert set(raw["phase"]) == {"tuning", "evaluation"}
    assert set(raw["preprocessor_sample_size"]) == {8}
    assert set(RAW_COLUMNS) == set(raw.columns)
    tuning = raw.loc[raw["phase"] == "tuning"]
    evaluation = raw.loc[raw["phase"] == "evaluation"]
    assert np.isfinite(tuning["val_logloss"]).all()
    assert evaluation["val_logloss"].isna().all()
    assert (
        raw.groupby(["phase", "data_seed", "model", "lr", "init_seed"])[
            "train_size"
        ]
        .nunique()
        .eq(2)
        .all()
    )
    assert raw["test_logloss"].notna().sum() == 10
    assert raw["test_brier"].notna().sum() == 10
    assert {"q25_test_brier", "q75_test_brier"} <= set(summary)
    assert len(summary) == 10
    assert len(list(cache_dir.glob("preprocessor_*.npz"))) == 2


def test_variant_run_uses_only_its_preprocessor(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        variant="linear-new",
    )

    assert set(raw["model"]) == {"linear-new"}
    assert len(list(cache_dir.glob("preprocessor_*.npz"))) == 1


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
        "protocol": "one_pass",
        "optimizer": "adam+sparseadam",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_size": 32,
        "model": "linear",
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
