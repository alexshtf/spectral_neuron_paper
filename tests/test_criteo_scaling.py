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
    PROFILES,
    RAW_COLUMNS,
    ModelSpec,
    Profile,
    RunGrid,
    SeedGrid,
    build_arg_parser,
    default_raw_path,
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


def test_run_grid_declares_five_variants_with_matched_dimensions():
    grid = RunGrid(
        Profile(
            train_sizes=(16, 32),
            dims=(5,),
            lrs=(1e-2,),
            tuning_seeds=SeedGrid(),
            evaluation_seeds=SeedGrid(),
        )
    )

    assert grid.model_specs == (
        ModelSpec("linear"),
        ModelSpec("linear-new"),
        ModelSpec("fm", fm_rank=14, parameters_per_feature=15),
        ModelSpec("spectral-old", matrix_dim=5, parameters_per_feature=15),
        ModelSpec("spectral-new", matrix_dim=5, parameters_per_feature=15),
    )
    assert len(grid) == 5


def test_tiny_profile_runs_end_to_end(tmp_path):
    raw_path = tmp_path / "train.txt"
    cache_dir = tmp_path / "cache"
    _write_tiny_criteo(raw_path)

    corpus = prepare_corpus(raw_path, cache_dir, chunk_size=23)
    raw = run_profile(
        _tiny_profile(),
        raw_path=raw_path,
        cache_dir=cache_dir,
        chunk_size=23,
    )
    summary = summarize_raw(raw)

    assert corpus.rows == 100
    assert set(raw["model"]) == {
        "linear",
        "linear-new",
        "fm",
        "spectral-old",
        "spectral-new",
    }
    assert set(raw["preprocessing"]) == {"bucket", "hybrid"}
    assert set(raw["protocol"]) == {"one_pass"}
    assert set(raw["optimizer"]) == {"adam"}
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
    assert set(raw["preprocessing"]) == {"hybrid"}
    assert [path.name for path in cache_dir.glob("preprocessor_*.npz")] == [
        "preprocessor_hybrid_sample8_seed0_min2_b32.npz"
    ]
    assert default_raw_path("full", "linear-new").name == (
        "criteo_scaling_full_linear-new.csv"
    )


def test_variant_cli_defaults_to_all():
    parser = build_arg_parser()

    assert parser.parse_args(["--data", "train.txt"]).variant is None
    assert (
        parser.parse_args(
            ["--data", "train.txt", "--variant", "spectral-new"]
        ).variant
        == "spectral-new"
    )


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

    columns = [column for column in RAW_COLUMNS if column != "elapsed_seconds"]
    pd.testing.assert_frame_equal(serial[columns], parallel[columns])


def test_lr_selection_uses_validation_not_test():
    common = {
        "protocol": "one_pass",
        "optimizer": "adam",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_size": 32,
        "model": "linear",
        "preprocessing": "bucket",
        "matrix_dim": 0,
        "eig_idx": -1,
        "fm_rank": 0,
        "parameters_per_feature": 1,
        "num_parameters": 100,
    }
    raw = pd.DataFrame(
        [
            common | {"lr": 0.01, "val_logloss": 0.4, "test_logloss": 4.0},
            common | {"lr": 0.1, "val_logloss": 0.5, "test_logloss": 0.1},
        ]
    )

    selected = select_lr(raw)

    assert selected["selected_lr"].tolist() == [0.01]


def test_lr_selection_is_frozen_before_evaluation():
    common = {
        "protocol": "one_pass",
        "optimizer": "adam",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_size": 32,
        "model": "linear",
        "preprocessing": "bucket",
        "matrix_dim": 0,
        "eig_idx": -1,
        "fm_rank": 0,
        "parameters_per_feature": 1,
        "num_parameters": 100,
    }
    raw = pd.DataFrame(
        [
            common | {"phase": "tuning", "lr": 0.01, "val_logloss": 0.4},
            common | {"phase": "tuning", "lr": 0.1, "val_logloss": 0.5},
            common
            | {
                "phase": "evaluation",
                "lr": 0.01,
                "val_logloss": 9.0,
                "test_logloss": 4.0,
            },
            common
            | {
                "phase": "evaluation",
                "lr": 0.1,
                "val_logloss": 0.1,
                "test_logloss": 0.1,
            },
        ]
    )

    selected = select_lr(raw)

    assert selected[["lr", "median_val_logloss", "test_logloss"]].to_dict(
        "records"
    ) == [{"lr": 0.01, "median_val_logloss": 0.4, "test_logloss": 4.0}]


def test_full_profile_trades_scale_for_crossed_evaluation_seeds():
    profile = PROFILES["full"]

    assert profile.train_sizes == (
        2**11,
        2**13,
        2**15,
        2**17,
        2**19,
        2**21,
        2**22,
        36_672_493 // 8,
    )
    assert profile.tuning_seeds == SeedGrid(init_seeds=range(3))
    assert profile.evaluation_seeds == SeedGrid(
        data_seeds=range(1, 5),
        init_seeds=range(3, 9),
    )
    assert len(profile.evaluation_seeds) == 24
