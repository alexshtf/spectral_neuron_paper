from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from paper.criteo import (
    NUM_FIELDS,
    CriteoPreprocessor,
    _bucket_numeric,
    fit_preprocessor,
    prepare_corpus,
)
from paper.experiments.criteo_scaling import (
    RAW_COLUMNS,
    ModelSpec,
    Profile,
    RunGrid,
    run_profile,
    select_lr,
    summarize_raw,
)


def _write_tiny_criteo(path: Path, rows: int = 100) -> None:
    lines = []
    for row in range(rows):
        label = int(row % 4 == 0)
        numeric = [str((row + field) % 20) for field in range(13)]
        categorical = [f"{(row + field) % (field + 3):08x}" for field in range(26)]
        lines.append("\t".join((str(label), *numeric, *categorical)))
    path.write_text("\n".join(lines) + "\n")


def _tiny_profile() -> Profile:
    return Profile(
        train_sizes=(16, 32),
        dims=(3,),
        lrs=(1e-3, 1e-2),
        init_seeds=range(1),
        batch_size=16,
        min_count=2,
        buckets_per_field=32,
    )


def test_winner_style_numeric_buckets():
    values = np.array([np.nan, -1, 0, 1, 2, 3, np.e**3])
    buckets = _bucket_numeric(values)

    assert buckets[:6].tolist() == [0, 2, 1, 3, 5, (1 << 31) | 1]
    assert buckets[6] == (1 << 31) | 9


def test_preprocessor_uses_field_disjoint_hash_ranges():
    frequent = tuple(np.array([7], dtype=np.uint32) for _ in range(26))
    preprocessor = CriteoPreprocessor(16, frequent)
    tokens = np.full((2, NUM_FIELDS), 7, dtype=np.uint32)
    tokens[0, 13] = 8
    tokens[1, 14] = 0

    encoded = preprocessor.encode(tokens)

    assert np.all(encoded >= np.arange(NUM_FIELDS) * 16)
    assert np.all(encoded < (np.arange(NUM_FIELDS) + 1) * 16)
    assert encoded[0, 13] == 13 * 16 + 1
    assert encoded[1, 14] == 14 * 16


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
    preprocessor_path = fit_preprocessor(
        corpus,
        sample_size=8,
        sample_seed=7,
        min_count=2,
        buckets_per_field=32,
        progress=True,
        progress_file=output,
    )

    printed = output.getvalue()
    assert "Preparing Criteo corpus" in printed
    assert "Fitting preprocessor on 8 rows" in printed
    assert "sample8_seed7" in preprocessor_path.name


def test_run_grid_parameter_matches_fm_and_spectral():
    grid = RunGrid(
        Profile(
            train_sizes=(16, 32),
            dims=(5,),
            lrs=(1e-2,),
            init_seeds=range(1),
        )
    )

    assert grid.model_specs == (
        ModelSpec("linear"),
        ModelSpec("fm", fm_rank=14, parameters_per_feature=15),
        ModelSpec("spectral", matrix_dim=5, parameters_per_feature=15),
    )
    assert len(grid) == 3


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
    assert set(raw["model"]) == {"linear", "fm", "spectral"}
    assert set(raw["protocol"]) == {"one_pass"}
    assert set(raw["preprocessor_sample_size"]) == {8}
    assert set(RAW_COLUMNS) == set(raw.columns)
    assert np.isfinite(raw["val_logloss"]).all()
    assert (
        raw.groupby(["data_seed", "model", "lr", "init_seed"])["train_size"]
        .nunique()
        .eq(2)
        .all()
    )
    assert raw["test_logloss"].notna().sum() == 6
    assert raw["test_brier"].notna().sum() == 6
    assert {"q25_test_brier", "q75_test_brier"} <= set(summary)
    assert len(summary) == 6
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

    columns = [column for column in RAW_COLUMNS if column != "elapsed_seconds"]
    pd.testing.assert_frame_equal(serial[columns], parallel[columns])


def test_lr_selection_uses_validation_not_test():
    common = {
        "protocol": "one_pass",
        "preprocessor_sample_size": 8,
        "preprocessor_seed": 0,
        "train_size": 32,
        "model": "linear",
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
