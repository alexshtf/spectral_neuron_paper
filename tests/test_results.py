import pandas as pd
import pytest

from paper.experiments.results import write_csv


def test_write_mode_appends_or_replaces_results(tmp_path):
    path = tmp_path / "results.csv"
    write_csv(pd.DataFrame({"value": [1]}), path, write_mode="append")
    write_csv(pd.DataFrame({"value": [2]}), path, write_mode="append")
    assert pd.read_csv(path)["value"].tolist() == [1, 2]

    write_csv(pd.DataFrame({"value": [3]}), path)
    assert pd.read_csv(path)["value"].tolist() == [3]


def test_zstandard_result_path_round_trips_through_pandas(tmp_path):
    path = tmp_path / "results.csv.zst"
    expected = pd.DataFrame({"value": [1, 2, 3]})

    write_csv(expected, path)

    pd.testing.assert_frame_equal(pd.read_csv(path), expected)


def test_append_rejects_an_incompatible_csv_schema(tmp_path):
    path = tmp_path / "results.csv"
    original = "value,metric\n1,2\n"
    path.write_text(original)

    with pytest.raises(ValueError, match="CSV header"):
        write_csv(
            pd.DataFrame({"metric": [3], "value": [4]}), path, write_mode="append"
        )

    assert path.read_text() == original
