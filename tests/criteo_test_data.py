from pathlib import Path

from paper.criteo import NUM_CATEGORICAL_FIELDS, NUM_NUMERIC_FIELDS


def write_tiny_criteo(path: Path, rows: int = 100) -> None:
    lines = []
    for row in range(rows):
        label = int(row % 4 == 0)
        numerics = [
            str((row + field_index) % 20)
            for field_index in range(NUM_NUMERIC_FIELDS)
        ]
        if row % 11 == 0:
            numerics[0] = "-1"
        if row % 17 == 0:
            numerics[1] = "-2"
        if row % 23 == 0:
            numerics[2] = ""
        categoricals = [
            f"{(row + field_index) % (field_index + 3):08x}"
            for field_index in range(NUM_CATEGORICAL_FIELDS)
        ]
        lines.append("\t".join((str(label), *numerics, *categoricals)))
    path.write_text("\n".join(lines) + "\n")
