import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    GROUND_TRUTH,
)

from blocking.io import read_tsv, get_columns


def test_file(name, path):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    columns = get_columns(path)

    print("Columns:")
    for column in columns:
        print(f"  - {column}")

    sample = read_tsv(path, nrows=3)

    print("\nRows loaded:", len(sample))
    print(sample.to_string(index=False))


def main():
    test_file("SOURCE 1", S1_TRAIN)
    test_file("SOURCE 2", S2_TRAIN)
    test_file("SOURCE 3", S3_TRAIN)
    test_file("GROUND TRUTH", GROUND_TRUTH)


if __name__ == "__main__":
    main()