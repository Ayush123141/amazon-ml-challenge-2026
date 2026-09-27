import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    GROUND_TRUTH,
)


def inspect_file(name, path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MB")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=5,
    )

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    print("\nColumn types:")
    print(df.dtypes)

    print()


def main():
    inspect_file("SOURCE 1", S1_TRAIN)
    inspect_file("SOURCE 2", S2_TRAIN)
    inspect_file("SOURCE 3", S3_TRAIN)
    inspect_file("GROUND TRUTH", GROUND_TRUTH)


if __name__ == "__main__":
    main()