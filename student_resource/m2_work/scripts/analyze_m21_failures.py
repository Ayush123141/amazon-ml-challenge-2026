import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import BLOCKING_100K_DIR


def main():

    path = BLOCKING_100K_DIR / "m21_failures.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("=" * 80)
    print("M2.1 FAILURE ANALYSIS")
    print("=" * 80)

    print(f"\nFailure rows: {len(df):,}")

    if df.empty:
        print("No failures.")
        return

    numeric_cols = [
        "true_match_count",
        "retrieved_match_count",
        "missing_match_count",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    print("\nMissing-match distribution:")
    print(
        df["missing_match_count"]
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    print("\nTrue-match count distribution:")
    print(
        df["true_match_count"]
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    print("\nRetrieved-match count distribution:")
    print(
        df["retrieved_match_count"]
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    print("\nExamples of failures:")
    print(
        df.head(20).to_string(index=False)
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "Failures with zero retrieved matches:",
        (df["retrieved_match_count"] == 0).sum()
    )

    print(
        "Failures with some retrieved matches:",
        (df["retrieved_match_count"] > 0).sum()
    )

    print(
        "Failures with exactly one missing match:",
        (df["missing_match_count"] == 1).sum()
    )

    print(
        "Failures with >1 missing match:",
        (df["missing_match_count"] > 1).sum()
    )


if __name__ == "__main__":
    main()