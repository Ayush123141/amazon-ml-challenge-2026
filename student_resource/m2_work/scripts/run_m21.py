import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    BLOCKING_100K_DIR,
    S1_SAMPLE_SIZE,
)

from blocking.io import read_tsv
from blocking.indexes import build_all_indexes
from blocking.blocking import get_candidates


def main():

    print("=" * 80)
    print("M2.1 EXACT BLOCKING")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load S2 + S3
    # ---------------------------------------------------------

    print("\n[1/4] Loading S2 + S3...")

    s2 = read_tsv(S2_TRAIN)
    s3 = read_tsv(S3_TRAIN)

    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    rows = list(s2.to_dict("records"))
    rows.extend(s3.to_dict("records"))

    print(f"Combined rows: {len(rows):,}")

    # ---------------------------------------------------------
    # 2. Build indexes
    # ---------------------------------------------------------

    print("\n[2/4] Building indexes...")

    indexes = build_all_indexes(rows)

    for name, index in indexes.items():
        print(f"{name:15s}: {len(index):,} keys")

    # ---------------------------------------------------------
    # 3. Load S1 sample
    # ---------------------------------------------------------

    print("\n[3/4] Loading S1 sample...")

    s1 = read_tsv(
        S1_TRAIN,
        nrows=S1_SAMPLE_SIZE,
    )

    print(f"S1 rows processed: {len(s1):,}")

    # ---------------------------------------------------------
    # 4. Generate candidates
    # ---------------------------------------------------------

    print("\n[4/4] Generating candidates...")

    results = []

    total_candidates = 0
    zero_candidates = 0

    for i, row in enumerate(
        s1.to_dict("records"),
        start=1,
    ):

        candidates = get_candidates(
            row,
            indexes,
        )

        candidates = sorted(candidates)

        if not candidates:
            zero_candidates += 1

        total_candidates += len(candidates)

        results.append(
            {
                "source1_entity_id": row["entity_id"],
                "candidate_entity_ids": ",".join(candidates),
            }
        )

        if i % 10_000 == 0:
            print(
                f"Processed {i:,}/{len(s1):,} "
                f"| candidates generated: {total_candidates:,}"
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    BLOCKING_100K_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        BLOCKING_100K_DIR
        / "candidate_pairs_m21.tsv"
    )

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    mean_candidates = (
        total_candidates / len(s1)
        if len(s1) > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("M2.1 COMPLETE")
    print("=" * 80)

    print(f"S1 processed       : {len(s1):,}")
    print(f"Total candidates   : {total_candidates:,}")
    print(f"Mean candidates    : {mean_candidates:.2f}")
    print(f"Zero-candidate S1  : {zero_candidates:,}")
    print(f"Output             : {output_path}")


if __name__ == "__main__":
    main()