import sys
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

M2_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(M2_ROOT / "src")
)


# ============================================================
# IMPORT THE PROVEN M2.2 IMPLEMENTATION
# ============================================================

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    DF_THRESHOLDS,
    S1_SAMPLE_SIZE,
)

from blocking.io import read_tsv

from blocking.token_index import (
    build_token_frequency,
    build_rare_token_index,
)

from blocking.normalization import (
    normalize_country,
    normalize_name,
    normalize_address,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Use the exact M2.2 threshold that gave us the best result.
DF_THRESHOLD = 50

# First run: validate on the same 100K scale.
MAX_S1_ROWS = 100_000

# Output
OUTPUT_DIR = M2_ROOT / "outputs"
REPORT_DIR = M2_ROOT / "reports"

OUTPUT_PATH = (
    OUTPUT_DIR /
    "candidate_pairs.tsv"
)

REPORT_PATH = (
    REPORT_DIR /
    "candidate_pairs_report.txt"
)


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def get_rare_token_candidates(
    row,
    rare_index,
):
    """
    EXACT M2.2 candidate retrieval logic.

    Uses:
        - country
        - unique business-name tokens
        - unique business-address tokens

    Candidate is retrieved when:
        country + token
        exists in the rare-token index.
    """

    country = normalize_country(
        row["country"]
    )

    if not country:
        return set()

    # ----------------------------
    # Business name tokens
    # ----------------------------

    name_normalized = normalize_name(
        row["business_name"]
    )

    name_tokens = set(
        name_normalized.split()
    ) if name_normalized else set()

    # ----------------------------
    # Business address tokens
    # ----------------------------

    address_normalized = normalize_address(
        row["business_address"]
    )

    address_tokens = set(
        address_normalized.split()
    ) if address_normalized else set()

    # ----------------------------
    # EXACT M2.2:
    # union name + address tokens
    # ----------------------------

    tokens = (
        name_tokens |
        address_tokens
    )

    candidates = set()

    for token in tokens:

        key = (
            country,
            token
        )

        candidates.update(
            rare_index.get(
                key,
                []
            )
        )

    return candidates


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values,
    p
):

    if not values:
        return 0.0

    return float(
        pd.Series(values).quantile(
            p / 100
        )
    )


# ============================================================
# BUILD CANDIDATE FILE
# ============================================================

def main():

    print("=" * 80)
    print("M2 CANDIDATE PAIRS GENERATION")
    print("=" * 80)

    print()
    print("Configuration")
    print("----------------------------")
    print(
        f"DF threshold : {DF_THRESHOLD}"
    )
    print(
        f"S1 rows      : {MAX_S1_ROWS:,}"
    )

    # ========================================================
    # 1. LOAD S2 + S3
    # ========================================================

    print()
    print("=" * 80)
    print("[1/5] Loading S2 + S3")
    print("=" * 80)

    s2 = read_tsv(
        S2_TRAIN
    )

    s3 = read_tsv(
        S3_TRAIN
    )

    print(
        f"S2 rows : {len(s2):,}"
    )

    print(
        f"S3 rows : {len(s3):,}"
    )

    print(
        f"Combined: "
        f"{len(s2) + len(s3):,}"
    )

    # ========================================================
    # 2. BUILD TOKEN FREQUENCY
    # ========================================================

    print()
    print("=" * 80)
    print("[2/5] Building token frequency")
    print("=" * 80)

    rows = list(
        s2.to_dict("records")
    )

    rows.extend(
        s3.to_dict("records")
    )

    token_frequency = (
        build_token_frequency(
            rows
        )
    )

    print(
        f"Token-frequency keys: "
        f"{len(token_frequency):,}"
    )

    # ========================================================
    # 3. BUILD RARE TOKEN INDEX
    # ========================================================

    print()
    print("=" * 80)
    print("[3/5] Building rare-token index")
    print("=" * 80)

    rare_index = (
        build_rare_token_index(
            rows,
            token_frequency,
            DF_THRESHOLD
        )
    )

    print(
        f"Rare index keys: "
        f"{len(rare_index):,}"
    )

    # ========================================================
    # 4. LOAD S1
    # ========================================================

    print()
    print("=" * 80)
    print("[4/5] Loading S1")
    print("=" * 80)

    s1 = read_tsv(
        S1_TRAIN
    )

    print(
        f"Full S1 rows: "
        f"{len(s1):,}"
    )

    if MAX_S1_ROWS is not None:

        s1 = s1.iloc[
            :MAX_S1_ROWS
        ].copy()

    print(
        f"S1 rows used: "
        f"{len(s1):,}"
    )

    # ========================================================
    # 5. GENERATE CANDIDATES
    # ========================================================

    print()
    print("=" * 80)
    print("[5/5] Generating candidate_pairs.tsv")
    print("=" * 80)

    output_rows = []

    candidate_counts = []

    zero_candidates = 0

    total_s1 = len(s1)

    for i, row in enumerate(
        s1.to_dict("records"),
        start=1
    ):

        s1_id = str(
            row["entity_id"]
        )

        candidates = (
            get_rare_token_candidates(
                row,
                rare_index
            )
        )

        # Defensive protection.
        candidates.discard(
            s1_id
        )

        candidates = sorted(
            candidates
        )

        count = len(
            candidates
        )

        candidate_counts.append(
            count
        )

        if count == 0:
            zero_candidates += 1

        output_rows.append(
            {
                "source1_entity_id":
                    s1_id,

                "candidate_entity_ids":
                    ",".join(candidates)
            }
        )

        if i % 10_000 == 0:

            print(
                f"Processed "
                f"{i:,}/{total_s1:,} "
                f"({i / total_s1:.1%})"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    candidate_df = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids"
        ]
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("VALIDATING OUTPUT")
    print("=" * 80)

    # One row per S1.
    assert len(candidate_df) == len(s1)

    # No duplicate S1 IDs.
    assert not (
        candidate_df[
            "source1_entity_id"
        ].duplicated().any()
    )

    # No empty S1 IDs.
    assert not (
        candidate_df[
            "source1_entity_id"
        ].astype(str)
        .str.strip()
        .eq("")
        .any()
    )

    # No duplicate candidate IDs inside rows.
    duplicate_candidate_rows = 0

    for value in candidate_df[
        "candidate_entity_ids"
    ]:

        if not value:
            continue

        ids = value.split(",")

        if len(ids) != len(set(ids)):

            duplicate_candidate_rows += 1

    assert duplicate_candidate_rows == 0

    print(
        "Structural validation: PASSED"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    candidate_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_candidates = sum(
        candidate_counts
    )

    mean_candidates = (
        total_candidates /
        total_s1
    )

    median_candidates = float(
        pd.Series(
            candidate_counts
        ).median()
    )

    p95 = percentile(
        candidate_counts,
        95
    )

    p99 = percentile(
        candidate_counts,
        99
    )

    maximum = max(
        candidate_counts
    )

    zero_pct = (
        zero_candidates /
        total_s1
    )

    report = f"""
M2 CANDIDATE PAIRS REPORT
=========================

Configuration
-------------
DF threshold       : {DF_THRESHOLD}
S1 rows processed  : {total_s1:,}

Candidate statistics
--------------------
Total candidate links : {total_candidates:,}
Mean candidates/S1    : {mean_candidates:.4f}
Median candidates/S1  : {median_candidates:.4f}
P95                   : {p95:.4f}
P99                   : {p99:.4f}
Maximum               : {maximum:,}

Zero-candidate S1
-----------------
Count                 : {zero_candidates:,}
Percentage            : {zero_pct:.4%}

Output
------
{OUTPUT_PATH}
"""

    print(report)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report
        )

    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print("=" * 80)
    print("OUTPUT PREVIEW")
    print("=" * 80)

    print(
        candidate_df.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        f"Candidate file:\n{OUTPUT_PATH}"
    )

    print(
        f"Report:\n{REPORT_PATH}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()