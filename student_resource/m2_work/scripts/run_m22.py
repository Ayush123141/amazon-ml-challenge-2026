import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    GROUND_TRUTH,
    BLOCKING_100K_DIR,
    DF_THRESHOLDS,
    S1_SAMPLE_SIZE,
)

from blocking.io import read_tsv
from blocking.indexes import build_all_indexes
from blocking.blocking import get_candidates
from blocking.token_index import (
    build_token_frequency,
    build_rare_token_index,
)
from blocking.normalization import (
    normalize_country,
    normalize_name,
    normalize_address,
)


def parse_ids(value):
    if not value:
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def get_rare_token_candidates(
    row,
    rare_index,
):
    """
    Query rare-token index using all unique
    name + address tokens.
    """

    country = normalize_country(
        row["country"]
    )

    if not country:
        return set()

    name_tokens = set(
        normalize_name(
            row["business_name"]
        ).split()
    )

    address_tokens = set(
        normalize_address(
            row["business_address"]
        ).split()
    )

    tokens = name_tokens | address_tokens

    candidates = set()

    for token in tokens:

        key = (country, token)

        candidates.update(
            rare_index.get(key, [])
        )

    return candidates


def main():

    print("=" * 80)
    print("M2.2 RARE-TOKEN BLOCKING")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load S2 + S3
    # ---------------------------------------------------------

    print("\n[1/6] Loading S2 + S3...")

    s2 = read_tsv(S2_TRAIN)
    s3 = read_tsv(S3_TRAIN)

    rows = list(
        s2.to_dict("records")
    )

    rows.extend(
        s3.to_dict("records")
    )

    print(
        f"S2: {len(s2):,}"
    )

    print(
        f"S3: {len(s3):,}"
    )

    print(
        f"Combined: {len(rows):,}"
    )

    # ---------------------------------------------------------
    # 2. Build M2.1 indexes
    # ---------------------------------------------------------

    print("\n[2/6] Building M2.1 indexes...")

    exact_indexes = build_all_indexes(
        rows
    )

    # ---------------------------------------------------------
    # 3. Build token frequency
    # ---------------------------------------------------------

    print(
        "\n[3/6] Building token frequency..."
    )

    frequency = build_token_frequency(
        rows
    )

    print(
        f"Unique token keys: "
        f"{len(frequency):,}"
    )

    # ---------------------------------------------------------
    # 4. Load S1 + ground truth
    # ---------------------------------------------------------

    print(
        "\n[4/6] Loading S1 + ground truth..."
    )

    s1 = read_tsv(
        S1_TRAIN,
        nrows=S1_SAMPLE_SIZE,
    )

    truth = read_tsv(
        GROUND_TRUTH
    )

    sample_ids = set(
        s1["entity_id"]
    )

    truth = truth[
        truth["source1_entity_id"]
        .isin(sample_ids)
    ]

    truth_map = {
        row.source1_entity_id:
        parse_ids(row.matched_entity_ids)
        for row in truth.itertuples()
    }

    s1_rows = list(
        s1.to_dict("records")
    )

    # ---------------------------------------------------------
    # 5. Benchmark thresholds
    # ---------------------------------------------------------

    print(
        "\n[5/6] Benchmarking thresholds..."
    )

    results = []

    best_union_rows = None
    best_union_recall = -1

    for threshold in DF_THRESHOLDS:

        print("\n" + "-" * 80)
        print(
            f"THRESHOLD = {threshold}"
        )
        print("-" * 80)

        print("Building rare-token index...")

        rare_index = build_rare_token_index(
            rows,
            frequency,
            max_df=threshold,
        )

        print(
            f"Rare index keys: "
            f"{len(rare_index):,}"
        )

        total_candidates = 0
        total_true = 0
        rare_retrieved_true = 0
        union_retrieved_true = 0

        rare_complete = 0
        union_complete = 0

        zero_rare = 0
        zero_union = 0

        candidate_counts = []

        union_rows = []

        for i, row in enumerate(
            s1_rows,
            start=1,
        ):

            s1_id = row["entity_id"]

            true_ids = truth_map.get(
                s1_id,
                set()
            )

            # -----------------------------
            # M2.1
            # -----------------------------

            exact_candidates = get_candidates(
                row,
                exact_indexes,
            )

            # -----------------------------
            # M2.2
            # -----------------------------

            rare_candidates = (
                get_rare_token_candidates(
                    row,
                    rare_index,
                )
            )

            # -----------------------------
            # Union
            # -----------------------------

            union_candidates = (
                exact_candidates
                | rare_candidates
            )

            # -----------------------------
            # Metrics
            # -----------------------------

            rare_hits = (
                true_ids
                & rare_candidates
            )

            union_hits = (
                true_ids
                & union_candidates
            )

            total_true += len(true_ids)

            rare_retrieved_true += (
                len(rare_hits)
            )

            union_retrieved_true += (
                len(union_hits)
            )

            if true_ids:

                if rare_hits == true_ids:
                    rare_complete += 1

                if union_hits == true_ids:
                    union_complete += 1

            if not rare_candidates:
                zero_rare += 1

            if not union_candidates:
                zero_union += 1

            candidate_counts.append(
                len(union_candidates)
            )

            union_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_ids":
                        ",".join(
                            sorted(
                                union_candidates
                            )
                        ),
                }
            )

            if i % 10_000 == 0:

                print(
                    f"Processed "
                    f"{i:,}/{len(s1_rows):,}"
                )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        pair_recall_rare = (
            rare_retrieved_true / total_true
            if total_true
            else 0
        )

        pair_recall_union = (
            union_retrieved_true / total_true
            if total_true
            else 0
        )

        entity_recall_rare = (
            rare_complete / len(truth_map)
            if truth_map
            else 0
        )

        entity_recall_union = (
            union_complete / len(truth_map)
            if truth_map
            else 0
        )

        candidate_series = pd.Series(
            candidate_counts
        )

        mean_candidates = (
            candidate_series.mean()
        )

        p95 = (
            candidate_series.quantile(
                0.95
            )
        )

        p99 = (
            candidate_series.quantile(
                0.99
            )
        )

        print(
            f"\nRare pair recall       : "
            f"{pair_recall_rare:.6f}"
        )

        print(
            f"Union pair recall      : "
            f"{pair_recall_union:.6f}"
        )

        print(
            f"Rare entity recall     : "
            f"{entity_recall_rare:.6f}"
        )

        print(
            f"Union entity recall    : "
            f"{entity_recall_union:.6f}"
        )

        print(
            f"Mean union candidates  : "
            f"{mean_candidates:.2f}"
        )

        print(
            f"P95 union candidates   : "
            f"{p95:.0f}"
        )

        print(
            f"P99 union candidates   : "
            f"{p99:.0f}"
        )

        print(
            f"Zero union candidates  : "
            f"{zero_union:,}"
        )

        results.append(
            {
                "df_threshold": threshold,
                "rare_pair_recall":
                    pair_recall_rare,
                "union_pair_recall":
                    pair_recall_union,
                "rare_entity_complete_recall":
                    entity_recall_rare,
                "union_entity_complete_recall":
                    entity_recall_union,
                "mean_union_candidates":
                    mean_candidates,
                "p95_union_candidates":
                    p95,
                "p99_union_candidates":
                    p99,
                "zero_union_candidates":
                    zero_union,
                "zero_rare_candidates":
                    zero_rare,
            }
        )

        # Keep the highest-recall result.
        if (
            pair_recall_union
            > best_union_recall
        ):
            best_union_recall = (
                pair_recall_union
            )

            best_union_rows = (
                union_rows
            )

    # ---------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------

    print("\n[6/6] Saving results...")

    BLOCKING_100K_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        BLOCKING_100K_DIR
        / "m22_threshold_results.csv"
    )

    pd.DataFrame(results).to_csv(
        results_path,
        index=False,
    )

    best_path = (
        BLOCKING_100K_DIR
        / "candidate_pairs_m22_best.tsv"
    )

    pd.DataFrame(
        best_union_rows
    ).to_csv(
        best_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nThreshold results:"
        f"\n{results_path}"
    )

    print(
        f"\nBest union candidates:"
        f"\n{best_path}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()