import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import (
    BLOCKING_100K_DIR,
    GROUND_TRUTH,
)


def parse_ids(value):
    if not value:
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


def main():

    candidate_path = (
        BLOCKING_100K_DIR
        / "candidate_pairs_m21.tsv"
    )

    print("=" * 80)
    print("M2.1 RECALL EVALUATION")
    print("=" * 80)

    print("\nLoading candidate file...")
    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Candidate S1 rows: {len(candidates):,}")

    print("\nLoading ground truth...")
    truth = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    # Restrict GT to the 100k S1 sample
    sample_s1 = set(candidates["source1_entity_id"])

    truth = truth[
        truth["source1_entity_id"].isin(sample_s1)
    ].copy()

    print(f"Ground-truth S1 rows: {len(truth):,}")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    total_true_pairs = 0
    retrieved_true_pairs = 0

    total_s1_with_truth = 0
    fully_retrieved_s1 = 0

    zero_candidate_s1 = 0

    candidate_counts = []

    # Fast lookup
    candidate_map = dict(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_ids"],
        )
    )

    failure_rows = []

    for row in truth.itertuples(index=False):

        s1_id = row.source1_entity_id

        true_ids = parse_ids(
            row.matched_entity_ids
        )

        candidate_ids = parse_ids(
            candidate_map.get(s1_id, "")
        )

        total_true_pairs += len(true_ids)

        retrieved = true_ids & candidate_ids

        retrieved_true_pairs += len(retrieved)

        candidate_counts.append(
            len(candidate_ids)
        )

        if true_ids:
            total_s1_with_truth += 1

            if retrieved == true_ids:
                fully_retrieved_s1 += 1
            else:
                failure_rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "true_match_count": len(true_ids),
                        "retrieved_match_count": len(retrieved),
                        "missing_match_count": len(
                            true_ids - candidate_ids
                        ),
                        "missing_entity_ids": ",".join(
                            sorted(true_ids - candidate_ids)
                        ),
                    }
                )

        if not candidate_ids:
            zero_candidate_s1 += 1

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    pair_recall = (
        retrieved_true_pairs / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    entity_complete_recall = (
        fully_retrieved_s1 / total_s1_with_truth
        if total_s1_with_truth
        else 0.0
    )

    candidate_counts_series = pd.Series(
        candidate_counts,
        dtype="int64",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Total true pairs          : "
        f"{total_true_pairs:,}"
    )

    print(
        f"Retrieved true pairs      : "
        f"{retrieved_true_pairs:,}"
    )

    print(
        f"PAIR RECALL               : "
        f"{pair_recall:.6f}"
    )

    print(
        f"\nS1 with ground truth      : "
        f"{total_s1_with_truth:,}"
    )

    print(
        f"Fully retrieved S1        : "
        f"{fully_retrieved_s1:,}"
    )

    print(
        f"ENTITY-COMPLETE RECALL    : "
        f"{entity_complete_recall:.6f}"
    )

    print(
        f"\nZero-candidate S1         : "
        f"{zero_candidate_s1:,}"
    )

    if len(candidate_counts_series) > 0:

        print(
            f"\nCandidate count statistics:"
        )

        print(
            f"Mean                      : "
            f"{candidate_counts_series.mean():.2f}"
        )

        print(
            f"Median                    : "
            f"{candidate_counts_series.median():.0f}"
        )

        print(
            f"P95                       : "
            f"{candidate_counts_series.quantile(0.95):.0f}"
        )

        print(
            f"P99                       : "
            f"{candidate_counts_series.quantile(0.99):.0f}"
        )

        print(
            f"Maximum                   : "
            f"{candidate_counts_series.max():,}"
        )

    # ---------------------------------------------------------
    # Save failures
    # ---------------------------------------------------------

    failure_path = (
        BLOCKING_100K_DIR
        / "m21_failures.tsv"
    )

    pd.DataFrame(failure_rows).to_csv(
        failure_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nFailure analysis saved to:"
        f"\n{failure_path}"
    )


if __name__ == "__main__":
    main()