import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

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


def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0

    beta2 = 0.25

    return (
        (1 + beta2) * precision * recall
        / (beta2 * precision + recall)
    )


def main():

    candidate_path = (
        BLOCKING_100K_DIR
        / "candidate_pairs_m22_best.tsv"
    )

    print("=" * 80)
    print("M2.2 CANDIDATE ORACLE EVALUATION")
    print("=" * 80)

    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    candidate_map = {
        row.source1_entity_id:
            parse_ids(row.candidate_entity_ids)
        for row in candidates.itertuples()
    }

    truth_map = {
        row.source1_entity_id:
            parse_ids(row.matched_entity_ids)
        for row in truth.itertuples()
        if row.source1_entity_id
        in candidate_map
    }

    # ---------------------------------------------------------
    # Candidate-level oracle
    #
    # Predict every candidate that is actually a GT match.
    # ---------------------------------------------------------

    tp = 0
    fp = 0
    fn = 0

    for s1_id, true_ids in truth_map.items():

        candidate_ids = candidate_map[s1_id]

        tp += len(
            true_ids & candidate_ids
        )

        fp += len(
            candidate_ids - true_ids
        )

        fn += len(
            true_ids - candidate_ids
        )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    score = f05(
        precision,
        recall,
    )

    print(
        f"\nTP: {tp:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"\nOracle precision : {precision:.6f}"
    )

    print(
        f"Oracle recall    : {recall:.6f}"
    )

    print(
        f"Oracle F0.5      : {score:.6f}"
    )


if __name__ == "__main__":
    main()