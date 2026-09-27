from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from m2_work.src.blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    GROUND_TRUTH,
)
from m2_work.src.blocking.io import read_tsv
from m2_work.src.blocking.token_index import (
    build_token_frequency,
)
from m2_work.src.blocking.approx_blocking import (
    build_approx_index,
    get_approx_candidates,
)


S1_ROWS = 10_000
MAX_DF = 50
MAX_CANDIDATES = 200


def load_truth(path):
    df = read_tsv(path)

    truth = {}

    for row in df.to_dict(orient="records"):
        truth[row["source1_entity_id"]] = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        }

    return truth


def evaluate(predictions, truth, sample_ids):
    true_pairs = 0
    retrieved_pairs = 0
    fully_retrieved = 0
    zero_retrieval = 0

    candidate_counts = []

    macro_f05_sum = 0.0

    for source1_id in sample_ids:

        true_ids = truth.get(source1_id, set())
        predicted_ids = predictions.get(
            source1_id,
            set(),
        )

        candidate_counts.append(len(predicted_ids))

        if not true_ids:
            f05 = 1.0 if not predicted_ids else 0.0

        else:
            true_pairs += len(true_ids)

            overlap = true_ids & predicted_ids

            retrieved_pairs += len(overlap)

            if not overlap:
                zero_retrieval += 1

            if true_ids.issubset(predicted_ids):
                fully_retrieved += 1

            if not predicted_ids or not overlap:
                f05 = 0.0

            else:
                precision = len(overlap) / len(predicted_ids)
                recall = len(overlap) / len(true_ids)

                f05 = (
                    1.25 * precision * recall
                    / (0.25 * precision + recall)
                )

        macro_f05_sum += f05

    n = len(sample_ids)

    return {
        "evaluated": n,
        "oracle_macro_f05": macro_f05_sum / n,
        "pair_recall": (
            retrieved_pairs / true_pairs
            if true_pairs
            else 0.0
        ),
        "mean_candidates": (
            sum(candidate_counts) / len(candidate_counts)
        ),
        "zero_candidates": sum(
            1
            for x in candidate_counts
            if x == 0
        ),
        "zero_true_retrieval": zero_retrieval,
        "fully_retrieved": fully_retrieved,
        "true_pairs": true_pairs,
        "retrieved_true_pairs": retrieved_pairs,
        "max_candidates": max(candidate_counts),
    }


def main():

    print("=" * 80)
    print("M2.7 — 10K SMOKE TEST")
    print("=" * 80)

    print("\nLoading S1...")

    s1 = read_tsv(
        S1_TRAIN,
        nrows=S1_ROWS,
    )

    s1_rows = s1.to_dict(
        orient="records"
    )

    sample_ids = [
        row["entity_id"]
        for row in s1_rows
    ]

    print(f"S1 rows: {len(s1_rows):,}")

    print("\nLoading S2...")

    s2 = read_tsv(S2_TRAIN)

    print(f"S2 rows: {len(s2):,}")

    print("\nLoading S3...")

    s3 = read_tsv(S3_TRAIN)

    print(f"S3 rows: {len(s3):,}")

    target_rows = (
        s2.to_dict(orient="records")
        + s3.to_dict(orient="records")
    )

    print(
        f"Combined target rows: "
        f"{len(target_rows):,}"
    )

    print("\nBuilding token frequency...")

    start = time.time()

    frequency = build_token_frequency(
        target_rows
    )

    print(
        f"Frequency built in "
        f"{time.time() - start:.2f}s"
    )

    print("\nBuilding M2.7 approximate index...")

    start = time.time()

    index = build_approx_index(
        target_rows,
        frequency,
        max_df=MAX_DF,
    )

    elapsed = time.time() - start

    print(
        f"Index built in {elapsed:.2f}s"
    )

    print(
        f"Index keys: {len(index):,}"
    )

    print("\nLoading ground truth...")

    truth = load_truth(GROUND_TRUTH)

    print(
        f"Ground-truth entities: "
        f"{len(truth):,}"
    )

    print("\nGenerating candidates...")

    predictions = {}

    start = time.time()

    for i, row in enumerate(
        s1_rows,
        start=1,
    ):

        predictions[
            row["entity_id"]
        ] = get_approx_candidates(
            row,
            index,
            max_candidates=MAX_CANDIDATES,
        )

        if i % 1_000 == 0:
            print(
                f"Processed "
                f"{i:,}/{len(s1_rows):,}"
            )

    elapsed = time.time() - start

    print(
        f"\nCandidate generation: "
        f"{elapsed:.2f}s"
    )

    print("\nRESULTS")

    metrics = evaluate(
        predictions,
        truth,
        sample_ids,
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("M2.7 10K TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()