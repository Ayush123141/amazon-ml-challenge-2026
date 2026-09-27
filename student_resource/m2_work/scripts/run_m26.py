from pathlib import Path
import csv
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from m2_work.src.blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    GROUND_TRUTH,
    S1_SAMPLE_SIZE,
)
from m2_work.src.blocking.io import read_tsv
from m2_work.src.blocking.similarity_blocking import (
    build_similarity_index,
    get_similarity_candidates,
)


OUTPUT_DIR = PROJECT_ROOT / "m2_work" / "outputs" / "blocking_100k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = S1_SAMPLE_SIZE
MAX_CANDIDATES_VALUES = [50, 100, 250, 500]


def load_ground_truth(path):
    rows = read_tsv(path)

    truth = {}

    for row in rows.to_dict(orient="records"):
        source1_id = row["source1_entity_id"]

        matched = {
            value.strip()
            for value in row["matched_entity_ids"].split(",")
            if value.strip()
        }

        truth[source1_id] = matched

    return truth


def evaluate(predictions, truth, sample_ids):
    total_true_pairs = 0
    retrieved_true_pairs = 0

    total_entities_with_truth = 0
    fully_retrieved = 0

    candidate_counts = []
    zero_candidates = 0
    zero_true_retrieval = 0

    for source1_id in sample_ids:
        true_ids = truth.get(source1_id, set())
        predicted_ids = predictions.get(source1_id, set())

        if true_ids:
            total_entities_with_truth += 1
            total_true_pairs += len(true_ids)

            overlap = true_ids & predicted_ids
            retrieved_true_pairs += len(overlap)

            if true_ids.issubset(predicted_ids):
                fully_retrieved += 1

            if not overlap:
                zero_true_retrieval += 1

        candidate_counts.append(len(predicted_ids))

        if not predicted_ids:
            zero_candidates += 1

    pair_recall = (
        retrieved_true_pairs / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    # Candidate-constrained oracle:
    #
    # For each S1 entity:
    # - if all true matches are in candidates, an ideal matcher could
    #   reproduce the ground truth perfectly => F0.5 = 1
    # - otherwise the best possible prediction is the retrieved true
    #   subset.
    #
    # We compute the actual macro F0.5 of that oracle.

    macro_f05_sum = 0.0

    for source1_id in sample_ids:
        true_ids = truth.get(source1_id, set())
        predicted_ids = predictions.get(source1_id, set())

        if not true_ids and not predicted_ids:
            f05 = 1.0

        elif not true_ids and predicted_ids:
            f05 = 0.0

        else:
            tp = len(true_ids & predicted_ids)
            predicted_count = len(predicted_ids)
            true_count = len(true_ids)

            if predicted_count == 0 or tp == 0:
                f05 = 0.0
            else:
                precision = tp / predicted_count
                recall = tp / true_count

                if precision == 0.0 or recall == 0.0:
                    f05 = 0.0
                else:
                    f05 = (
                        1.25 * precision * recall
                        / (0.25 * precision + recall)
                    )

        macro_f05_sum += f05

    macro_f05 = macro_f05_sum / len(sample_ids)

    sorted_counts = sorted(candidate_counts)

    def percentile(values, p):
        if not values:
            return 0

        index = int((len(values) - 1) * p)
        return values[index]

    return {
        "evaluated": len(sample_ids),
        "pair_recall": pair_recall,
        "oracle_macro_f05": macro_f05,
        "mean_candidates": (
            sum(candidate_counts) / len(candidate_counts)
            if candidate_counts
            else 0.0
        ),
        "median_candidates": (
            sorted_counts[len(sorted_counts) // 2]
            if sorted_counts
            else 0
        ),
        "p95_candidates": percentile(sorted_counts, 0.95),
        "p99_candidates": percentile(sorted_counts, 0.99),
        "max_candidates": max(candidate_counts)
        if candidate_counts
        else 0,
        "zero_candidates": zero_candidates,
        "zero_true_retrieval": zero_true_retrieval,
        "fully_retrieved": fully_retrieved,
        "non_singletons": total_entities_with_truth,
        "true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
    }


def main():
    print("=" * 80)
    print("M2.6 SIMILARITY BLOCKING BENCHMARK")
    print("=" * 80)

    print("\nLoading Source 1 sample...")

    s1 = read_tsv(S1_TRAIN, nrows=SAMPLE_SIZE)

    s1_rows = s1.to_dict(orient="records")
    sample_ids = [row["entity_id"] for row in s1_rows]

    print(f"S1 rows: {len(s1_rows):,}")

    print("\nLoading Source 2...")

    s2 = read_tsv(S2_TRAIN)

    print(f"S2 rows: {len(s2):,}")

    print("\nLoading Source 3...")

    s3 = read_tsv(S3_TRAIN)

    print(f"S3 rows: {len(s3):,}")

    target_rows = (
        s2.to_dict(orient="records")
        + s3.to_dict(orient="records")
    )

    print(f"Combined target rows: {len(target_rows):,}")

    print("\nBuilding similarity index...")

    start = time.time()

    indexes = build_similarity_index(target_rows)

    elapsed = time.time() - start

    print(f"Index built in {elapsed:.2f} seconds")

    print(
        f"Name index keys: {len(indexes['name']):,}"
    )
    print(
        f"Address index keys: {len(indexes['address']):,}"
    )

    print("\nLoading ground truth...")

    truth = load_ground_truth(GROUND_TRUTH)

    print(f"Ground-truth entities: {len(truth):,}")

    results = []

    for max_candidates in MAX_CANDIDATES_VALUES:

        print("\n" + "-" * 80)
        print(f"MAX CANDIDATES = {max_candidates}")
        print("-" * 80)

        predictions = {}

        start = time.time()

        for i, row in enumerate(s1_rows, start=1):

            candidates = get_similarity_candidates(
                row,
                indexes,
                max_candidates=max_candidates,
            )

            predictions[row["entity_id"]] = candidates

            if i % 10_000 == 0:
                print(f"Processed {i:,}/{len(s1_rows):,}")

        elapsed = time.time() - start

        metrics = evaluate(
            predictions,
            truth,
            sample_ids,
        )

        metrics["max_candidates_setting"] = max_candidates
        metrics["runtime_seconds"] = elapsed

        results.append(metrics)

        print("\nRESULTS")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        output_path = (
            OUTPUT_DIR
            / f"candidate_pairs_m26_max{max_candidates}.tsv"
        )

        print(f"\nWriting: {output_path}")

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f, delimiter="\t")

            writer.writerow(
                [
                    "source1_entity_id",
                    "candidate_entity_ids",
                ]
            )

            for source1_id in sample_ids:
                candidates = sorted(
                    predictions.get(source1_id, set())
                )

                writer.writerow(
                    [
                        source1_id,
                        ",".join(candidates),
                    ]
                )

    summary_path = OUTPUT_DIR / "m26_similarity_results.csv"

    print("\nWriting summary:")
    print(summary_path)

    fieldnames = list(results[0].keys())

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("M2.6 COMPLETE")
    print("=" * 80)

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()