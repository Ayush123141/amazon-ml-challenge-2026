from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def parse_ids(value):
    if value is None:
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def load_ground_truth():
    df = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {sorted(missing)}"
        )

    return {
        row["source1_entity_id"]:
            parse_ids(row["matched_entity_ids"])
        for _, row in df.iterrows()
    }


def load_candidates(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "candidate_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate file missing columns: {sorted(missing)}"
        )

    return {
        row["source1_entity_id"]:
            parse_ids(row["candidate_entity_ids"])
        for _, row in df.iterrows()
    }


def evaluate(gt_map, candidate_map):
    """
    Candidate-constrained oracle.

    For every S1 entity present in the candidate file:

        predicted = true_ids ∩ candidates

    This assumes a perfect downstream classifier:
    - every true candidate is selected
    - every false candidate is rejected

    The evaluation population is ONLY the S1 entities
    present in the candidate file.
    """

    scores = []

    total_true = 0
    total_retrieved = 0

    zero_candidate = 0
    zero_true_retrieved = 0

    full_recall_entities = 0

    non_singletons = 0
    singletons = 0

    candidate_counts = []

    evaluated_entities = 0

    for s1_id, candidates in candidate_map.items():

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        evaluated_entities += 1

        candidate_counts.append(
            len(candidates)
        )

        # Perfect downstream classifier:
        # keep only candidates that are actually
        # ground-truth matches.
        predicted = true_ids & candidates

        tp = len(predicted)
        fp = 0
        fn = len(true_ids - predicted)

        # --------------------------------------------------
        # Singleton / empty-ground-truth case
        # --------------------------------------------------
        if len(true_ids) == 0:

            singletons += 1

            # Perfect classifier predicts no match.
            score = 1.0

        # --------------------------------------------------
        # Non-singleton case
        # --------------------------------------------------
        else:

            non_singletons += 1

            total_true += len(true_ids)
            total_retrieved += tp

            if tp == 0:
                zero_true_retrieved += 1

            if tp == len(true_ids):
                full_recall_entities += 1

            # Oracle has zero false positives.
            precision = 1.0 if tp > 0 else 0.0

            recall = tp / len(true_ids)

            score = f05(
                precision,
                recall,
            )

        if len(candidates) == 0:
            zero_candidate += 1

        scores.append(score)

    if not scores:
        raise ValueError(
            "Candidate file contains no S1 entities."
        )

    results = pd.Series(
        scores,
        dtype=float,
    )

    macro_f05 = results.mean()

    pair_recall = (
        total_retrieved / total_true
        if total_true > 0
        else 0.0
    )

    candidate_series = pd.Series(
        candidate_counts,
        dtype=float,
    )

    return {
        "evaluated_entities": evaluated_entities,

        "macro_f05": macro_f05,

        "pair_recall": pair_recall,

        "mean_candidates":
            candidate_series.mean(),

        "median_candidates":
            candidate_series.median(),

        "p95_candidates":
            candidate_series.quantile(0.95),

        "p99_candidates":
            candidate_series.quantile(0.99),

        "max_candidates":
            int(candidate_series.max()),

        "zero_candidate":
            zero_candidate,

        "zero_true_retrieved":
            zero_true_retrieved,

        "full_recall_entities":
            full_recall_entities,

        "non_singletons":
            non_singletons,

        "singletons":
            singletons,

        "total_true":
            total_true,

        "total_retrieved":
            total_retrieved,
    }


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python evaluate_blocking_ceiling.py "
            "<candidate_file>"
        )
        sys.exit(1)

    candidate_path = Path(
        sys.argv[1]
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file does not exist: "
            f"{candidate_path}"
        )

    print(
        "Loading ground truth..."
    )

    gt_map = load_ground_truth()

    print(
        f"Ground-truth S1 entities: "
        f"{len(gt_map):,}"
    )

    print(
        "Loading candidates..."
    )

    candidate_map = load_candidates(
        candidate_path
    )

    print(
        f"Candidate-file S1 entities: "
        f"{len(candidate_map):,}"
    )

    print(
        "Evaluating candidate ceiling..."
    )

    result = evaluate(
        gt_map,
        candidate_map,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CANDIDATE-CONSTRAINED ORACLE"
    )

    print(
        "=" * 60
    )

    print(
        f"Evaluated S1 entities    : "
        f"{result['evaluated_entities']:,}"
    )

    print(
        f"Oracle Macro F0.5       : "
        f"{result['macro_f05']:.6f}"
    )

    print(
        f"Pair recall              : "
        f"{result['pair_recall']:.6f}"
    )

    print(
        f"Mean candidates/S1       : "
        f"{result['mean_candidates']:.2f}"
    )

    print(
        f"Median candidates/S1     : "
        f"{result['median_candidates']:.0f}"
    )

    print(
        f"P95 candidates/S1        : "
        f"{result['p95_candidates']:.0f}"
    )

    print(
        f"P99 candidates/S1        : "
        f"{result['p99_candidates']:.0f}"
    )

    print(
        f"Maximum candidates       : "
        f"{result['max_candidates']:,}"
    )

    print(
        f"Zero-candidate S1        : "
        f"{result['zero_candidate']:,}"
    )

    print(
        f"Zero-true-retrieval S1   : "
        f"{result['zero_true_retrieved']:,}"
    )

    print(
        f"Fully-retrieved S1       : "
        f"{result['full_recall_entities']:,}"
    )

    print(
        f"Non-singletons           : "
        f"{result['non_singletons']:,}"
    )

    print(
        f"Singletons               : "
        f"{result['singletons']:,}"
    )

    print(
        f"True pairs               : "
        f"{result['total_true']:,}"
    )

    print(
        f"Retrieved true pairs     : "
        f"{result['total_retrieved']:,}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()