from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "m2_work" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from blocking.io import read_tsv
from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    BLOCKING_100K_DIR,
)
from blocking.indexes import build_all_indexes
from blocking.token_index import (
    build_token_frequency,
    build_rare_token_index,
)
from blocking.blocking import get_candidates


SAMPLE_SIZE = 100_000

DF_THRESHOLDS = [
    2,
    5,
    10,
    25,
    50,
]


def load_s1():

    return read_tsv(
        S1_TRAIN,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=SAMPLE_SIZE,
    )


def load_targets():

    print("Loading Source 2...")

    s2 = read_tsv(
        S2_TRAIN,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(
        f"S2 rows: {len(s2):,}"
    )

    print("Loading Source 3...")

    s3 = read_tsv(
        S3_TRAIN,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(
        f"S3 rows: {len(s3):,}"
    )

    combined = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    return combined.to_dict(
        orient="records"
    )


def load_ground_truth():

    path = (
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    gt = read_tsv(path)

    return {
        row["source1_entity_id"]: (
            {
                x.strip()
                for x in row["matched_entity_ids"].split(",")
                if x.strip()
            }
            if row["matched_entity_ids"]
            else set()
        )
        for _, row in gt.iterrows()
    }


def f05(precision, recall):

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def evaluate(
    s1_df,
    gt_map,
    indexes,
    rare_index,
):

    scores = []

    total_true = 0
    total_retrieved = 0

    candidate_counts = []

    zero_candidates = 0
    zero_true_retrieval = 0
    fully_retrieved = 0

    for _, row in s1_df.iterrows():

        s1_id = row["entity_id"]

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        # --------------------------------------------------
        # M2.1 exact candidates
        # --------------------------------------------------

        exact_candidates = get_candidates(
            row,
            indexes,
        )

        # --------------------------------------------------
        # M2.2 rare-token candidates
        # --------------------------------------------------

        rare_candidates = set()

        country = row["country"]

        from blocking.normalization import (
            normalize_country,
        )

        normalized_country = normalize_country(
            country
        )

        from blocking.token_index import (
            tokenize_row,
        )

        tokens = tokenize_row(row)

        for token in tokens:

            key = (
                normalized_country,
                token,
            )

            rare_candidates.update(
                rare_index.get(
                    key,
                    [],
                )
            )

        # --------------------------------------------------
        # UNION
        # --------------------------------------------------

        candidates = (
            exact_candidates
            | rare_candidates
        )

        candidate_counts.append(
            len(candidates)
        )

        if not candidates:
            zero_candidates += 1

        retrieved = (
            true_ids & candidates
        )

        tp = len(retrieved)

        if true_ids:

            total_true += len(true_ids)

            total_retrieved += tp

            if tp == 0:
                zero_true_retrieval += 1

            if tp == len(true_ids):
                fully_retrieved += 1

            precision = (
                1.0
                if tp > 0
                else 0.0
            )

            recall = (
                tp / len(true_ids)
            )

            score = f05(
                precision,
                recall,
            )

        else:

            score = 1.0

        scores.append(score)

    candidate_series = pd.Series(
        candidate_counts,
        dtype=float,
    )

    return {
        "macro_f05": sum(scores) / len(scores),

        "pair_recall": (
            total_retrieved / total_true
            if total_true
            else 0.0
        ),

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

        "zero_candidates":
            zero_candidates,

        "zero_true_retrieval":
            zero_true_retrieval,

        "fully_retrieved":
            fully_retrieved,

        "total_true":
            total_true,

        "total_retrieved":
            total_retrieved,
    }


def main():

    print("=" * 70)
    print("M2.4 UNION BLOCKING")
    print("=" * 70)

    s1_df = load_s1()

    gt_map = load_ground_truth()

    target_rows = load_targets()

    print()
    print("Building exact indexes...")

    indexes = build_all_indexes(
        target_rows
    )

    print(
        f"Name keys: "
        f"{len(indexes['name']):,}"
    )

    print(
        f"Address keys: "
        f"{len(indexes['address']):,}"
    )

    print(
        f"Name+address keys: "
        f"{len(indexes['name_address']):,}"
    )

    print()
    print("Building token frequency...")

    frequency = build_token_frequency(
        target_rows
    )

    results = []

    for df in DF_THRESHOLDS:

        print()
        print("=" * 60)

        print(
            f"Building rare-token index "
            f"DF <= {df}"
        )

        print("=" * 60)

        rare_index = build_rare_token_index(
            target_rows,
            frequency,
            max_df=df,
        )

        result = evaluate(
            s1_df,
            gt_map,
            indexes,
            rare_index,
        )

        result["df_threshold"] = df

        results.append(result)

        print(
            f"Oracle Macro F0.5 : "
            f"{result['macro_f05']:.6f}"
        )

        print(
            f"Pair recall       : "
            f"{result['pair_recall']:.6f}"
        )

        print(
            f"Mean candidates   : "
            f"{result['mean_candidates']:.2f}"
        )

        print(
            f"P95 candidates    : "
            f"{result['p95_candidates']:.0f}"
        )

        print(
            f"P99 candidates    : "
            f"{result['p99_candidates']:.0f}"
        )

        print(
            f"Zero candidates   : "
            f"{result['zero_candidates']:,}"
        )

        print(
            f"Fully retrieved   : "
            f"{result['fully_retrieved']:,}"
        )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        BLOCKING_100K_DIR
        / "m24_union_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL M2.4 RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "df_threshold",
                "macro_f05",
                "pair_recall",
                "mean_candidates",
                "p95_candidates",
                "p99_candidates",
                "zero_candidates",
                "fully_retrieved",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Saved results to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()