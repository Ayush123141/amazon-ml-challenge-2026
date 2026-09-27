from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "m2_work" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from blocking.io import read_tsv

from blocking.config import (
    S1_TRAIN,
    S2_TRAIN,
    S3_TRAIN,
    BLOCKING_100K_DIR,
)

from blocking.char_ngram_index import (
    build_char_ngram_frequency,
    build_char_ngram_index,
)

from blocking.char_ngram_blocking import (
    get_char_ngram_candidates,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SAMPLE_SIZE = 100_000

N_VALUES = [3]

MAX_DF_VALUES = [
    50,
    100,
    250,
    500,
]

MIN_SHARED_VALUES = [
    1,
    2,
]


# ============================================================
# LOAD VALIDATION SOURCE 1
# ============================================================

def load_validation_s1():
    """
    Load the first 100,000 Source 1 entities.

    This is the same validation population used by
    the previous M2.1 and M2.2 experiments.
    """

    df = read_tsv(
        S1_TRAIN,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=SAMPLE_SIZE,
    )

    return df


# ============================================================
# LOAD SOURCE 2 + SOURCE 3
# ============================================================

def load_target_rows():
    """
    Load Source 2 and Source 3 and convert them into
    a list of dictionaries.

    The character n-gram index expects rows like:

        {
            "entity_id": "...",
            "business_name": "...",
            "business_address": "...",
            "country": "..."
        }

    IMPORTANT:
    Iterating directly over a pandas DataFrame returns
    column names, not rows. Therefore we explicitly convert
    the DataFrame using orient="records".
    """

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

    print(
        f"Combined target rows: "
        f"{len(combined):,}"
    )

    rows = combined.to_dict(
        orient="records"
    )

    return rows


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():
    """
    Load training ground truth into:

        source1_entity_id -> set(matched_entity_ids)
    """

    path = (
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    gt = read_tsv(path)

    gt_map = {}

    for _, row in gt.iterrows():

        value = row[
            "matched_entity_ids"
        ]

        if not value:

            gt_map[
                row["source1_entity_id"]
            ] = set()

        else:

            gt_map[
                row["source1_entity_id"]
            ] = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

    return gt_map


# ============================================================
# EVALUATE ONE SETTING
# ============================================================

def evaluate_setting(
    s1_df,
    gt_map,
    index,
    n,
    min_shared,
):
    """
    Evaluate one character n-gram blocking configuration.

    The downstream classifier is treated as perfect.

    Therefore:

        predicted = true_ids ∩ candidates

    This gives the candidate-constrained oracle F0.5.
    """

    total_true_pairs = 0
    retrieved_true_pairs = 0

    zero_candidates = 0
    zero_true_retrieval = 0
    fully_retrieved = 0

    candidate_counts = []

    scores = []

    for _, row in s1_df.iterrows():

        s1_id = row[
            "entity_id"
        ]

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        candidates = get_char_ngram_candidates(
            row,
            index,
            n=n,
            min_shared_ngrams=min_shared,
        )

        candidate_count = len(
            candidates
        )

        candidate_counts.append(
            candidate_count
        )

        if candidate_count == 0:
            zero_candidates += 1

        # True matches available to blocker.
        retrieved = (
            true_ids & candidates
        )

        tp = len(retrieved)

        # ----------------------------------------------------
        # Singleton
        # ----------------------------------------------------

        if not true_ids:

            # Perfect oracle correctly predicts empty.
            score = 1.0

        # ----------------------------------------------------
        # Non-singleton
        # ----------------------------------------------------

        else:

            total_true_pairs += len(
                true_ids
            )

            retrieved_true_pairs += tp

            if tp == 0:
                zero_true_retrieval += 1

            if tp == len(true_ids):
                fully_retrieved += 1

            # Perfect oracle has no false positives.
            precision = (
                1.0
                if tp > 0
                else 0.0
            )

            recall = (
                tp / len(true_ids)
            )

            if precision == 0 and recall == 0:

                score = 0.0

            else:

                score = (
                    1.25
                    * precision
                    * recall
                    / (
                        0.25
                        * precision
                        + recall
                    )
                )

        scores.append(score)

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    macro_f05 = (
        sum(scores)
        / len(scores)
    )

    pair_recall = (
        retrieved_true_pairs
        / total_true_pairs
        if total_true_pairs > 0
        else 0.0
    )

    candidate_series = pd.Series(
        candidate_counts,
        dtype=float,
    )

    return {
        "n": n,
        "max_df": None,
        "min_shared": min_shared,

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
            int(
                candidate_series.max()
            ),

        "zero_candidates":
            zero_candidates,

        "zero_true_retrieval":
            zero_true_retrieval,

        "fully_retrieved":
            fully_retrieved,

        "total_true_pairs":
            total_true_pairs,

        "retrieved_true_pairs":
            retrieved_true_pairs,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "M2.3 CHARACTER N-GRAM BLOCKING BENCHMARK"
    )

    print("=" * 70)

    print(
        f"Validation S1 sample: "
        f"{SAMPLE_SIZE:,}"
    )

    print()

    # --------------------------------------------------------
    # Load validation S1
    # --------------------------------------------------------

    print(
        "Loading validation Source 1..."
    )

    s1_df = load_validation_s1()

    print(
        f"S1 validation rows: "
        f"{len(s1_df):,}"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print(
        "Loading ground truth..."
    )

    gt_map = load_ground_truth()

    print(
        f"Ground-truth S1 entities: "
        f"{len(gt_map):,}"
    )

    # --------------------------------------------------------
    # Load target sources
    # --------------------------------------------------------

    print()

    target_rows = load_target_rows()

    # --------------------------------------------------------
    # Results container
    # --------------------------------------------------------

    all_results = []

    # --------------------------------------------------------
    # N-gram experiments
    # --------------------------------------------------------

    for n in N_VALUES:

        print()
        print(
            "=" * 60
        )

        print(
            f"Building {n}-gram frequency"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Build frequency
        # ----------------------------------------------------

        frequency = (
            build_char_ngram_frequency(
                target_rows,
                n=n,
            )
        )

        print(
            f"Unique "
            f"(country, ngram) keys: "
            f"{len(frequency):,}"
        )

        # ----------------------------------------------------
        # Test DF thresholds
        # ----------------------------------------------------

        for max_df in MAX_DF_VALUES:

            print()

            print(
                "-" * 60
            )

            print(
                f"Building index:"
                f" n={n},"
                f" max_df={max_df}"
            )

            print(
                "-" * 60
            )

            index = (
                build_char_ngram_index(
                    target_rows,
                    frequency,
                    max_df=max_df,
                    n=n,
                )
            )

            print(
                f"Index keys: "
                f"{len(index):,}"
            )

            # ------------------------------------------------
            # Test minimum shared grams
            # ------------------------------------------------

            for min_shared in MIN_SHARED_VALUES:

                print()

                print(
                    f"Testing:"
                    f" n={n},"
                    f" max_df={max_df},"
                    f" min_shared={min_shared}"
                )

                result = evaluate_setting(
                    s1_df,
                    gt_map,
                    index,
                    n,
                    min_shared,
                )

                result[
                    "max_df"
                ] = max_df

                all_results.append(
                    result
                )

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
                    f"Median candidates : "
                    f"{result['median_candidates']:.0f}"
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
                    f"Max candidates    : "
                    f"{result['max_candidates']:,}"
                )

                print(
                    f"Zero candidates   : "
                    f"{result['zero_candidates']:,}"
                )

                print(
                    f"Zero true retrieval: "
                    f"{result['zero_true_retrieval']:,}"
                )

                print(
                    f"Fully retrieved   : "
                    f"{result['fully_retrieved']:,}"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = BLOCKING_100K_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "m23_char_ngram_results.csv"
    )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL M2.3 RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "n",
        "max_df",
        "min_shared",
        "macro_f05",
        "pair_recall",
        "mean_candidates",
        "median_candidates",
        "p95_candidates",
        "p99_candidates",
        "zero_candidates",
        "fully_retrieved",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved results to:"
    )

    print(
        output_path
    )

    print()

    print(
        "M2.2 baseline Oracle F0.5: "
        "0.675934"
    )

    print(
        "Compare every M2.3 configuration "
        "against this baseline."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()