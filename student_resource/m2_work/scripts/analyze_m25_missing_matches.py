from pathlib import Path
import sys
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "m2_work" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from blocking.io import read_tsv
from blocking.normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)


SAMPLE_SIZE = 100_000


def load_s1():
    return read_tsv(
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_source1.tsv",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=SAMPLE_SIZE,
    )


def load_targets():

    s2 = read_tsv(
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_source2.tsv",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s3 = read_tsv(
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_source3.tsv",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    return pd.concat(
        [s2, s3],
        ignore_index=True,
    )


def load_ground_truth():

    gt = read_tsv(
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    result = {}

    for _, row in gt.iterrows():

        value = row[
            "matched_entity_ids"
        ]

        result[
            row["source1_entity_id"]
        ] = (
            {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }
            if value
            else set()
        )

    return result


def token_set(value):

    normalized = normalize_name(value)

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def char_similarity(a, b):

    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0.0

    # Simple normalized character overlap.
    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))

    if longer == 0:
        return 0.0

    common = 0

    b_counter = Counter(b)

    for char in a:
        if b_counter[char] > 0:
            common += 1
            b_counter[char] -= 1

    return common / longer


def main():

    print("=" * 70)
    print("M2.5 MISSING TRUE-MATCH ANALYSIS")
    print("=" * 70)

    print(
        f"Validation S1 sample: "
        f"{SAMPLE_SIZE:,}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading Source 1...")
    s1 = load_s1()

    print("Loading target sources...")
    targets = load_targets()

    print(
        f"Target rows: "
        f"{len(targets):,}"
    )

    print("Loading ground truth...")
    gt_map = load_ground_truth()

    # --------------------------------------------------------
    # Build target lookup
    # --------------------------------------------------------

    target_lookup = {
        row["entity_id"]: row
        for _, row in targets.iterrows()
    }

    # --------------------------------------------------------
    # Load M2.2 candidates
    # --------------------------------------------------------

    candidate_path = (
        PROJECT_ROOT
        / "m2_work"
        / "outputs"
        / "blocking_100k"
        / "candidate_pairs_m22_best.tsv"
    )

    candidates_df = read_tsv(
        candidate_path
    )

    candidate_map = {}

    for _, row in candidates_df.iterrows():

        value = row[
            "candidate_entity_ids"
        ]

        candidate_map[
            row["source1_entity_id"]
        ] = (
            {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }
            if value
            else set()
        )

    # --------------------------------------------------------
    # Analyze missing pairs
    # --------------------------------------------------------

    records = []

    missing_pairs = 0

    for _, s1_row in s1.iterrows():

        s1_id = s1_row[
            "entity_id"
        ]

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        candidates = candidate_map.get(
            s1_id,
            set(),
        )

        missing_ids = (
            true_ids - candidates
        )

        for target_id in missing_ids:

            target = target_lookup.get(
                target_id
            )

            if target is None:
                continue

            s1_name = s1_row[
                "business_name"
            ]

            target_name = target[
                "business_name"
            ]

            s1_address = s1_row[
                "business_address"
            ]

            target_address = target[
                "business_address"
            ]

            s1_country = normalize_country(
                s1_row["country"]
            )

            target_country = normalize_country(
                target["country"]
            )

            name_tokens_1 = token_set(
                s1_name
            )

            name_tokens_2 = token_set(
                target_name
            )

            address_tokens_1 = token_set(
                s1_address
            )

            address_tokens_2 = token_set(
                target_address
            )

            records.append(
                {
                    "source1_entity_id": s1_id,
                    "target_entity_id": target_id,

                    "country_same":
                        s1_country
                        == target_country,

                    "name_exact":
                        normalize_name(
                            s1_name
                        )
                        == normalize_name(
                            target_name
                        ),

                    "address_exact":
                        normalize_address(
                            s1_address
                        )
                        == normalize_address(
                            target_address
                        ),

                    "name_jaccard":
                        jaccard(
                            name_tokens_1,
                            name_tokens_2,
                        ),

                    "address_jaccard":
                        jaccard(
                            address_tokens_1,
                            address_tokens_2,
                        ),

                    "name_char_similarity":
                        char_similarity(
                            s1_name,
                            target_name,
                        ),

                    "name_token_overlap":
                        len(
                            name_tokens_1
                            & name_tokens_2
                        ),

                    "address_token_overlap":
                        len(
                            address_tokens_1
                            & address_tokens_2
                        ),

                    "s1_name":
                        s1_name,

                    "target_name":
                        target_name,

                    "s1_address":
                        s1_address,

                    "target_address":
                        target_address,

                    "s1_country":
                        s1_row["country"],

                    "target_country":
                        target["country"],
                }
            )

            missing_pairs += 1

    result = pd.DataFrame(
        records
    )

    print()
    print("=" * 70)

    print(
        f"Missing true pairs: "
        f"{missing_pairs:,}"
    )

    print("=" * 70)

    if result.empty:

        print(
            "No missing matches found."
        )

        return

    # --------------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------------

    print()

    print(
        "Country agreement:"
    )

    print(
        result[
            "country_same"
        ].value_counts(
            normalize=True
        ).to_string()
    )

    print()

    print(
        "Exact normalized name:"
    )

    print(
        result[
            "name_exact"
        ].value_counts(
            normalize=True
        ).to_string()
    )

    print()

    print(
        "Exact normalized address:"
    )

    print(
        result[
            "address_exact"
        ].value_counts(
            normalize=True
        ).to_string()
    )

    print()

    print(
        "Name Jaccard distribution:"
    )

    print(
        result[
            "name_jaccard"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        ).to_string()
    )

    print()

    print(
        "Address Jaccard distribution:"
    )

    print(
        result[
            "address_jaccard"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        ).to_string()
    )

    print()

    print(
        "Name character similarity:"
    )

    print(
        result[
            "name_char_similarity"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        ).to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "m2_work"
        / "reports"
        / "m25_missing_pairs.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()

    print(
        f"Saved detailed analysis to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()