import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from blocking.config import S2_TRAIN, S3_TRAIN
from blocking.io import read_tsv
from blocking.token_index import build_token_frequency


def main():

    print("=" * 80)
    print("TOKEN FREQUENCY TEST")
    print("=" * 80)

    print("\nLoading S2...")
    s2 = read_tsv(S2_TRAIN)

    print("Loading S3...")
    s3 = read_tsv(S3_TRAIN)

    rows = list(
        s2.to_dict("records")
    )

    rows.extend(
        s3.to_dict("records")
    )

    print(
        f"Total entities: {len(rows):,}"
    )

    print("\nBuilding token frequency...")

    frequency = build_token_frequency(
        rows
    )

    print(
        f"Unique (country, token) keys: "
        f"{len(frequency):,}"
    )

    # Show rare-token counts
    thresholds = [1, 2, 5, 10, 25, 50]

    print("\nToken frequency distribution:")

    for threshold in thresholds:

        count = sum(
            1
            for value in frequency.values()
            if value <= threshold
        )

        print(
            f"DF <= {threshold:2d}: "
            f"{count:,} tokens"
        )


if __name__ == "__main__":
    main()