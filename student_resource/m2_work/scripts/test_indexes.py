import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.config import S2_TRAIN, S3_TRAIN
from blocking.io import read_tsv
from blocking.indexes import build_all_indexes


def main():

    print("=" * 70)
    print("LOADING S2 + S3")
    print("=" * 70)

    s2 = read_tsv(S2_TRAIN)
    s3 = read_tsv(S3_TRAIN)

    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    rows = list(s2.to_dict("records"))
    rows.extend(s3.to_dict("records"))

    print(f"Combined rows: {len(rows):,}")

    print("\nBuilding indexes...")

    indexes = build_all_indexes(rows)

    for name, index in indexes.items():
        print(f"{name:15s}: {len(index):,} keys")

    print("\nSample keys:")

    for index_name, index in indexes.items():
        print(f"\n--- {index_name} ---")

        for i, (key, entity_ids) in enumerate(index.items()):
            print(f"{key} -> {entity_ids}")

            if i >= 2:
                break


if __name__ == "__main__":
    main()