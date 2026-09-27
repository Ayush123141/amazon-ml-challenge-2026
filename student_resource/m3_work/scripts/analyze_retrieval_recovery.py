"""Measure which exact canonical signatures can recover M2-missed true links.

This is a diagnosis only: labels are used to select/evaluate misses, never as retrieval input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "m2_work" / "src"))
from blocking.normalization import normalize_address, normalize_country, normalize_name

LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "company", "co",
    "pvt", "private", "plc", "llp", "lp", "gmbh", "ag", "bv", "sa", "sarl",
}


def parse_ids(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def sorted_tokens(value: str) -> str:
    return " ".join(sorted(value.split()))


def name_without_legal_suffix(value: str) -> str:
    return " ".join(token for token in value.split() if token not in LEGAL_SUFFIXES)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["entity_id"] = result["entity_id"].astype(str)
    result["country_norm"] = result["country"].map(normalize_country)
    result["name_norm"] = result["business_name"].map(normalize_name)
    result["address_norm"] = result["business_address"].map(normalize_address)
    result["name_sorted"] = result["name_norm"].map(sorted_tokens)
    result["name_no_legal"] = result["name_norm"].map(name_without_legal_suffix)
    result["address_sorted"] = result["address_norm"].map(sorted_tokens)
    return result


def main() -> None:
    dataset = PROJECT_ROOT / "dataset" / "train"
    candidate_file = PROJECT_ROOT / "m2_work" / "outputs" / "candidate_pairs.tsv"
    candidates = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)
    source_ids = set(candidates["source1_entity_id"])
    truth = pd.read_csv(dataset / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    truth_map = {
        row.source1_entity_id: parse_ids(row.matched_entity_ids)
        for row in truth.itertuples(index=False) if row.source1_entity_id in source_ids
    }
    candidate_map = {
        row.source1_entity_id: parse_ids(row.candidate_entity_ids)
        for row in candidates.itertuples(index=False)
    }
    missing_rows = [
        (source, target)
        for source, true_ids in truth_map.items()
        for target in true_ids - candidate_map.get(source, set())
    ]
    missing = pd.DataFrame(missing_rows, columns=["source1_entity_id", "target_entity_id"])
    print(f"M2-missed true links: {len(missing):,}", flush=True)

    s1 = pd.read_csv(dataset / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                     nrows=len(candidates))
    s1 = normalize_frame(s1).rename(columns={"entity_id": "source1_entity_id"})
    required_targets = set(missing["target_entity_id"])
    target_chunks = []
    columns = ["entity_id", "business_name", "business_address", "country"]
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        for chunk in pd.read_csv(dataset / filename, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=columns, chunksize=200_000):
            selected = chunk[chunk["entity_id"].isin(required_targets)]
            if not selected.empty:
                target_chunks.append(normalize_frame(selected))
        print(f"Scanned {filename}", flush=True)
    targets = pd.concat(target_chunks, ignore_index=True).rename(columns={"entity_id": "target_entity_id"})
    pairs = missing.merge(
        s1[["source1_entity_id", "country_norm", "name_sorted", "name_no_legal", "address_sorted"]],
        on="source1_entity_id", validate="many_to_one", suffixes=("", "_s1"),
    ).merge(
        targets[["target_entity_id", "country_norm", "name_sorted", "name_no_legal", "address_sorted"]],
        on="target_entity_id", validate="many_to_one", suffixes=("_s1", "_target"),
    )
    same_country = pairs["country_norm_s1"].eq(pairs["country_norm_target"])
    pairs["recover_sorted_name"] = same_country & pairs["name_sorted_s1"].ne("") & pairs["name_sorted_s1"].eq(pairs["name_sorted_target"])
    pairs["recover_name_without_legal"] = same_country & pairs["name_no_legal_s1"].str.split().str.len().ge(2) & pairs["name_no_legal_s1"].eq(pairs["name_no_legal_target"])
    pairs["recover_sorted_address"] = same_country & pairs["address_sorted_s1"].ne("") & pairs["address_sorted_s1"].eq(pairs["address_sorted_target"])
    recovery_columns = ["recover_sorted_name", "recover_name_without_legal", "recover_sorted_address"]
    pairs["recover_any"] = pairs[recovery_columns].any(axis=1)
    report = pd.DataFrame({
        "signature": recovery_columns + ["recover_any"],
        "recovered_pairs": [int(pairs[column].sum()) for column in recovery_columns + ["recover_any"]],
        "recovered_pair_rate": [float(pairs[column].mean()) for column in recovery_columns + ["recover_any"]],
        "recovered_source1_entities": [int(pairs.loc[pairs[column], "source1_entity_id"].nunique()) for column in recovery_columns + ["recover_any"]],
    })
    output_dir = PROJECT_ROOT / "m3_work" / "outputs"
    report_dir = PROJECT_ROOT / "m3_work" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "m4_signature_recovery_100k.csv", index=False)
    (report_dir / "M4_SIGNATURE_RECOVERY_100K.md").write_text(
        "# Targeted retrieval signature recovery\n\n"
        f"- M2-missed true links analysed: {len(pairs):,}\n"
        f"- Target rows located: {len(targets):,}\n\n"
        + "```csv\n" + report.to_csv(index=False) + "```\n",
        encoding="utf-8",
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
