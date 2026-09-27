"""Build M4 Union Candidate Set (M2 Base Candidates + 3 Proven M4 Recovery Signatures).

Correctly loads the exact 100K sampled S1 entities and evaluates the combined oracle.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "m2_work" / "src"))
from blocking.normalization import normalize_address, normalize_country, normalize_name

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"

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


def f05(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def main():
    print("=" * 80)
    print("M4 UNION CANDIDATE GENERATION & ORACLE CEILING EVALUATION")
    print("=" * 80)

    start_time = time.time()

    # 1. Load Base M2.2 Candidates
    base_cand_file = PROJECT_ROOT / "m2_work" / "outputs" / "candidate_pairs.tsv"
    print(f"\n[1/5] Loading Base M2.2 Candidates from {base_cand_file.name}...")
    base_df = pd.read_csv(base_cand_file, sep="\t", dtype=str, keep_default_na=False)
    candidate_map: dict[str, set[str]] = {
        row.source1_entity_id: parse_ids(row.candidate_entity_ids)
        for row in base_df.itertuples(index=False)
    }
    s1_ids = list(candidate_map.keys())
    s1_id_set = set(s1_ids)
    print(f"Loaded {len(s1_ids):,} Source-1 entities with {sum(len(c) for c in candidate_map.values()):,} base candidate pairs")

    # 2. Correctly Load & Index Matching 100K S1 records from train_source1.tsv
    print("\n[2/5] Filtering and indexing matching 100K S1 records from train_source1.tsv...")
    s1_chunks = []
    cols = ["entity_id", "business_name", "business_address", "country"]
    for chunk in pd.read_csv(DATASET_DIR / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=cols, chunksize=300_000):
        matched = chunk[chunk["entity_id"].isin(s1_id_set)]
        if not matched.empty:
            s1_chunks.append(matched)

    s1_df = pd.concat(s1_chunks, ignore_index=True)
    print(f"Loaded {len(s1_df):,} matched S1 records")
    s1_norm = normalize_frame(s1_df)

    s1_sorted_name: dict[tuple[str, str], list[str]] = defaultdict(list)
    s1_no_legal: dict[tuple[str, str], list[str]] = defaultdict(list)
    s1_sorted_addr: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in s1_norm.itertuples(index=False):
        c = row.country_norm
        if not c:
            continue
        s_id = row.entity_id
        if row.name_sorted:
            s1_sorted_name[(c, row.name_sorted)].append(s_id)
        if row.name_no_legal and len(row.name_no_legal.split()) >= 2:
            s1_no_legal[(c, row.name_no_legal)].append(s_id)
        if row.address_sorted:
            s1_sorted_addr[(c, row.address_sorted)].append(s_id)

    print(f"Indexed S1 signatures: Sorted Name={len(s1_sorted_name):,}, No Legal={len(s1_no_legal):,}, Sorted Addr={len(s1_sorted_addr):,}")

    # 3. Stream S2 & S3 and Union Matching Candidates
    print("\n[3/5] Streaming S2 & S3 to recover missing signature matches...")
    new_links_added = 0

    for filename in ("train_source2.tsv", "train_source3.tsv"):
        f_start = time.time()
        file_rows = 0
        for chunk in pd.read_csv(DATASET_DIR / filename, sep="\t", dtype=str, keep_default_na=False, usecols=cols, chunksize=250_000):
            norm_chunk = normalize_frame(chunk)
            for row in norm_chunk.itertuples(index=False):
                c = row.country_norm
                if not c:
                    continue
                t_id = row.entity_id

                # Rule A: Sorted Name
                if row.name_sorted:
                    for s_id in s1_sorted_name.get((c, row.name_sorted), []):
                        if t_id not in candidate_map[s_id]:
                            candidate_map[s_id].add(t_id)
                            new_links_added += 1

                # Rule B: Name without legal suffix
                if row.name_no_legal and len(row.name_no_legal.split()) >= 2:
                    for s_id in s1_no_legal.get((c, row.name_no_legal), []):
                        if t_id not in candidate_map[s_id]:
                            candidate_map[s_id].add(t_id)
                            new_links_added += 1

                # Rule C: Sorted address
                if row.address_sorted:
                    for s_id in s1_sorted_addr.get((c, row.address_sorted), []):
                        if t_id not in candidate_map[s_id]:
                            candidate_map[s_id].add(t_id)
                            new_links_added += 1

            file_rows += len(chunk)
            print(f"  Processed {file_rows:,} rows from {filename}...", flush=True)

        print(f"Finished {filename} ({file_rows:,} rows) in {time.time() - f_start:.1f}s", flush=True)

    print(f"\nTotal new candidate links unioned: {new_links_added:,}")

    # 4. Save Unioned Candidates
    print("\n[4/5] Exporting Unioned Candidates to TSV...")
    out_dir = PROJECT_ROOT / "m3_work" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    union_cand_file = out_dir / "candidate_pairs_m4_union_100k.tsv"

    rows = [
        {"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(sorted(candidate_map.get(s1_id, set())))}
        for s1_id in s1_ids
    ]
    pd.DataFrame(rows).to_csv(union_cand_file, sep="\t", index=False)
    print(f"Saved: {union_cand_file}")

    # 5. Evaluate Oracle against matching Ground Truth
    print("\n[5/5] Evaluating Oracle Ceiling against Ground Truth...")
    gt_df = pd.read_csv(DATASET_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_matched = gt_df[gt_df["source1_entity_id"].isin(s1_id_set)]
    truth_map = {
        row.source1_entity_id: parse_ids(row.matched_entity_ids)
        for row in gt_matched.itertuples(index=False)
    }

    total_true = sum(len(ids) for ids in truth_map.values())
    retrieved_true = sum(len(candidate_map.get(s1, set()) & ids) for s1, ids in truth_map.items())

    entity_f05s = []
    for s1 in s1_ids:
        true_ids = truth_map.get(s1, set())
        if not true_ids:
            entity_f05s.append(1.0)
        else:
            pred_ids = candidate_map.get(s1, set()) & true_ids
            tp = len(pred_ids)
            rec = tp / len(true_ids)
            prec = 1.0 if tp > 0 else 0.0
            entity_f05s.append(f05(prec, rec))

    macro_oracle = sum(entity_f05s) / len(entity_f05s)
    total_cand_pairs = sum(len(c) for c in candidate_map.values())
    zero_cand_entities = sum(1 for c in candidate_map.values() if not c)

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("M4 UNION ORACLE EVALUATION RESULTS:")
    print("=" * 80)
    print(f"Total Candidate Pairs       : {total_cand_pairs:,} (was 1,799,071)")
    print(f"Mean Candidates per Entity  : {total_cand_pairs / len(s1_ids):.2f}")
    print(f"Zero-Candidate Entities     : {zero_cand_entities:,} (was 14,060)")
    print(f"True Matches Retrieved      : {retrieved_true:,} / {total_true:,} (was 172,163)")
    print(f"Candidate Pair Recall       : {retrieved_true / total_true:.4%} (was 49.74%)")
    print(f"MACRO F0.5 ORACLE CEILING   : {macro_oracle:.6f} (was 0.675934)")
    print(f"Total Elapsed Time          : {elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
