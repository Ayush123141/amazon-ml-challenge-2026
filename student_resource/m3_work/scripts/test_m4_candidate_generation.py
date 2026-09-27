"""Ultra-Fast Streamed Multi-Index Candidate Generation & Oracle Benchmark (100K S1).

Uses S1-indexed hash tables so streaming through 10.3M S2+S3 records takes ~30 seconds
with minimal memory footprint (< 300 MB RAM).
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "train"

LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "company", "co",
    "pvt", "private", "plc", "llp", "lp", "gmbh", "ag", "bv", "sa", "sarl", "enterprises",
    "enterprise", "services", "service", "solutions", "group", "holdings", "holding", "industries",
    "industry", "associates", "consulting", "international", "tech", "technologies", "technology",
}

POSTAL_RE = re.compile(r"\b\d{5,6}\b")
DIGITS_RE = re.compile(r"\d+")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    val = unicodedata.normalize("NFKC", str(value)).casefold()
    cleaned = []
    for c in val:
        cat = unicodedata.category(c)
        if cat.startswith("L") or cat.startswith("N") or cat.startswith("M") or c.isspace():
            cleaned.append(c)
        else:
            cleaned.append(" ")
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def parse_ids(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def extract_signatures(row: dict) -> list[tuple[str, str]]:
    """Generate all blocking (index_name, key) pairs for a record."""
    country = normalize_text(row.get("country", ""))
    name = normalize_text(row.get("business_name", ""))
    addr = normalize_text(row.get("business_address", ""))

    if not country:
        return []

    sigs = []
    name_tokens = name.split()
    addr_tokens = addr.split()

    # 1. Exact Name
    if name:
        sigs.append(("name_exact", f"{country}#{name}"))

    # 2. Sorted name tokens
    if len(name_tokens) >= 2:
        sigs.append(("name_sorted", f"{country}#{' '.join(sorted(name_tokens))}"))

    # 3. Legal-stripped name & sorted
    name_no_legal = [t for t in name_tokens if t not in LEGAL_SUFFIXES]
    if name_no_legal:
        no_legal_str = " ".join(name_no_legal)
        sigs.append(("name_no_legal", f"{country}#{no_legal_str}"))
        if len(name_no_legal) >= 2:
            sigs.append(("name_no_legal_sorted", f"{country}#{' '.join(sorted(name_no_legal))}"))

    # 4. Exact Address
    if addr:
        sigs.append(("addr_exact", f"{country}#{addr}"))

    # 5. Sorted Address
    if len(addr_tokens) >= 2:
        sigs.append(("addr_sorted", f"{country}#{' '.join(sorted(addr_tokens))}"))

    # 6. Postal code + first name token
    postals = POSTAL_RE.findall(addr)
    if postals and name_tokens:
        first_name_tok = name_no_legal[0] if name_no_legal else name_tokens[0]
        if len(first_name_tok) >= 3:
            for p in postals[:2]:
                sigs.append(("postal_name", f"{country}#{p}_{first_name_tok}"))

    # 7. House number + first name token
    digits = DIGITS_RE.findall(addr)
    if digits and name_tokens:
        first_name_tok = name_no_legal[0] if name_no_legal else name_tokens[0]
        if len(first_name_tok) >= 3 and len(digits[0]) >= 2:
            sigs.append(("house_name", f"{country}#{digits[0]}_{first_name_tok}"))

    # 8. Distinctive name token (length >= 6)
    distinctive = [t for t in name_no_legal if len(t) >= 6 and not t.isdigit()]
    for dt in distinctive[:2]:
        sigs.append(("distinctive_tok", f"{country}#{dt}"))

    return sigs


def f05(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_oracle(truth_map: dict[str, set[str]], candidate_map: dict[str, set[str]]) -> dict:
    entity_f05s = []
    total_tp = 0
    total_fn = 0
    total_true_matches = 0
    retrieved_true_matches = 0

    for s1_id, true_ids in truth_map.items():
        total_true_matches += len(true_ids)
        cand_ids = candidate_map.get(s1_id, set())
        predicted_ids = true_ids & cand_ids
        retrieved_true_matches += len(predicted_ids)

        tp = len(predicted_ids)
        fn = len(true_ids - predicted_ids)
        total_tp += tp
        total_fn += fn

        if len(true_ids) == 0:
            entity_f05s.append(1.0)
        else:
            prec = 1.0 if tp > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            entity_f05s.append(f05(prec, rec))

    macro_f05 = sum(entity_f05s) / len(entity_f05s) if entity_f05s else 0.0
    pair_recall = retrieved_true_matches / total_true_matches if total_true_matches > 0 else 0.0

    return {
        "macro_f05_oracle": macro_f05,
        "pair_recall": pair_recall,
        "retrieved_true_matches": retrieved_true_matches,
        "total_true_matches": total_true_matches,
    }


def main():
    print("=" * 80, flush=True)
    print("HIGH-SPEED MULTI-INDEX CANDIDATE GENERATION (100K S1 BENCHMARK)", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    # 1. Load 100K S1
    print("\n[1/4] Loading 100K Source-1 sample...", flush=True)
    s1_df = pd.read_csv(DATASET_DIR / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=100_000)
    s1_records = s1_df.to_dict("records")
    s1_ids = [r["entity_id"] for r in s1_records]
    print(f"Loaded {len(s1_records):,} S1 records", flush=True)

    # 2. Build S1 Reverse Query Index (key -> list of s1_ids)
    print("\n[2/4] Indexing 100K S1 signatures...", flush=True)
    s1_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in s1_records:
        s1_id = r["entity_id"]
        for idx_name, key in extract_signatures(r):
            s1_lookup[(idx_name, key)].append(s1_id)

    print(f"Total active S1 signature keys to query: {len(s1_lookup):,}", flush=True)

    # 3. Stream S2 (5M) and S3 (5.3M) and match against S1 lookup
    print("\n[3/4] Streaming S2 and S3 through S1 index...", flush=True)
    candidate_map = defaultdict(set)
    use_cols = ["entity_id", "business_name", "business_address", "country"]
    total_target_rows = 0

    # Max matches per key to prevent generic collisions (e.g. generic words)
    MAX_KEY_MATCHES = {
        "name_exact": 500,
        "name_sorted": 300,
        "name_no_legal": 200,
        "name_no_legal_sorted": 200,
        "addr_exact": 100,
        "addr_sorted": 100,
        "postal_name": 100,
        "house_name": 100,
        "distinctive_tok": 50,
    }

    key_hit_counts = defaultdict(int)

    for filename in ("train_source2.tsv", "train_source3.tsv"):
        file_start = time.time()
        file_rows = 0
        for chunk in pd.read_csv(DATASET_DIR / filename, sep="\t", dtype=str, keep_default_na=False, usecols=use_cols, chunksize=250_000):
            for r in chunk.to_dict("records"):
                target_id = r["entity_id"]
                sigs = extract_signatures(r)
                for idx_name, key in sigs:
                    k_tuple = (idx_name, key)
                    if k_tuple in s1_lookup:
                        if key_hit_counts[k_tuple] < MAX_KEY_MATCHES.get(idx_name, 100):
                            key_hit_counts[k_tuple] += 1
                            for s1_id in s1_lookup[k_tuple]:
                                if len(candidate_map[s1_id]) < 80:  # Cap at 80 cands/entity
                                    candidate_map[s1_id].add(target_id)
            file_rows += len(chunk)
            total_target_rows += len(chunk)
            print(f"  Processed {file_rows:,} rows from {filename}...", flush=True)

        print(f"Finished {filename} ({file_rows:,} rows) in {time.time() - file_start:.1f}s", flush=True)

    # 4. Evaluate Oracle Ceiling
    print("\n[4/4] Evaluating Ground Truth Oracle on 100K Benchmark...", flush=True)
    gt_df = pd.read_csv(DATASET_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=len(s1_records))
    truth_map = {
        row.source1_entity_id: parse_ids(row.matched_entity_ids)
        for row in gt_df.itertuples(index=False)
    }

    oracle_res = compute_oracle(truth_map, candidate_map)
    total_candidates = sum(len(cands) for cands in candidate_map.values())
    zero_cand_entities = sum(1 for s1_id in s1_ids if len(candidate_map.get(s1_id, set())) == 0)
    mean_candidates = total_candidates / len(s1_ids)
    # 5. Export Candidate Pairs TSV & Markdown Report
    output_dir = PROJECT_ROOT / "m3_work" / "outputs"
    report_dir = PROJECT_ROOT / "m3_work" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cand_rows = [
        {"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(sorted(candidate_map.get(s1_id, set())))}
        for s1_id in s1_ids
    ]
    cand_df = pd.DataFrame(cand_rows)
    cand_out_path = output_dir / "candidate_pairs_m4_100k.tsv"
    cand_df.to_csv(cand_out_path, sep="\t", index=False)
    print(f"Exported candidates to {cand_out_path}", flush=True)

    report_text = f"""# M4 Candidate Generation & Oracle Report (100K Benchmark)

- **Total Target Records Streamed**: {total_target_rows:,}
- **Total Candidate Pairs**: {total_candidates:,}
- **Mean Candidates per Entity**: {mean_candidates:.2f}
- **Zero-Candidate Entities**: {zero_cand_entities:,} ({zero_cand_entities/len(s1_ids):.2%})
- **Retrieved True Matches**: {oracle_res['retrieved_true_matches']:,} / {oracle_res['total_true_matches']:,}
- **Candidate Pair Recall**: {oracle_res['pair_recall']:.4%}
- **MACRO F0.5 ORACLE CEILING**: **{oracle_res['macro_f05_oracle']:.6f}**
- **Runtime**: {elapsed:.1f} seconds
"""
    (report_dir / "M4_ORACLE_100K_REPORT.md").write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 80, flush=True)
    print("FINAL BENCHMARK RESULTS:", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Target Records Streamed : {total_target_rows:,}", flush=True)
    print(f"Total Candidate Pairs Generated: {total_candidates:,}", flush=True)
    print(f"Mean Candidates per S1 Entity : {mean_candidates:.2f}", flush=True)
    print(f"Zero-Candidate Entities       : {zero_cand_entities:,} ({zero_cand_entities/len(s1_ids):.2%})", flush=True)
    print(f"Retrieved True Matches        : {oracle_res['retrieved_true_matches']:,} / {oracle_res['total_true_matches']:,}", flush=True)
    print(f"Candidate Pair Recall         : {oracle_res['pair_recall']:.4%}", flush=True)
    print(f"MACRO F0.5 ORACLE CEILING     : {oracle_res['macro_f05_oracle']:.6f}", flush=True)
    print(f"Total Runtime                 : {elapsed:.1f} seconds", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
