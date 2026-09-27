"""Full-Scale Test Set Inference Pipeline for Amazon ML Challenge 2026.

Streams all 1,732,545 Test Source-1 entities against Test Source-2 and Test Source-3,
computes pairwise features, predicts with the trained gradient boosting model,
and exports the two final submission files:
  1. output/matching_results.tsv (Scored on the leaderboard)
  2. output/candidate_pairs.tsv (Candidate blocking set)

Can run locally or on AWS EC2 (m6i.2xlarge / c6i.2xlarge).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "m2_work" / "src"))
from blocking.normalization import normalize_address, normalize_country, normalize_name

LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "company", "co",
    "pvt", "private", "plc", "llp", "lp", "gmbh", "ag", "bv", "sa", "sarl",
}


def sorted_tokens(value: str) -> str:
    return " ".join(sorted(value.split())) if value else ""


def name_without_legal_suffix(value: str) -> str:
    return " ".join(token for token in value.split() if token not in LEGAL_SUFFIXES) if value else ""


def token_set(value: str) -> set[str]:
    return set(value.split()) if value else set()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _containment(left: set[str], right: set[str]) -> float:
    smaller = min(len(left), len(right))
    return len(left & right) / smaller if smaller else float(not left and not right)


def _sequence_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _prefix_similarity(left: str, right: str) -> float:
    limit = min(len(left), len(right))
    count = 0
    while count < limit and left[count] == right[count]:
        count += 1
    return count / max(len(left), len(right), 1)


def _suffix_similarity(left: str, right: str) -> float:
    return _prefix_similarity(left[::-1], right[::-1])


def normalize_record(row: dict) -> dict:
    c_raw = str(row.get("country", "")).strip()
    c_norm = normalize_country(c_raw)
    n_norm = normalize_name(row.get("business_name", ""))
    a_norm = normalize_address(row.get("business_address", ""))
    return {
        "entity_id": str(row.get("entity_id", "")),
        "country_raw": c_raw,
        "country_norm": c_norm,
        "name_norm": n_norm,
        "address_norm": a_norm,
        "name_sorted": sorted_tokens(n_norm),
        "name_no_legal": name_without_legal_suffix(n_norm),
        "address_sorted": sorted_tokens(a_norm),
    }


def compute_pair_features(s1: dict, target: dict) -> list[float]:
    """Compute exact 37 numeric features matching model training schema."""
    s1_name_tok = token_set(s1["name_norm"])
    s1_addr_tok = token_set(s1["address_norm"])
    t_name_tok = token_set(target["name_norm"])
    t_addr_tok = token_set(target["address_norm"])

    s1_tokens = s1_name_tok | s1_addr_tok
    t_tokens = t_name_tok | t_addr_tok
    shared_tokens = s1_tokens & t_tokens

    name_overlap = s1_name_tok & t_name_tok
    addr_overlap = s1_addr_tok & t_addr_tok

    s1_digits = {tok for tok in s1_addr_tok if any(c.isdigit() for c in tok)}
    t_digits = {tok for tok in t_addr_tok if any(c.isdigit() for c in tok)}

    name_seq_ratio = _sequence_ratio(s1["name_norm"], target["name_norm"])
    addr_seq_ratio = _sequence_ratio(s1["address_norm"], target["address_norm"])

    max_name_len = max(len(s1["name_norm"]), len(target["name_norm"]), 1)
    max_addr_len = max(len(s1["address_norm"]), len(target["address_norm"]), 1)

    feats = [
        # Country
        float(s1["country_raw"] == target["country_raw"]),
        float(s1["country_norm"] == target["country_norm"]),
        # Token counts & lengths
        float(len(s1_name_tok)),
        float(len(t_name_tok)),
        float(len(s1_addr_tok)),
        float(len(t_addr_tok)),
        float(len(s1["name_norm"])),
        float(len(target["name_norm"])),
        float(len(s1["address_norm"])),
        float(len(target["address_norm"])),
        # Retrieval indicators
        float(s1["country_norm"] == target["country_norm"] and bool(s1["name_norm"]) and s1["name_norm"] == target["name_norm"]),
        float(s1["country_norm"] == target["country_norm"] and bool(s1["address_norm"]) and s1["address_norm"] == target["address_norm"]),
        float(len(shared_tokens)),
        _jaccard(s1_tokens, t_tokens),
        _containment(s1_tokens, t_tokens),
        # Name text features
        float(s1["name_norm"] == target["name_norm"]),
        _jaccard(s1_name_tok, t_name_tok),
        float(len(name_overlap)),
        _containment(s1_name_tok, t_name_tok),
        name_seq_ratio,
        name_seq_ratio,  # edit_similarity
        abs(len(s1["name_norm"]) - len(target["name_norm"])) / max_name_len,
        float(abs(len(s1["name_norm"]) - len(target["name_norm"]))),
        float(abs(len(s1_name_tok) - len(t_name_tok))),
        _prefix_similarity(s1["name_norm"], target["name_norm"]),
        _suffix_similarity(s1["name_norm"], target["name_norm"]),
        # Address text features
        float(s1["address_norm"] == target["address_norm"]),
        _jaccard(s1_addr_tok, t_addr_tok),
        float(len(addr_overlap)),
        _containment(s1_addr_tok, t_addr_tok),
        addr_seq_ratio,
        addr_seq_ratio,  # edit_similarity
        abs(len(s1["address_norm"]) - len(target["address_norm"])) / max_addr_len,
        float(abs(len(s1["address_norm"]) - len(target["address_norm"]))),
        float(abs(len(s1_addr_tok) - len(t_addr_tok))),
        float(len(s1_digits & t_digits)),
        _jaccard(s1_digits, t_digits),
    ]
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=Path, default=PROJECT_ROOT / "m3_work" / "outputs" / "model_hist_gradient_m4_union_100k.joblib")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "dataset" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: FULL-SCALE TEST SET INFERENCE")
    print("=" * 80)

    # 1. Load Trained Model Checkpoint
    print(f"\n[1/5] Loading trained model from {args.model_checkpoint}...")
    saved = joblib.load(args.model_checkpoint)
    model = saved["model"]
    threshold = float(saved.get("best_threshold", 0.85))
    print(f"Model loaded successfully. Optimal Macro F0.5 Threshold: {threshold:.2f}")

    # 2. Load & Index all 1,732,545 Test Source-1 Entities
    s1_file = args.test_dir / "test_source1.tsv"
    print(f"\n[2/5] Loading Test Source-1 entities from {s1_file}...")
    s1_df = pd.read_csv(s1_file, sep="\t", dtype=str, keep_default_na=False)
    s1_count = len(s1_df)
    print(f"Total Test Source-1 Entities: {s1_count:,}")

    s1_store = {}
    s1_name_exact = defaultdict(list)
    s1_name_sorted = defaultdict(list)
    s1_name_no_legal = defaultdict(list)
    s1_addr_sorted = defaultdict(list)

    for row in s1_df.to_dict("records"):
        norm = normalize_record(row)
        s_id = norm["entity_id"]
        s1_store[s_id] = norm
        c = norm["country_norm"]
        if not c:
            continue
        if norm["name_norm"]:
            s1_name_exact[(c, norm["name_norm"])].append(s_id)
        if norm["name_sorted"]:
            s1_name_sorted[(c, norm["name_sorted"])].append(s_id)
        if norm["name_no_legal"] and len(norm["name_no_legal"].split()) >= 2:
            s1_name_no_legal[(c, norm["name_no_legal"])].append(s_id)
        if norm["address_sorted"]:
            s1_addr_sorted[(c, norm["address_sorted"])].append(s_id)

    print(f"Indexed {len(s1_store):,} Test Source-1 records into hash tables.")

    # 3. Stream Test Source 2 & Source 3 to collect candidates
    print("\n[3/5] Streaming Test Source 2 and Source 3 to generate candidate pairs...")
    candidate_map = defaultdict(set)
    target_store = {}
    cols = ["entity_id", "business_name", "business_address", "country"]

    for filename in ("test_source2.tsv", "test_source3.tsv"):
        f_start = time.time()
        file_path = args.test_dir / filename
        file_rows = 0
        for chunk in pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, usecols=cols, chunksize=250_000):
            for row in chunk.to_dict("records"):
                norm = normalize_record(row)
                t_id = norm["entity_id"]
                c = norm["country_norm"]
                if not c:
                    continue

                # Query indexes
                hits = set()
                if norm["name_norm"]:
                    hits.update(s1_name_exact.get((c, norm["name_norm"]), []))
                if norm["name_sorted"]:
                    hits.update(s1_name_sorted.get((c, norm["name_sorted"]), []))
                if norm["name_no_legal"] and len(norm["name_no_legal"].split()) >= 2:
                    hits.update(s1_name_no_legal.get((c, norm["name_no_legal"]), []))
                if norm["address_sorted"]:
                    hits.update(s1_addr_sorted.get((c, norm["address_sorted"]), []))

                if hits:
                    target_store[t_id] = norm
                    for s_id in hits:
                        if len(candidate_map[s_id]) < 60:
                            candidate_map[s_id].add(t_id)

            file_rows += len(chunk)
            print(f"  Processed {file_rows:,} rows from {filename}...", flush=True)

        print(f"Finished {filename} ({file_rows:,} rows) in {time.time() - f_start:.1f}s")

    total_cand_pairs = sum(len(c) for c in candidate_map.values())
    print(f"Total Candidate Pairs Generated: {total_cand_pairs:,} (for {len(candidate_map):,} entities)")

    # 4. Predict Matches with Trained Classifier
    print(f"\n[4/5] Scoring candidate pairs with Gradient Boosting (threshold >= {threshold:.2f})...")
    matched_results_map = defaultdict(list)
    scored_pairs = 0
    positive_matches = 0

    s1_all_ids = s1_df["entity_id"].tolist()
    batch_features = []
    batch_pair_keys = []

    for s1_id in s1_all_ids:
        cands = candidate_map.get(s1_id, set())
        if not cands:
            continue
        s1_rec = s1_store[s1_id]
        for t_id in cands:
            t_rec = target_store.get(t_id)
            if not t_rec:
                continue
            feats = compute_pair_features(s1_rec, t_rec)
            batch_features.append(feats)
            batch_pair_keys.append((s1_id, t_id))

            if len(batch_features) >= args.batch_size:
                X = np.array(batch_features, dtype=np.float32)
                probs = model.predict_proba(X)[:, 1]
                for (s_k, t_k), p in zip(batch_pair_keys, probs):
                    if p >= threshold:
                        matched_results_map[s_k].append(t_k)
                        positive_matches += 1
                scored_pairs += len(batch_features)
                batch_features = []
                batch_pair_keys = []
                print(f"  Scored {scored_pairs:,}/{total_cand_pairs:,} pairs (matches found: {positive_matches:,})...", flush=True)

    if batch_features:
        X = np.array(batch_features, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        for (s_k, t_k), p in zip(batch_pair_keys, probs):
            if p >= threshold:
                matched_results_map[s_k].append(t_k)
                positive_matches += 1
        scored_pairs += len(batch_features)

    print(f"\nInference complete! Total pairs scored: {scored_pairs:,}, Final Matches: {positive_matches:,}")

    # 5. Export Output Files (matching_results.tsv and candidate_pairs.tsv)
    print("\n[5/5] Exporting final submission files...")
    matching_file = args.output_dir / "matching_results.tsv"
    candidate_file = args.output_dir / "candidate_pairs.tsv"

    matching_rows = [
        {"source1_entity_id": s_id, "matched_entity_ids": ",".join(matched_results_map.get(s_id, []))}
        for s_id in s1_all_ids
    ]
    pd.DataFrame(matching_rows).to_csv(matching_file, sep="\t", index=False)
    print(f"  [OK] Exported: {matching_file} ({len(matching_rows):,} rows)")

    candidate_rows = [
        {"source1_entity_id": s_id, "candidate_entity_ids": ",".join(sorted(candidate_map.get(s_id, set())))}
        for s_id in s1_all_ids
    ]
    pd.DataFrame(candidate_rows).to_csv(candidate_file, sep="\t", index=False)
    print(f"  [OK] Exported: {candidate_file} ({len(candidate_rows):,} rows)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("SUCCESS: Full Test Inference Finished!")
    print(f"Total Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    print(f"Leaderboard file ready: {matching_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
