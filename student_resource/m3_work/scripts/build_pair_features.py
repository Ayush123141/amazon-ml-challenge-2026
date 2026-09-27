"""Build labelled M3 features from the frozen 100K M2 candidate artifact."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "m2_work" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "m3_work" / "src"))

from blocking.normalization import normalize_address, normalize_country, normalize_name
from features.pair_features import build_pair_features


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_entities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["entity_id"] = result["entity_id"].astype(str)
    result["country_raw"] = result["country"].fillna("").astype(str)
    result["country_normalized"] = result["country_raw"].map(normalize_country)
    result["name_normalized"] = result["business_name"].fillna("").map(normalize_name)
    result["address_normalized"] = result["business_address"].fillna("").map(normalize_address)
    return result


def build_target_cache(dataset: Path, target_ids: set[str], cache_path: Path) -> None:
    """Build a disk-backed normalized lookup for only candidate target IDs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("""CREATE TABLE IF NOT EXISTS targets (
        target_entity_id TEXT PRIMARY KEY, country_raw TEXT, country_normalized TEXT,
        name_normalized TEXT, address_normalized TEXT
    )""")
    existing = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    if existing == len(target_ids):
        connection.close()
        return
    if existing:
        # An interrupted cache build is not a valid input to a reproducible feature run.
        connection.execute("DELETE FROM targets")
        connection.commit()
    use_columns = ["entity_id", "business_name", "business_address", "country"]
    inserted = 0
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        for chunk in pd.read_csv(dataset / filename, sep="\t", dtype=str, keep_default_na=False, usecols=use_columns,
                                 chunksize=200_000):
            selected = chunk[chunk["entity_id"].isin(target_ids)]
            if not selected.empty:
                normalized = normalize_entities(selected)
                records = normalized[["entity_id", "country_raw", "country_normalized", "name_normalized", "address_normalized"]]
                connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?)", records.itertuples(index=False, name=None))
                inserted += len(records)
        connection.commit()
        print(f"Cached candidate targets from {filename}: {inserted:,}", flush=True)
    if not inserted:
        raise ValueError("No candidate target IDs were found in Sources 2 and 3.")
    connection.close()


def load_batch_targets(connection: sqlite3.Connection, target_ids: list[str]) -> pd.DataFrame:
    """Read at most one feature batch worth of targets from SQLite."""
    rows: list[tuple[str, str, str, str, str]] = []
    unique_ids = list(set(target_ids))
    for start in range(0, len(unique_ids), 900):
        chunk = unique_ids[start:start + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(connection.execute(
            f"SELECT target_entity_id, country_raw, country_normalized, name_normalized, address_normalized "
            f"FROM targets WHERE target_entity_id IN ({placeholders})", chunk
        ).fetchall())
    result = pd.DataFrame(rows, columns=["target_entity_id", "country_raw", "country_normalized", "name_normalized", "address_normalized"])
    if len(result) != len(unique_ids):
        raise ValueError(f"Target cache is missing {len(unique_ids) - len(result):,} candidate IDs.")
    return result


def write_report(features: pd.DataFrame, report_path: Path, max_source1: int | None) -> None:
    numeric = features.drop(columns=["source1_entity_id", "target_entity_id"])
    summary = numeric.groupby("label").mean(numeric_only=True).T
    selected = [column for column in summary.index if column.endswith(("exact", "jaccard", "char_similarity", "rare_token_count"))]
    lines = [
        "# M3.1 Feature Report",
        "",
        f"- Candidate pairs: {len(features):,}",
        f"- Source-1 entities represented: {features['source1_entity_id'].nunique():,}",
        f"- Positives: {int(features['label'].sum()):,}",
        f"- Negatives: {int((features['label'] == 0).sum()):,}",
        f"- Positive rate: {features['label'].mean():.6%}",
        f"- Source-1 limit: {max_source1 if max_source1 is not None else 'all candidate rows'}",
        f"- Duplicate pair rows: {int(features.duplicated(['source1_entity_id', 'target_entity_id']).sum()):,}",
        f"- Missing feature values: {int(features.isna().sum().sum()):,}",
        "",
        "## Selected mean feature values by label",
        "",
        "```csv",
        summary.loc[selected].round(6).to_csv(),
        "```",
        "",
        "## Missing values by column",
        "",
        "```csv",
        features.isna().sum().to_frame("missing_values").to_csv(),
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-source1", type=int, default=None, help="Smoke-test limit in candidate-file order.")
    parser.add_argument("--batch-source1", type=int, default=5_000,
                        help="Number of Source-1 candidate rows to materialize per feature batch.")
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=PROJECT_ROOT / "m3_work" / "outputs" / "candidate_pairs_m4_union_100k.tsv",
        help="Input candidate pairs TSV file",
    )
    parser.add_argument("--target-cache", type=Path, default=PROJECT_ROOT / "m3_work" / "outputs" / "candidate_target_cache_m4.sqlite")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "m3_work" / "outputs" / "pair_features_m4_union_100k.csv.gz")
    args = parser.parse_args()

    dataset = PROJECT_ROOT / "dataset" / "train"
    candidate_path = args.candidate_file
    print(f"Loading candidate file from {candidate_path} ...", flush=True)
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    if args.max_source1 is not None:
        candidates = candidates.iloc[:args.max_source1].copy()

    source_ids = set(candidates["source1_entity_id"])
    print(f"Loading matching Source-1 records ({len(source_ids):,} entities) ...", flush=True)
    s1_chunks = []
    cols = ["entity_id", "business_name", "business_address", "country"]
    for chunk in pd.read_csv(dataset / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=cols, chunksize=300_000):
        matched = chunk[chunk["entity_id"].isin(source_ids)]
        if not matched.empty:
            s1_chunks.append(matched)

    s1 = pd.concat(s1_chunks, ignore_index=True)
    s1 = normalize_entities(s1).rename(columns={"entity_id": "source1_entity_id"})
    print("Streaming candidate targets from Sources 2 and 3 ...", flush=True)
    target_ids = {target for values in candidates["candidate_entity_ids"].map(parse_ids) for target in values}
    build_target_cache(dataset, target_ids, args.target_cache)

    gt = pd.read_csv(dataset / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    print("Expanding and labelling candidate pairs ...", flush=True)
    truth = {row.source1_entity_id: set(parse_ids(row.matched_entity_ids)) for row in gt.itertuples(index=False)}
    source_cols = ["source1_entity_id", "country_raw", "country_normalized", "name_normalized", "address_normalized"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature artifact: {args.output}")

    s1_lookup = s1[source_cols].set_index("source1_entity_id")
    target_connection = sqlite3.connect(args.target_cache)
    total_pairs = positives = missing_values = duplicate_pairs = 0
    feature_sums: dict[int, pd.Series] = {}
    feature_counts: dict[int, int] = {}
    written_header = False
    for start in range(0, len(candidates), args.batch_source1):
        candidate_batch = candidates.iloc[start:start + args.batch_source1]
        expanded = candidate_batch.assign(target_entity_id=candidate_batch["candidate_entity_ids"].map(parse_ids)).explode("target_entity_id")
        expanded = expanded[expanded["target_entity_id"].notna() & expanded["target_entity_id"].ne("")]
        expanded = expanded[["source1_entity_id", "target_entity_id"]].copy()
        expanded["label"] = [int(target in truth.get(source, set())) for source, target in expanded.itertuples(index=False, name=None)]
        joined = expanded.join(s1_lookup, on="source1_entity_id", how="left", validate="many_to_one")
        joined = joined.rename(columns={column: f"{column}_s1" for column in ["country_raw", "country_normalized", "name_normalized", "address_normalized"]})
        target_batch = load_batch_targets(target_connection, expanded["target_entity_id"].tolist()).set_index("target_entity_id")
        joined = joined.join(target_batch, on="target_entity_id", how="left", validate="many_to_one")
        joined = joined.rename(columns={column: f"{column}_target" for column in ["country_raw", "country_normalized", "name_normalized", "address_normalized"]})
        if joined[["name_normalized_s1", "name_normalized_target"]].isna().any().any():
            raise ValueError("A candidate ID was absent from its source table.")
        features = build_pair_features(joined)
        duplicate_pairs += int(features.duplicated(["source1_entity_id", "target_entity_id"]).sum())
        missing_values += int(features.isna().sum().sum())
        features.to_csv(args.output, mode="a", header=not written_header, index=False, compression="gzip")
        written_header = True
        total_pairs += len(features)
        positives += int(features["label"].sum())
        for label, group in features.groupby("label"):
            numeric = group.drop(columns=["source1_entity_id", "target_entity_id"]).sum(numeric_only=True)
            feature_sums[label] = feature_sums.get(label, numeric * 0).add(numeric, fill_value=0)
            feature_counts[label] = feature_counts.get(label, 0) + len(group)
        print(f"Feature batches: {min(start + len(candidate_batch), len(candidates)):,}/{len(candidates):,}; pairs={total_pairs:,}", flush=True)
    if duplicate_pairs:
        raise ValueError(f"Duplicate candidate pairs found: {duplicate_pairs:,}")
    target_connection.close()
    report_path = PROJECT_ROOT / "m3_work" / "reports" / "M3_1_FEATURE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    means = pd.DataFrame({label: values / feature_counts[label] for label, values in feature_sums.items()})
    report_path.write_text(
        "# M3.1 Feature Report\n\n"
        f"- Candidate pairs: {total_pairs:,}\n"
        f"- Source-1 entities represented: {int((candidates['candidate_entity_ids'] != '').sum()):,}\n"
        f"- Positives: {positives:,}\n"
        f"- Negatives: {total_pairs - positives:,}\n"
        f"- Positive rate: {positives / total_pairs:.6%}\n"
        f"- Source-1 limit: {args.max_source1 if args.max_source1 is not None else 'all candidate rows'}\n"
        f"- Duplicate pair rows: {duplicate_pairs:,}\n"
        f"- Missing feature values: {missing_values:,}\n\n"
        "## Mean feature values by label\n\n```csv\n" + means.round(6).to_csv() + "```\n",
        encoding="utf-8",
    )
    print(f"Features: {args.output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
