"""Train an entity-disjoint logistic-regression M3 baseline and tune its threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "m2_work" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "m3_work" / "src"))

from evaluate_f05 import f05, score_entity
from features.pair_features import numeric_feature_columns


def evaluate_threshold(
    frame: pd.DataFrame, threshold: float, evaluation_ids: list[str], truth_map: dict[str, set[str]]
) -> dict[str, float | int]:
    predicted = frame[frame["probability"] >= threshold]
    predicted_map = predicted.groupby("source1_entity_id")["target_entity_id"].agg(set).to_dict()
    entities = [score_entity(truth_map.get(source, set()), predicted_map.get(source, set())) for source in evaluation_ids]
    result = pd.DataFrame(entities)
    labels = frame["label"].to_numpy()
    pair_predictions = (frame["probability"].to_numpy() >= threshold).astype(int)
    pair_precision, pair_recall, pair_f05, _ = precision_recall_fscore_support(
        labels, pair_predictions, beta=0.5, average="binary", zero_division=0
    )
    singleton = result[result["tp"] + result["fn"] == 0]
    non_singleton = result[result["tp"] + result["fn"] > 0]
    return {
        "threshold": threshold,
        "macro_f05": float(result["f05"].mean()),
        "pair_precision": float(pair_precision),
        "pair_recall": float(pair_recall),
        "pair_f05": float(pair_f05),
        "singleton_accuracy": float(singleton["f05"].mean()) if len(singleton) else 0.0,
        "non_singleton_f05": float(non_singleton["f05"].mean()) if len(non_singleton) else 0.0,
        "predicted_pairs": int(pair_predictions.sum()),
        "empty_predictions": int(sum(source not in predicted_map for source in evaluation_ids)),
        "false_positives_on_singletons": int(singleton["fp"].sum()) if len(singleton) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--candidate-file", type=Path,
        default=PROJECT_ROOT / "m2_work" / "outputs" / "candidate_pairs.tsv",
        help="Frozen grouped candidate file; supplies zero-candidate Source-1 entities for macro scoring.",
    )
    parser.add_argument("--max-source1", type=int, default=None,
                        help="Use the same prefix limit used to build a smoke feature artifact.")
    parser.add_argument("--run-name", default=None, help="Artifact name; defaults to the feature-file stem.")
    parser.add_argument("--model", choices=["logistic", "hist_gradient"], default="logistic")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    print("Loading feature artifact ...", flush=True)
    frame = pd.read_csv(args.features, compression="infer")
    run_name = args.run_name or args.features.name.removesuffix(".csv.gz").removesuffix(".csv")
    feature_columns = numeric_feature_columns(frame)
    candidate_ids = pd.read_csv(args.candidate_file, sep="\t", dtype=str, keep_default_na=False)["source1_entity_id"]
    if args.max_source1 is not None:
        candidate_ids = candidate_ids.iloc[:args.max_source1]
    candidate_ids = candidate_ids.tolist()
    ground_truth = pd.read_csv(PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                               keep_default_na=False)
    truth_map = {
        row.source1_entity_id: {item.strip() for item in row.matched_entity_ids.split(",") if item.strip()}
        for row in ground_truth.itertuples(index=False)
    }
    # Feature rows omit entities with no retrieved candidates. Split the full population first.
    entity_frame = pd.DataFrame({"source1_entity_id": candidate_ids})
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    entity_train_idx, entity_validation_idx = next(
        splitter.split(entity_frame, groups=entity_frame["source1_entity_id"])
    )
    train_ids = set(entity_frame.iloc[entity_train_idx]["source1_entity_id"])
    validation_ids = entity_frame.iloc[entity_validation_idx]["source1_entity_id"].tolist()
    train = frame[frame["source1_entity_id"].isin(train_ids)]
    validation = frame[frame["source1_entity_id"].isin(validation_ids)].copy()
    print(f"Training {args.model} model ...", flush=True)
    if args.model == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train[feature_columns], train["label"])
        validation["probability"] = model.predict_proba(validation[feature_columns])[:, 1]
    else:
        # Non-linear interactions are important: e.g., a matching house number is only useful with a similar street.
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, class_weight="balanced", random_state=args.seed,
        )
        train_matrix = train[feature_columns].astype(np.float32)
        validation_matrix = validation[feature_columns].astype(np.float32)
        # The Windows execution sandbox disallows sklearn's worker-pipe creation.
        # One thread is deterministic and avoids that environment limitation.
        with threadpool_limits(limits=1):
            model.fit(train_matrix, train["label"])
            validation["probability"] = model.predict_proba(validation_matrix)[:, 1]

    print("Optimizing per-entity prediction threshold ...", flush=True)
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    metrics = pd.DataFrame([
        evaluate_threshold(validation, float(threshold), validation_ids, truth_map) for threshold in thresholds
    ])
    best = metrics.sort_values(["macro_f05", "threshold"], ascending=[False, False]).iloc[0].to_dict()
    outputs = PROJECT_ROOT / "m3_work" / "outputs"
    reports = PROJECT_ROOT / "m3_work" / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = outputs / f"model_{args.model}_{run_name}.joblib"
    joblib.dump({"model": model, "features": feature_columns, "best_threshold": best["threshold"]}, model_path)
    print(f"Saved trained model checkpoint to {model_path}", flush=True)

    metrics.to_csv(outputs / f"m3_{args.model}_{run_name}_threshold_metrics.csv", index=False)
    validation[["source1_entity_id", "target_entity_id", "label", "probability"]].to_csv(
        outputs / f"m3_{args.model}_{run_name}_validation_scores.csv.gz", index=False, compression="gzip"
    )
    (reports / f"M3_2_{args.model.upper()}_{run_name.upper()}_REPORT.md").write_text(
        f"# M3.2 {args.model} baseline\n\n"
        f"- Entity-disjoint train entities: {len(train_ids):,}\n"
        f"- Entity-disjoint validation entities: {len(validation_ids):,}\n"
        f"- Features: {len(feature_columns)}\n"
        f"- Best threshold: {best['threshold']:.2f}\n"
        f"- Best validation Macro F0.5: {best['macro_f05']:.6f}\n"
        f"- Pair precision / recall / F0.5: {best['pair_precision']:.6f} / {best['pair_recall']:.6f} / {best['pair_f05']:.6f}\n"
        f"- Singleton accuracy: {best['singleton_accuracy']:.6f}\n"
        f"- Non-singleton Macro F0.5: {best['non_singleton_f05']:.6f}\n",
        encoding="utf-8",
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
