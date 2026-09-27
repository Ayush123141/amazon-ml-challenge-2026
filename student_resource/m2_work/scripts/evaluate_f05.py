from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

M2_DIR = PROJECT_ROOT / "m2_work"
OUTPUT_DIR = M2_DIR / "outputs"


def parse_id_list(value):
    if value is None:
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def score_entity(true_ids, predicted_ids):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    tp = len(true_ids & predicted_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    # Special singleton / empty-ground-truth case.
    if len(true_ids) == 0:
        if len(predicted_ids) == 0:
            return {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f05": 1.0,
            }

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
        }

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = tp / (tp + fn)

    score = f05(precision, recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f05": score,
    }


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {sorted(missing)}"
        )

    return df


def load_predictions(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Predictions missing columns: {sorted(missing)}"
        )

    return df


def evaluate(ground_truth_path, prediction_path):
    gt = load_ground_truth(ground_truth_path)
    pred = load_predictions(prediction_path)

    gt_map = {
        row["source1_entity_id"]: parse_id_list(
            row["matched_entity_ids"]
        )
        for _, row in gt.iterrows()
    }

    pred_map = {
        row["source1_entity_id"]: parse_id_list(
            row["matched_entity_ids"]
        )
        for _, row in pred.iterrows()
    }

    results = []

    missing_predictions = 0
    extra_predictions = 0

    for source1_id, true_ids in gt_map.items():

        predicted_ids = pred_map.get(source1_id)

        if predicted_ids is None:
            predicted_ids = set()
            missing_predictions += 1

        result = score_entity(
            true_ids,
            predicted_ids,
        )

        result["source1_entity_id"] = source1_id
        result["true_count"] = len(true_ids)
        result["predicted_count"] = len(predicted_ids)

        results.append(result)

    for source1_id in pred_map:
        if source1_id not in gt_map:
            extra_predictions += 1

    results_df = pd.DataFrame(results)

    macro_f05 = results_df["f05"].mean()

    macro_precision = results_df["precision"].mean()
    macro_recall = results_df["recall"].mean()

    total_tp = results_df["tp"].sum()
    total_fp = results_df["fp"].sum()
    total_fn = results_df["fn"].sum()

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    singleton_mask = results_df["true_count"] == 0

    singleton_count = singleton_mask.sum()

    if singleton_count:
        singleton_accuracy = (
            results_df.loc[
                singleton_mask,
                "f05",
            ].mean()
        )
    else:
        singleton_accuracy = 0.0

    non_singleton_mask = results_df["true_count"] > 0

    non_singleton_count = non_singleton_mask.sum()

    if non_singleton_count:
        non_singleton_f05 = (
            results_df.loc[
                non_singleton_mask,
                "f05",
            ].mean()
        )
    else:
        non_singleton_f05 = 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "singleton_count": int(singleton_count),
        "singleton_accuracy": singleton_accuracy,
        "non_singleton_count": int(non_singleton_count),
        "non_singleton_f05": non_singleton_f05,
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "entity_count": len(results_df),
        "results_df": results_df,
    }


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python evaluate_f05.py "
            "<prediction_file>"
        )
        sys.exit(1)

    prediction_path = Path(sys.argv[1])

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: "
            f"{prediction_path}"
        )

    result = evaluate(
        GROUND_TRUTH,
        prediction_path,
    )

    print("\n" + "=" * 60)
    print("CANONICAL F0.5 EVALUATION")
    print("=" * 60)

    print(
        f"Entities evaluated       : "
        f"{result['entity_count']:,}"
    )

    print(
        f"Macro F0.5                : "
        f"{result['macro_f05']:.6f}"
    )

    print(
        f"Macro precision           : "
        f"{result['macro_precision']:.6f}"
    )

    print(
        f"Macro recall              : "
        f"{result['macro_recall']:.6f}"
    )

    print(
        f"Non-singleton F0.5        : "
        f"{result['non_singleton_f05']:.6f}"
    )

    print(
        f"Singleton count           : "
        f"{result['singleton_count']:,}"
    )

    print(
        f"Singleton accuracy        : "
        f"{result['singleton_accuracy']:.6f}"
    )

    print(
        f"Micro precision           : "
        f"{result['micro_precision']:.6f}"
    )

    print(
        f"Micro recall              : "
        f"{result['micro_recall']:.6f}"
    )

    print(
        f"TP                        : "
        f"{result['total_tp']:,}"
    )

    print(
        f"FP                        : "
        f"{result['total_fp']:,}"
    )

    print(
        f"FN                        : "
        f"{result['total_fn']:,}"
    )

    print(
        f"Missing predictions       : "
        f"{result['missing_predictions']:,}"
    )

    print(
        f"Extra prediction S1 IDs   : "
        f"{result['extra_predictions']:,}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()