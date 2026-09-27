"""Deterministic, Unicode-safe pair features for candidate entity matches."""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
import pandas as pd


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


def _text_features(prefix: str, left: str, right: str, *, name: bool) -> dict[str, float | int]:
    left_tokens, right_tokens = token_set(left), token_set(right)
    overlap = left_tokens & right_tokens
    max_length = max(len(left), len(right), 1)
    result: dict[str, float | int] = {
        f"{prefix}_exact": int(left == right),
        f"{prefix}_token_jaccard": _jaccard(left_tokens, right_tokens),
        f"{prefix}_token_overlap_count": len(overlap),
        f"{prefix}_token_containment": _containment(left_tokens, right_tokens),
        f"{prefix}_char_similarity": _sequence_ratio(left, right),
        # SequenceMatcher ratio is a normalized edit-like similarity available without extra dependencies.
        f"{prefix}_edit_similarity": _sequence_ratio(left, right),
        f"{prefix}_length_difference_normalized": abs(len(left) - len(right)) / max_length,
        f"{prefix}_length_difference_raw": abs(len(left) - len(right)),
        f"{prefix}_token_count_difference": abs(len(left_tokens) - len(right_tokens)),
    }
    if name:
        result[f"{prefix}_prefix_similarity"] = _prefix_similarity(left, right)
        result[f"{prefix}_suffix_similarity"] = _suffix_similarity(left, right)
    else:
        left_digits = {token for token in left_tokens if any(char.isdigit() for char in token)}
        right_digits = {token for token in right_tokens if any(char.isdigit() for char in token)}
        result[f"{prefix}_numeric_token_overlap"] = len(left_digits & right_digits)
        result[f"{prefix}_digit_agreement"] = _jaccard(left_digits, right_digits)
    return result


def build_pair_features(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build all engineered feature columns from an already joined pair dataframe."""
    records = []
    for row in pairs.itertuples(index=False):
        source_tokens = token_set(row.name_normalized_s1) | token_set(row.address_normalized_s1)
        target_tokens = token_set(row.name_normalized_target) | token_set(row.address_normalized_target)
        shared_tokens = source_tokens & target_tokens
        values: dict[str, object] = {
            "source1_entity_id": row.source1_entity_id,
            "target_entity_id": row.target_entity_id,
            "label": int(row.label),
            "country_exact": int(row.country_raw_s1 == row.country_raw_target),
            "country_normalized_exact": int(row.country_normalized_s1 == row.country_normalized_target),
            "name_token_count_s1": len(token_set(row.name_normalized_s1)),
            "name_token_count_target": len(token_set(row.name_normalized_target)),
            "address_token_count_s1": len(token_set(row.address_normalized_s1)),
            "address_token_count_target": len(token_set(row.address_normalized_target)),
            "name_length_s1": len(row.name_normalized_s1),
            "name_length_target": len(row.name_normalized_target),
            "address_length_s1": len(row.address_normalized_s1),
            "address_length_target": len(row.address_normalized_target),
            "retrieved_by_exact_name": int(
                row.country_normalized_s1 == row.country_normalized_target
                and bool(row.name_normalized_s1)
                and row.name_normalized_s1 == row.name_normalized_target
            ),
            "retrieved_by_exact_address": int(
                row.country_normalized_s1 == row.country_normalized_target
                and bool(row.address_normalized_s1)
                and row.address_normalized_s1 == row.address_normalized_target
            ),
            "retrieved_by_exact_name_address": int(
                row.country_normalized_s1 == row.country_normalized_target
                and bool(row.name_normalized_s1)
                and bool(row.address_normalized_s1)
                and row.name_normalized_s1 == row.name_normalized_target
                and row.address_normalized_s1 == row.address_normalized_target
            ),
            # This candidate artifact was generated by the frozen M2.2 rare-token retriever.
            # Exact-block candidates add no incremental rows in the validated M2.4 experiment.
            "retrieved_by_rare_token": 1,
            "matching_rare_token_count": len(shared_tokens),
        }
        values.update(_text_features("name", row.name_normalized_s1, row.name_normalized_target, name=True))
        values.update(_text_features("address", row.address_normalized_s1, row.address_normalized_target, name=False))
        records.append(values)
    return pd.DataFrame.from_records(records)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns suitable for classical models; excludes identifiers and label."""
    return [column for column in frame.columns if column not in {"source1_entity_id", "target_entity_id", "label"}]
