"""Pure validation, probability-table, blend, and diversity helpers."""

from __future__ import annotations

import collections
from typing import Any, Mapping, Sequence

import numpy as np

KEY_COLUMNS = ["query_id", "candidate_id", "candidate_source"]
BASELINE_B006A = {"macro_f05": 0.931218, "precision": 0.974840, "recall": 0.862974}
BASELINE_B007 = {"macro_f05": 0.9617940138448521, "precision": 0.987924, "recall": 0.915786}


def candidate_source(candidate_id: str) -> str:
    if candidate_id.startswith("S2-"):
        return "S2"
    if candidate_id.startswith("S3-"):
        return "S3"
    raise ValueError(f"Unknown candidate namespace: {candidate_id}")


def validate_numeric_matrix(matrix: Any, labels: np.ndarray, feature_names: Sequence[str], split: str) -> None:
    if tuple(matrix.shape) != (len(labels), len(feature_names)):
        raise ValueError(f"{split}: matrix/label/schema shape mismatch: {matrix.shape}, {len(labels)}, {len(feature_names)}")
    if any(str(dtype) == "object" for dtype in getattr(matrix, "dtypes", [])):
        raise TypeError(f"{split}: object feature column detected")
    values = matrix.to_numpy(copy=False) if hasattr(matrix, "to_numpy") else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{split}: non-numeric matrix dtype {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError(f"{split}: NaN or infinite feature value detected")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split}: labels are not binary")


def probability_rows(
    query_ids: Sequence[str], candidate_ids: Sequence[str], countries: Sequence[str],
    labels: np.ndarray, probabilities: Sequence[float], probability_name: str,
) -> dict[str, list[Any]]:
    count = len(query_ids)
    if not all(len(values) == count for values in (candidate_ids, countries, labels, probabilities)):
        raise ValueError("Probability output is not row-aligned")
    return {
        "query_id": [str(value) for value in query_ids],
        "candidate_id": [str(value) for value in candidate_ids],
        "candidate_source": [candidate_source(str(value)) for value in candidate_ids],
        "country": [str(value) for value in countries],
        "label": [int(value) for value in labels],
        probability_name: [float(value) for value in probabilities],
    }


def scores_from_columns(query_ids: Sequence[str], candidate_ids: Sequence[str], probabilities: Sequence[float]) -> dict[str, list[tuple[str, float]]]:
    scores: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for query_id, candidate_id, probability in zip(query_ids, candidate_ids, probabilities):
        scores[str(query_id)].append((str(candidate_id), float(probability)))
    return dict(scores)


def assert_canonical_alignment(left: Any, right: Any, left_name: str, right_name: str) -> tuple[Any, Any]:
    for frame, name in ((left, left_name), (right, right_name)):
        missing = [column for column in KEY_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{name}: missing key columns {missing}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name}: duplicate canonical keys")
    if len(left) != len(right):
        raise ValueError(f"row-count mismatch: {left_name}={len(left)}, {right_name}={len(right)}")
    left_sorted = left.sort_values(KEY_COLUMNS).reset_index(drop=True)
    right_sorted = right.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not left_sorted[KEY_COLUMNS].equals(right_sorted[KEY_COLUMNS]):
        raise ValueError(f"missing or differing canonical keys between {left_name} and {right_name}")
    return left_sorted, right_sorted


def blend_probabilities(primary: Sequence[float], auxiliary: Sequence[float], auxiliary_weight: float) -> np.ndarray:
    if not 0.0 <= auxiliary_weight <= 1.0:
        raise ValueError("blend weight must be in [0, 1]")
    primary_array = np.asarray(primary, dtype=np.float64)
    auxiliary_array = np.asarray(auxiliary, dtype=np.float64)
    if primary_array.shape != auxiliary_array.shape:
        raise ValueError("blend arrays have different shapes")
    return (1.0 - auxiliary_weight) * primary_array + auxiliary_weight * auxiliary_array


def probability_distribution(probabilities: Sequence[float]) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    return {
        "min": float(values.min()), "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)), "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
    }


def diversity_report(
    truth: Mapping[str, set[str]], primary_predictions: Mapping[str, set[str]],
    auxiliary_predictions: Mapping[str, set[str]], primary_probabilities: Sequence[float],
    auxiliary_probabilities: Sequence[float],
) -> dict[str, Any]:
    primary_missed_aux_got = primary_fp_aux_rejected = disagreement_count = 0
    for query_id, expected in truth.items():
        primary = primary_predictions.get(query_id, set())
        auxiliary = auxiliary_predictions.get(query_id, set())
        primary_missed_aux_got += len((expected - primary) & auxiliary)
        primary_fp_aux_rejected += len((primary - expected) - auxiliary)
        disagreement_count += len(primary ^ auxiliary)
    first = np.asarray(primary_probabilities, dtype=np.float64)
    second = np.asarray(auxiliary_probabilities, dtype=np.float64)
    correlation = float(np.corrcoef(first, second)[0, 1]) if first.size > 1 and first.std() and second.std() else None
    return {
        "positives_primary_misses_auxiliary_gets": primary_missed_aux_got,
        "false_positives_primary_makes_auxiliary_rejects": primary_fp_aux_rejected,
        "prediction_correlation": correlation,
        "entity_link_disagreement_count": disagreement_count,
    }

