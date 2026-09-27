"""Pure alignment and threshold-search helpers for B007 post-processing."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

KEYS = ["query_id", "candidate_id", "candidate_source"]
REQUIRED = KEYS + ["country", "label", "probability"]
CHAMPION = {"macro_f05": 0.9617940138448521, "precision": 0.987924, "recall": 0.915786}


def source(candidate_id: str) -> str:
    if candidate_id.startswith("S2-"): return "S2"
    if candidate_id.startswith("S3-"): return "S3"
    raise ValueError(candidate_id)


def align_frames(left: Any, right: Any, left_name: str, right_name: str) -> tuple[Any, Any, dict[str, Any]]:
    for frame, name in ((left, left_name), (right, right_name)):
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing: raise ValueError(f"{name}: missing columns {missing}")
        if frame.duplicated(KEYS).any(): raise ValueError(f"{name}: duplicate canonical keys")
        derived = frame["candidate_id"].map(source)
        if not derived.equals(frame["candidate_source"].astype(str)): raise ValueError(f"{name}: candidate_source mismatch")
    if len(left) != len(right): raise ValueError(f"row count mismatch: {len(left)} != {len(right)}")
    a = left.sort_values(KEYS).reset_index(drop=True); b = right.sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS]): raise ValueError("zero-missing canonical join assertion failed")
    if not a["label"].astype(int).equals(b["label"].astype(int)): raise ValueError("labels differ")
    if not a["country"].astype(str).equals(b["country"].astype(str)): raise ValueError("countries differ")
    return a, b, {"status": "PASS", "row_count": len(a), "duplicate_keys": 0, "missing_joins": 0, "labels_identical": True, "countries_identical": True, "candidate_sources_identical": True}


def scores(queries: Sequence[str], candidates: Sequence[str], probabilities: Sequence[float]) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for query, candidate, probability in zip(queries, candidates, probabilities):
        result.setdefault(str(query), []).append((str(candidate), float(probability)))
    return result


def local_threshold_search(
    truth: Mapping[str, set[str]], candidate_scores: Mapping[str, Sequence[tuple[str, float]]],
    old_s2: float, old_s3: float, apply: Any, evaluate: Any, use_quarter_step: bool = True,
) -> tuple[float, float, dict[str, Any], list[dict[str, Any]]]:
    cache: dict[tuple[float, float], dict[str, Any]] = {}; trials: list[dict[str, Any]] = []
    def trial(t2: float, t3: float, stage: str) -> dict[str, Any]:
        key = (round(float(t2), 6), round(float(t3), 6))
        if key not in cache:
            metrics = dict(evaluate(truth, apply(candidate_scores, *key))); cache[key] = metrics
            trials.append({"stage": stage, "threshold_s2": key[0], "threshold_s3": key[1], "macro_f05": float(metrics["macro_f05"])})
        return cache[key]
    t2, t3 = float(old_s2), float(old_s3)
    stages = [(0.01, 0.05), (0.005, 0.015)] + ([(0.0025, 0.0075)] if use_quarter_step else [])
    for step, radius in stages:
        values2 = np.arange(max(0, t2-radius), min(1, t2+radius)+step/2, step)
        t2 = float(max(values2, key=lambda value: (float(trial(value, t3, f"step_{step}")["macro_f05"]), -float(value))))
        values3 = np.arange(max(0, t3-radius), min(1, t3+radius)+step/2, step)
        t3 = float(max(values3, key=lambda value: (float(trial(t2, value, f"step_{step}")["macro_f05"]), -float(value))))
    metrics = trial(t2, t3, "winner")
    return round(t2, 6), round(t3, 6), metrics, trials


def probability_mix(p03: Sequence[float], p01: Sequence[float], weight01: float) -> np.ndarray:
    if not 0 <= weight01 <= 1: raise ValueError(weight01)
    a = np.asarray(p03, dtype=np.float64); b = np.asarray(p01, dtype=np.float64)
    if a.shape != b.shape: raise ValueError("probability shape mismatch")
    return (1-weight01)*a + weight01*b

