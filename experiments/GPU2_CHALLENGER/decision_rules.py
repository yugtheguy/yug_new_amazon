"""Pure candidate-competition decision logic for the GPU2 challenger.

This module deliberately knows nothing about feature generation or LightGBM.  It
can therefore be applied unchanged to B006A, B007, or final-test probabilities.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Mapping, Sequence


def candidate_source(candidate_id: str) -> str:
    if candidate_id.startswith("S2-"):
        return "S2"
    if candidate_id.startswith("S3-"):
        return "S3"
    raise ValueError(f"Unknown candidate namespace: {candidate_id}")


def competition_features(
    scores: Mapping[str, Sequence[tuple[str, float]]],
    strong_claimant_threshold: float = 0.80,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, float]]]:
    """Calculate query/source and target claimant competition statistics."""
    query_stats: dict[tuple[str, str], dict[str, float]] = {}
    claimants: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for query_id, rows in scores.items():
        by_source: dict[str, list[float]] = {"S2": [], "S3": []}
        for candidate_id, raw_probability in rows:
            probability = float(raw_probability)
            by_source[candidate_source(candidate_id)].append(probability)
            claimants[candidate_id].append((query_id, probability))
        for source, probabilities in by_source.items():
            ranked = sorted(probabilities, reverse=True)
            p1 = ranked[0] if ranked else 0.0
            p2 = ranked[1] if len(ranked) > 1 else 0.0
            p3 = ranked[2] if len(ranked) > 2 else 0.0
            query_stats[(query_id, source)] = {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p1_minus_p2": p1 - p2,
                "p1_minus_p3": p1 - p3,
                "number_candidates_above_0.50": float(sum(p >= 0.50 for p in ranked)),
                "above_0.70": float(sum(p >= 0.70 for p in ranked)),
                "above_0.80": float(sum(p >= 0.80 for p in ranked)),
                "above_0.90": float(sum(p >= 0.90 for p in ranked)),
            }
    target_stats: dict[str, dict[str, float]] = {}
    for candidate_id, rows in claimants.items():
        ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
        best = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        target_stats[candidate_id] = {
            "best_s1_claimant_score": best,
            "second_best_claimant_score": second,
            "ownership_margin": best - second,
            "number_strong_s1_claimants": float(sum(p >= strong_claimant_threshold for _, p in ranked)),
        }
    return query_stats, target_stats


def apply_decision_rule(
    scores: Mapping[str, Sequence[tuple[str, float]]],
    threshold_s2: float,
    threshold_s3: float,
    rule: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Apply an interpretable rule and global target uniqueness.

    The rule never forces top-1 and permits zero, one, or multiple matches per
    query.  The only global constraint is the production target uniqueness rule.
    """
    query_stats, target_stats = competition_features(scores)
    accepted: list[tuple[float, str, str]] = []
    ambiguity_bump = float(rule.get("ambiguity_bump", 0.0))
    ambiguity_count = int(rule.get("ambiguity_count", 2))
    ownership_margin = float(rule.get("ownership_margin", 0.0))
    confidence_override = float(rule.get("confidence_override", 1.01))
    query_tail_margin = rule.get("query_tail_margin")

    for query_id, rows in scores.items():
        for candidate_id, raw_probability in rows:
            probability = float(raw_probability)
            source = candidate_source(candidate_id)
            qstats = query_stats[(query_id, source)]
            threshold = threshold_s2 if source == "S2" else threshold_s3
            if qstats["above_0.80"] >= ambiguity_count:
                threshold += ambiguity_bump
            if probability < threshold:
                continue
            if query_tail_margin is not None and probability < confidence_override:
                if probability < qstats["p1"] - float(query_tail_margin):
                    continue
            tstats = target_stats[candidate_id]
            if probability < tstats["best_s1_claimant_score"]:
                continue
            if probability < confidence_override and tstats["ownership_margin"] < ownership_margin:
                continue
            accepted.append((probability, query_id, candidate_id))

    # Deterministic form of production target uniqueness.  Multiple targets may
    # still be assigned to a query; no S1 is ever forced to receive a match.
    accepted.sort(key=lambda item: (-item[0], item[1], item[2]))
    assigned: set[str] = set()
    result = {query_id: set() for query_id in scores}
    for _probability, query_id, candidate_id in accepted:
        if candidate_id not in assigned:
            result[query_id].add(candidate_id)
            assigned.add(candidate_id)
    return result


def default_rule_family() -> list[dict[str, Any]]:
    """Small, fixed calibration-only search family."""
    return [
        {"name": "baseline"},
        {"name": "ambiguity_bump_002", "ambiguity_bump": 0.02, "ambiguity_count": 2},
        {"name": "ambiguity_bump_005", "ambiguity_bump": 0.05, "ambiguity_count": 2},
        {"name": "ownership_margin_001", "ownership_margin": 0.01, "confidence_override": 0.95},
        {"name": "ownership_margin_003", "ownership_margin": 0.03, "confidence_override": 0.97},
        {"name": "ambiguity_002_ownership_001", "ambiguity_bump": 0.02, "ambiguity_count": 2, "ownership_margin": 0.01, "confidence_override": 0.95},
        {"name": "query_tail_030", "query_tail_margin": 0.30, "confidence_override": 0.95},
    ]


def select_rule_on_calibration(
    truth: Mapping[str, set[str]],
    scores: Mapping[str, Sequence[tuple[str, float]]],
    threshold_s2: float,
    threshold_s3: float,
    rules: Iterable[Mapping[str, Any]],
    evaluate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    for ordinal, raw_rule in enumerate(rules):
        rule = dict(raw_rule)
        predictions = apply_decision_rule(scores, threshold_s2, threshold_s3, rule)
        metrics = dict(evaluate(truth, predictions))
        trials.append({"ordinal": ordinal, "rule": rule, "metrics": metrics})
    # Stable tie-break favors the earlier/simpler rule, especially baseline.
    winner = max(trials, key=lambda row: (float(row["metrics"]["macro_f05"]), -int(row["ordinal"])))
    return dict(winner["rule"]), trials

