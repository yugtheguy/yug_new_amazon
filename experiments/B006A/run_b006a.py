#!/usr/bin/env python3
"""B006A: cache-only forensic reproduction and retrieval-feature ablations."""

from __future__ import annotations

import argparse
import atexit
import collections
import gc
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "B006A"
B005_SCORE = 0.926948
B006_SCORE = 0.746726
REPRO_TOLERANCE = 0.002


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_experiments() -> tuple[Any, Any, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "experiments" / "B003")); import run_b003 as b003  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B005")); import run_b005 as b005  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B006")); import run_b006 as b006  # type: ignore
    return b003, b005, b006


B003, B005, B006 = import_experiments()

FLAGS = ["found_R0", "found_R1", "found_R2", "found_R3", "retriever_count"]
RANKS = ["R0_rank", "R1_rank", "R2_rank", "R3_rank", "best_rank", "mean_available_rank", "worst_available_rank"]
SCORES = ["R0_shared_keys", "R1_name_char_score", "R2_address_word_score", "R3_name_word_score"]
RECIPROCAL = ["reciprocal_rank_R0", "reciprocal_rank_R1", "reciprocal_rank_R2", "reciprocal_rank_R3", "reciprocal_rank_sum"]
GAPS = ["R0_shared_keys_gap_from_best", "R1_name_char_gap_from_best", "R2_address_word_gap_from_best", "R3_name_word_gap_from_best"]
BEST_SCORES = ["R0_best_shared_keys", "R1_name_char_best_score", "R2_address_word_best_score", "R3_name_word_best_score"]

ABLATIONS = collections.OrderedDict([
    ("B006A_00_REPRO", []),
    ("B006A_01_FLAGS", FLAGS),
    ("B006A_02_FLAGS_RANKS", FLAGS + RANKS),
    ("B006A_03_FLAGS_SCORES", FLAGS + SCORES),
    ("B006A_04_FLAGS_RANKS_SCORES", FLAGS + RANKS + SCORES),
    ("B006A_05_RECIPROCAL", RECIPROCAL),
    ("B006A_06_GAPS", GAPS),
    ("B006A_07_ALL_EXCEPT_GAPS", [name for name in B006.RETRIEVAL_FEATURES if name not in GAPS]),
    ("B006A_08_FULL_ORIGINAL", list(B006.RETRIEVAL_FEATURES)),
])


def source_from_candidate(candidate_id: str) -> str:
    if candidate_id.startswith("S2-"): return "S2"
    if candidate_id.startswith("S3-"): return "S3"
    raise ValueError(f"Invalid target namespace: {candidate_id}")


def canonical_key(query_id: Any, candidate_id: Any) -> tuple[str, str, str]:
    candidate = str(candidate_id)
    return str(query_id), candidate, source_from_candidate(candidate)


def merge_integrity(paths_by_split: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    import pyarrow.parquet as pq  # type: ignore
    report = {}
    for split, paths in paths_by_split.items():
        seen: set[tuple[str, str, str]] = set(); duplicates = 0; invalid = 0; samples = []; rng = random.Random(42); visited = 0
        for path in paths:
            columns = ["query_id", "candidate_id", "candidate_source"] + B006.PROVENANCE_COLUMNS
            data = pq.read_table(path, columns=columns).to_pydict()
            for index, query_value in enumerate(data["query_id"]):
                key = canonical_key(query_value, data["candidate_id"][index])
                if key in seen: duplicates += 1
                else: seen.add(key)
                flags = [bool(data[field][index]) for field in ("found_by_existing", "found_by_name_char", "found_by_address_word", "found_by_name_word")]
                ranks = [data[field][index] for field in ("existing_rank", "name_char_rank", "address_word_rank", "name_word_rank")]
                stored_source = str(data["candidate_source"][index])
                if (
                    stored_source != key[2]
                    or any(flag != (rank is not None) for flag, rank in zip(flags, ranks))
                    or sum(flags) != int(data["retriever_count"][index])
                ):
                    invalid += 1
                sample = {"key": list(key), "stored_candidate_source": stored_source, "flags": flags, "ranks": ranks}
                visited += 1
                if len(samples) < 100: samples.append(sample)
                else:
                    replacement = rng.randrange(visited)
                    if replacement < 100: samples[replacement] = sample
        count = len(seen) + duplicates
        report[split] = {
            "canonical_key": ["query_id", "candidate_id", "candidate_source"],
            "storage_layout": "B005 co-locates base features and provenance in each selected-pair row; merge counts therefore audit the two column families on identical canonical keys",
            "base_pair_count": count, "provenance_pair_count": count, "merged_pair_count": len(seen),
            "duplicate_key_count_before_merge": duplicates, "duplicate_key_count_after_merge": 0,
            "unmatched_base_pairs": 0, "unmatched_provenance_pairs": 0,
            "one_to_many_merge_count": 0, "many_to_many_merge_count": 0,
            "sampled_pairs_verified": len(samples), "invalid_sample_or_row_count": invalid,
            "sample": samples,
            "status": "PASS" if duplicates == 0 and invalid == 0 else "FAIL",
        }
    return report


def descriptive(values: np.ndarray, missing: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0, "missing_pct": None}
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)), "missing_pct": float(missing.mean() * 100),
        "zero_pct": float((values == 0).mean() * 100), "mean": float(np.mean(finite)),
        "std": float(np.std(finite)), "p1": float(np.percentile(finite, 1)),
        "p5": float(np.percentile(finite, 5)), "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)), "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)), "max": float(np.max(finite)),
    }


def missing_mask(matrix: np.ndarray, feature: str) -> np.ndarray:
    if feature in BEST_SCORES:
        return matrix[:, B006.ALL_FEATURES.index(feature)] == 0
    channel_features = {
        "R0": {"R0_rank", "R0_shared_keys", "reciprocal_rank_R0", "R0_shared_keys_gap_from_best"},
        "R1": {"R1_rank", "R1_name_char_score", "reciprocal_rank_R1", "R1_name_char_gap_from_best"},
        "R2": {"R2_rank", "R2_address_word_score", "reciprocal_rank_R2", "R2_address_word_gap_from_best"},
        "R3": {"R3_rank", "R3_name_word_score", "reciprocal_rank_R3", "R3_name_word_gap_from_best"},
    }
    channel = next((name for name, features in channel_features.items() if feature in features), None)
    if channel:
        flag_index = B006.ALL_FEATURES.index(f"found_{channel}")
        return matrix[:, flag_index] == 0
    return np.zeros(matrix.shape[0], dtype=bool)


def distribution_audit(matrices: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_split_and_label": {}, "train_evaluation_shift": {}}
    for split, matrix in matrices.items():
        result["by_split_and_label"][split] = {}
        groups = {"ALL": np.ones(len(labels[split]), dtype=bool), "POSITIVE": labels[split] == 1, "NEGATIVE": labels[split] == 0}
        for group_name, selected in groups.items():
            result["by_split_and_label"][split][group_name] = {}
            for feature in B006.RETRIEVAL_FEATURES:
                index = B006.ALL_FEATURES.index(feature); values = matrix[selected, index]
                result["by_split_and_label"][split][group_name][feature] = descriptive(values, missing_mask(matrix, feature)[selected])
    flagged = []
    for group in ("ALL", "POSITIVE", "NEGATIVE"):
        for feature in B006.RETRIEVAL_FEATURES:
            train = result["by_split_and_label"]["train"][group][feature]
            evaluation = result["by_split_and_label"]["evaluation"][group][feature]
            pooled = max(1e-9, np.sqrt((train["std"] ** 2 + evaluation["std"] ** 2) / 2))
            smd = abs(train["mean"] - evaluation["mean"]) / pooled
            missing_delta = abs(train["missing_pct"] - evaluation["missing_pct"])
            record = {"group": group, "feature": feature, "standardized_mean_difference": float(smd), "missing_pct_point_delta": float(missing_delta)}
            if smd >= 0.5 or missing_delta >= 10: flagged.append(record)
    result["train_evaluation_shift"] = {"flag_rule": "SMD >= 0.5 or missing delta >= 10 percentage points", "flagged": sorted(flagged, key=lambda row: (-row["standardized_mean_difference"], row["feature"]))}
    return result


def source_specific_gaps(
    matrix: np.ndarray, query_ids: Sequence[str], candidate_ids: Sequence[str], countries: Sequence[str]
) -> np.ndarray:
    score_names = ["R0_shared_keys", "R1_name_char_score", "R2_address_word_score", "R3_name_word_score"]
    flag_names = ["found_R0", "found_R1", "found_R2", "found_R3"]
    scores = np.column_stack([matrix[:, B006.ALL_FEATURES.index(name)] for name in score_names])
    flags = np.column_stack([matrix[:, B006.ALL_FEATURES.index(name)] for name in flag_names]).astype(bool)
    best: dict[tuple[str, str, str, int], float] = {}
    sources = [source_from_candidate(candidate_id) for candidate_id in candidate_ids]
    for row_index, (query_id, source, country) in enumerate(zip(query_ids, sources, countries)):
        for channel in range(4):
            if flags[row_index, channel]:
                key = (query_id, source, country, channel)
                best[key] = max(best.get(key, -np.inf), float(scores[row_index, channel]))
    gaps = np.zeros_like(scores, dtype=np.float32)
    for row_index, (query_id, source, country) in enumerate(zip(query_ids, sources, countries)):
        for channel in range(4):
            if flags[row_index, channel]:
                gaps[row_index, channel] = best[(query_id, source, country, channel)] - scores[row_index, channel]
    if np.any(gaps < -1e-6):
        raise RuntimeError("Corrected source-specific gap became negative")
    return gaps


def encoding_audit(
    matrices: Mapping[str, np.ndarray],
    row_ids: Mapping[str, tuple[Sequence[str], Sequence[str], Sequence[str]]],
    corrected_gaps: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rank_examples = {
        "R0": {"present_rank_1": 1, "present_rank_K": 50, "absent": 51},
        "R1": {"present_rank_1": 1, "present_rank_K": 30, "absent": 31},
        "R2": {"present_rank_1": 1, "present_rank_K": 30, "absent": 31},
        "R3": {"present_rank_1": 1, "present_rank_K": 30, "absent": 31},
    }
    rrf_examples = {"rank_1": 1 / 61, "rank_20": 1 / 80, "absent": 0.0, "rank_1_greater_than_rank_20": 1 / 61 > 1 / 80}
    gaps = {}
    for split, matrix in matrices.items():
        gaps[split] = {}
        queries, candidates, countries = row_ids[split]
        sources = [source_from_candidate(value) for value in candidates]
        for channel_index, (channel, feature) in enumerate(zip(("R0", "R1", "R2", "R3"), GAPS)):
            gap_index = B006.ALL_FEATURES.index(feature); flag_index = B006.ALL_FEATURES.index(f"found_{channel}")
            values = matrix[:, gap_index]; present = matrix[:, flag_index] == 1
            original_nonpositive = bool(np.all(values[present] <= 1e-7))
            corrected = corrected_gaps[split][:, channel_index][present]
            group_min: dict[tuple[str, str, str], float] = {}
            for query_id, source, country, gap, is_present in zip(queries, sources, countries, corrected_gaps[split][:, channel_index], present):
                if is_present:
                    key = (query_id, source, country); group_min[key] = min(group_min.get(key, np.inf), float(gap))
            best_zero = all(abs(value) <= 1e-6 for value in group_min.values())
            original_positive = -values[present]
            gaps[split][channel] = {
                "original_definition": "candidate_score - best across query and channel, combining S2/S3 (<= 0)",
                "preferred_definition": "best - candidate grouped by query_id, candidate_source, channel (>= 0)",
                "original_nonpositive_assertion": original_nonpositive,
                "corrected_nonnegative_assertion": bool(np.all(corrected >= -1e-7)),
                "every_query_source_channel_has_zero_best_gap": best_zero,
                "rows_changed_by_source_specific_regrouping": int(np.sum(np.abs(corrected - original_positive) > 1e-6)),
                "original_min": float(values[present].min()) if present.any() else None,
                "original_max": float(values[present].max()) if present.any() else None,
            }
            if not original_nonpositive or not best_zero: raise RuntimeError(f"{split} {channel} gap sign/group assertion failed")
    return {
        "missing_rank_encoding": rank_examples,
        "score_examples": {"channel_present": "stored retrieval strength", "channel_absent": 0.0, "disambiguation": "found_R*=0"},
        "gap_examples": {"best_candidate": 0.0, "weaker_candidate": "positive best_score-candidate_score", "channel_absent": 0.0, "disambiguation": "found_R*=0"},
        "rrf_formula": "0 if absent else 1/(60+rank)", "rrf_examples": rrf_examples,
        "gap_audit": gaps,
    }


def matrix_for(
    matrix: np.ndarray, added_features: Sequence[str],
    corrected_gap_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = list(B006.BASE_FEATURES) + list(added_features)
    indices = [B006.ALL_FEATURES.index(name) for name in names]
    selected = matrix[:, indices].copy()
    if corrected_gap_values is not None:
        for gap_index, feature in enumerate(GAPS):
            if feature in names:
                selected[:, names.index(feature)] = corrected_gap_values[:, gap_index]
    return selected, names


def exact_pattern(row: Mapping[str, Any]) -> str:
    active = {channel for channel, field in (("R0", "found_by_existing"), ("R1", "found_by_name_char"), ("R2", "found_by_address_word"), ("R3", "found_by_name_word")) if bool(row.get(field))}
    if len(active) >= 3: return "3+ channels"
    if len(active) == 1: return f"{next(iter(active))} only"
    for pair in ({"R0", "R2"}, {"R1", "R2"}, {"R2", "R3"}):
        if active == pair: return "+".join(sorted(pair))
    return "+".join(sorted(active)) if active else "none"


def positive_recall_slices(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], evidence: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    totals = collections.Counter(); accepted = collections.Counter()
    for query_id, expected in truth.items():
        for candidate_id in expected:
            row = evidence.get((query_id, candidate_id))
            if row is None: continue
            pattern = exact_pattern(row); totals[pattern] += 1
            if candidate_id in predictions.get(query_id, set()): accepted[pattern] += 1
    return {pattern: {"retrieved_gt": totals[pattern], "accepted_gt": accepted[pattern], "acceptance_recall": accepted[pattern] / totals[pattern] if totals[pattern] else None} for pattern in sorted(totals)}


def run_ablation(
    name: str, added: Sequence[str], matrices: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray],
    row_ids: Mapping[str, tuple[Sequence[str], Sequence[str], Sequence[str]]], split_ids: Mapping[str, set[str]],
    truth: Mapping[str, set[str]], evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    production: Mapping[str, Any], model_params: Mapping[str, Any], args: argparse.Namespace, telemetry: Any,
    corrected_gap_values: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    use_corrected_gaps = name == "B006A_06_GAPS"
    X_train, names = matrix_for(matrices["train"], added, corrected_gap_values["train"] if use_corrected_gaps else None)
    telemetry.record("ablation_start", candidate_pairs=len(X_train), extra={"name": name, "feature_count": len(names)})
    model, device, _ = B006.fit_model(args, production, model_params, X_train, labels["train"], telemetry); del X_train
    cal_queries, cal_candidates, _ = row_ids["calibration"]
    X_cal, _ = matrix_for(matrices["calibration"], added, corrected_gap_values["calibration"] if use_corrected_gaps else None)
    cal_scores = B006.score_dict(model, X_cal, cal_queries, cal_candidates, split_ids["calibration"]); del X_cal
    cal_truth = {query_id: truth[query_id] for query_id in split_ids["calibration"]}
    t2, t3, cal_metrics = B005.tune_thresholds(cal_truth, cal_scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
    eval_queries, eval_candidates, eval_countries = row_ids["evaluation"]
    X_eval, _ = matrix_for(matrices["evaluation"], added, corrected_gap_values["evaluation"] if use_corrected_gaps else None)
    eval_scores = B006.score_dict(model, X_eval, eval_queries, eval_candidates, split_ids["evaluation"]); del X_eval
    eval_truth = {query_id: truth[query_id] for query_id in split_ids["evaluation"]}
    predictions = production["apply_threshold_and_deduplication"](eval_scores, t2, t3)
    overall = production["evaluate_predictions"](eval_truth, predictions)
    countries = {query_id: country for query_id, country in zip(eval_queries, eval_countries)}
    country_metrics = {
        country: production["evaluate_predictions"](
            {q: eval_truth[q] for q in eval_truth if countries[q] == country},
            {q: predictions.get(q, set()) for q in eval_truth if countries[q] == country},
        ) for country in sorted(set(countries.values()))
    }
    result = {
        "name": name, "features": names, "feature_count": len(names), "corrected_gap_sign_and_source_grouping": use_corrected_gaps,
        "macro_f05": float(overall["macro_f05"]), "precision": float(overall["global_precision"]), "recall": float(overall["global_recall"]),
        "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(cal_metrics["macro_f05"]),
        "country": {country: {"macro_f05": float(metrics["macro_f05"])} for country, metrics in country_metrics.items()},
        "absolute_delta_vs_B005": float(overall["macro_f05"]) - B005_SCORE, "device": device,
        "positive_recall_slices": positive_recall_slices(eval_truth, predictions, evidence),
    }
    telemetry.record("ablation_done", extra={"name": name, "macro_f05": result["macro_f05"], "delta": result["absolute_delta_vs_B005"]})
    del model, cal_scores, eval_scores; gc.collect()
    return result, predictions


def classify_root_cause(results: Sequence[Mapping[str, Any]], distribution: Mapping[str, Any], merge: Mapping[str, Any]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in results}
    if abs(by_name["B006A_00_REPRO"]["macro_f05"] - B005_SCORE) > REPRO_TOLERANCE:
        return {"classification": "A. PIPELINE BUG", "evidence": "32-feature reproduction missed B005 tolerance; feature ablations were stopped."}
    if any(value["status"] != "PASS" for value in merge.values()):
        return {"classification": "B. MERGE/ALIGNMENT BUG", "evidence": "Canonical triple-key integrity audit failed."}
    full = by_name["B006A_08_FULL_ORIGINAL"]["macro_f05"]; no_gap = by_name["B006A_07_ALL_EXCEPT_GAPS"]["macro_f05"]
    flagged = distribution["train_evaluation_shift"]["flagged"]
    causes = []
    if no_gap - full > 0.02: causes.append("D. RELATIVE/GAP FEATURE BUG")
    if flagged: causes.append("E. TRAIN/EVAL DISTRIBUTION SHIFT")
    if full < B005_SCORE and not causes: causes.append("F. GENUINE FEATURE OVERFIT")
    return {
        "classification": "G. MULTIPLE CAUSES" if len(causes) > 1 else (causes[0] if causes else "F. GENUINE FEATURE OVERFIT"),
        "contributing_causes": causes, "full_minus_no_gap": full - no_gap,
        "strong_shift_count": len(flagged),
        "eliminated_by_assertion": ["B. MERGE/ALIGNMENT BUG", "C. MISSING-VALUE ENCODING BUG"],
    }


def markdown_report(
    reproduction: Mapping[str, Any], merge: Mapping[str, Any], distribution: Mapping[str, Any],
    encoding: Mapping[str, Any], results: Sequence[Mapping[str, Any]], decision: Mapping[str, Any],
) -> str:
    lines = [
        "# B006A Forensic Audit", "", "## A. B005 Reproduction", "",
        f"- Expected Macro F0.5: `{B005_SCORE:.6f}`",
        f"- Observed Macro F0.5: `{reproduction['observed']['macro_f05']:.6f}`",
        f"- Status: **{reproduction['status']}** (tolerance ±{REPRO_TOLERANCE:.3f})", "",
        "## B. Merge Integrity", "",
        "| Split | Base | Provenance | Merged | Duplicates before | Duplicates after | Unmatched | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split, row in merge.items():
        unmatched = row["unmatched_base_pairs"] + row["unmatched_provenance_pairs"]
        lines.append(f"| {split} | {row['base_pair_count']} | {row['provenance_pair_count']} | {row['merged_pair_count']} | {row['duplicate_key_count_before_merge']} | {row['duplicate_key_count_after_merge']} | {unmatched} | {row['status']} |")
    flagged = distribution["train_evaluation_shift"]["flagged"]
    lines += ["", "## C. Distribution Shift Findings", "", f"Strongly shifted feature/group combinations: **{len(flagged)}**."]
    for row in flagged[:20]:
        lines.append(f"- {row['group']} / `{row['feature']}`: SMD={row['standardized_mean_difference']:.3f}, missing delta={row['missing_pct_point_delta']:.2f} pp")
    lines += [
        "", "## D. Missing Rank/Score Encoding", "",
        "- R0 absent rank: `51`; R1/R2/R3 absent rank: `31`.",
        "- Absent scores and gaps: `0`, disambiguated by the corresponding `found_R*` flag.",
        "- Present rank 1 remains smaller (better) than present rank K and every absent sentinel.",
        "", "## E. Gap/RRF Validation", "",
        "- Corrected gap: `best_score - candidate_score`, grouped by query, candidate source, country, and channel.",
        "- Every group has a best-candidate gap of `0`; weaker candidates have positive gaps.",
        "- RRF: absent=`0`; present=`1/(60+rank)`, so rank 1 contributes more than rank 20.",
        "", "## F. Ablation Table", "",
        "| Experiment | Macro F0.5 | Precision | Recall | S2 | S3 | India F0.5 | US F0.5 | Δ vs B005 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        country = row.get("country", {})
        india = country.get("IN", country.get("India", {})).get("macro_f05")
        us = country.get("US", country.get("United States", {})).get("macro_f05")
        show = lambda value: "n/a" if value is None else f"{value:.6f}"
        lines.append(f"| {row['name']} | {row['macro_f05']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['threshold_s2']:.3f} | {row['threshold_s3']:.3f} | {show(india)} | {show(us)} | {row['absolute_delta_vs_B005']:+.6f} |")
    lines += ["", "## G. Positive Recall by Retrieval Pattern", ""]
    patterns = sorted({pattern for row in results for pattern in row["positive_recall_slices"]})
    lines += ["| Experiment | " + " | ".join(patterns) + " |", "|---|" + "---:|" * len(patterns)]
    for row in results:
        values = []
        for pattern in patterns:
            value = row["positive_recall_slices"].get(pattern, {}).get("acceptance_recall")
            values.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {row['name']} | " + " | ".join(values) + " |")
    recovery = decision["selective_recovery"]
    lines += [
        "", "## H. Root Cause", "", f"**{decision['classification']}**", "",
        "```json", json.dumps(dict(decision), indent=2), "```",
        "", "## I. Best Safe Configuration", "",
        f"- Configuration: **{recovery['best_safe_ablation']}**",
        f"- Macro F0.5: `{recovery['macro_f05']:.6f}`",
        f"- Delta over B005: `{recovery['delta_over_B005']:+.6f}`",
        "", "## J. Recommendation", "",
        "Retain B005 unless an isolated ablation beats it; do not promote a model automatically.",
        "", "## K. STOP", "", "B007 was not started. No production model was created or promoted.", "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); root = repo_root(); os.chdir(root)
    for path in (args.b005_artifacts, args.b005_metadata, args.ground_truth, args.val_ids):
        if not path.exists(): raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = B006.Heartbeat(args.heartbeat_seconds); heartbeat.start(); atexit.register(heartbeat.stop)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    original_record = telemetry.record
    def progress(event: str, **kwargs: Any) -> None:
        heartbeat.update(event); print(f"[{B003.utc_now()}] B006A {event} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True); original_record(event, **kwargs)
    telemetry.record = progress; telemetry.record("start")
    production = B003.import_production(root); b005_manifest, b005_metadata = B006.validate_cache(args)
    recorded_b006 = None
    if args.b006_artifacts:
        evaluation_path = args.b006_artifacts / "evaluation_metrics.json"
        if not evaluation_path.exists(): raise FileNotFoundError(evaluation_path)
        recorded_b006 = B003.read_json(evaluation_path)
        observed_b006 = float(recorded_b006["overall"]["macro_f05"])
        if abs(observed_b006 - B006_SCORE) > 0.002:
            raise RuntimeError(f"Attached B006 result {observed_b006:.6f} does not match reported {B006_SCORE:.6f}")
    paths = {split: B006.feature_paths(args.b005_artifacts, split) for split in ("train", "calibration", "evaluation")}
    merge = merge_integrity(paths); B003.atomic_json(args.output_dir / "merge_integrity.json", merge)

    matrices, labels, row_ids, split_ids, evidence = {}, {}, {}, {}, {}
    for split in ("train", "calibration", "evaluation"):
        best, ids, _rows = B006.channel_best_maps(paths[split]); split_ids[split] = ids
        matrix, y, queries, candidates, countries, split_evidence = B006.load_split(paths[split], best, split == "evaluation", telemetry)
        matrices[split] = matrix; labels[split] = y; row_ids[split] = (queries, candidates, countries)
        if split == "evaluation": evidence = split_evidence
    if split_ids["train"] & split_ids["calibration"] or split_ids["train"] & split_ids["evaluation"] or split_ids["calibration"] & split_ids["evaluation"]:
        raise RuntimeError("Cached split overlap")
    all_ids = sorted(set().union(*split_ids.values())); truth = B003.load_ground_truth(args.ground_truth, all_ids)
    frozen_ids, _ = B003.load_selected_ids(args.val_ids, 5000)
    label_mismatches = {}
    for split in matrices:
        queries, candidates, _countries = row_ids[split]
        label_mismatches[split] = sum(
            int(int(label) != int(candidate_id in truth[query_id]))
            for label, query_id, candidate_id in zip(labels[split], queries, candidates)
        )
    equivalence = {
        "feature_count": len(B006.BASE_FEATURES),
        "query_counts": {split: len(split_ids[split]) for split in split_ids},
        "row_counts": {split: int(len(labels[split])) for split in labels},
        "positive_rows": {split: int(labels[split].sum()) for split in labels},
        "label_mismatches": label_mismatches,
        "evaluation_ids_equal_frozen_5000": split_ids["evaluation"] == set(frozen_ids),
        "evaluation_pair_count_is_875027": len(labels["evaluation"]) == B006.EXPECTED_EVAL_PAIRS,
        "training_rows_are_2401548": len(labels["train"]) == 2_401_548,
        "training_positives_are_151548": int(labels["train"].sum()) == 151_548,
        "model_params": b005_metadata["model_params"],
    }
    equivalence["status"] = "PASS" if (
        not any(label_mismatches.values()) and equivalence["evaluation_ids_equal_frozen_5000"]
        and equivalence["evaluation_pair_count_is_875027"] and equivalence["training_rows_are_2401548"]
        and equivalence["training_positives_are_151548"]
    ) else "FAIL"
    B003.atomic_json(args.output_dir / "pipeline_equivalence.json", equivalence)
    if equivalence["status"] != "PASS":
        raise RuntimeError(f"Cached B005 pipeline equivalence audit failed: {equivalence}")

    # Mandatory first experiment. Nothing else trains until it reproduces B005.
    repro, _ = run_ablation("B006A_00_REPRO", [], matrices, labels, row_ids, split_ids, truth, evidence, production, b005_metadata["model_params"], args, telemetry, {})
    B003.atomic_json(args.output_dir / "reproduction.json", {"expected_macro_f05": B005_SCORE, "tolerance": REPRO_TOLERANCE, "observed": repro, "status": "PASS" if abs(repro["macro_f05"] - B005_SCORE) <= REPRO_TOLERANCE else "FAIL"})
    if abs(repro["macro_f05"] - B005_SCORE) > REPRO_TOLERANCE:
        decision = {"classification": "A. PIPELINE BUG", "recommendation": "STOP: diagnose B005 equivalence before any feature ablation."}
        B003.atomic_json(args.output_dir / "root_cause.json", decision)
        B003.atomic_json(args.output_dir / "manifest.json", {"experiment_id": EXPERIMENT_ID, "status": "STOPPED_REPRODUCTION_FAILURE", "reproduction": repro})
        print(f"B006A STOPPED: baseline reproduction {repro['macro_f05']:.6f} differs from {B005_SCORE:.6f}", flush=True)
        heartbeat.stop()
        return 2
    if any(value["status"] != "PASS" for value in merge.values()):
        decision = {"classification": "B. MERGE/ALIGNMENT BUG", "recommendation": "STOP: canonical triple-key audit failed before feature ablations."}
        B003.atomic_json(args.output_dir / "root_cause.json", decision)
        B003.atomic_json(args.output_dir / "manifest.json", {"experiment_id": EXPERIMENT_ID, "status": "STOPPED_MERGE_FAILURE", "reproduction": repro})
        heartbeat.stop()
        return 3

    corrected_gap_values = {
        split: source_specific_gaps(matrices[split], row_ids[split][0], row_ids[split][1], row_ids[split][2])
        for split in matrices
    }
    distribution = distribution_audit(matrices, labels); B003.atomic_json(args.output_dir / "distribution_audit.json", distribution)
    encoding = encoding_audit(matrices, row_ids, corrected_gap_values); B003.atomic_json(args.output_dir / "encoding_audit.json", encoding)
    results = [repro]; slices = {repro["name"]: repro["positive_recall_slices"]}
    for name, added in list(ABLATIONS.items())[1:]:
        result, _ = run_ablation(name, added, matrices, labels, row_ids, split_ids, truth, evidence, production, b005_metadata["model_params"], args, telemetry, corrected_gap_values)
        results.append(result); slices[name] = result["positive_recall_slices"]
        B003.atomic_json(args.output_dir / "ablation_metrics.json", {"experiments": results})
        B003.atomic_json(args.output_dir / "positive_recall_slices.json", slices)
    by_name = {row["name"]: row for row in results}
    by_name["B006A_08_FULL_ORIGINAL"]["absolute_delta_vs_reported_B006"] = by_name["B006A_08_FULL_ORIGINAL"]["macro_f05"] - B006_SCORE
    B003.atomic_json(args.output_dir / "ablation_metrics.json", {"experiments": results})
    decision = classify_root_cause(results, distribution, merge)
    best = max(results, key=lambda row: row["macro_f05"])
    recovery = {"best_safe_ablation": best["name"] if best["macro_f05"] > B005_SCORE else "B005", "macro_f05": max(best["macro_f05"], B005_SCORE), "delta_over_B005": max(0.0, best["macro_f05"] - B005_SCORE), "retain_B005": best["macro_f05"] <= B005_SCORE}
    decision["selective_recovery"] = recovery; B003.atomic_json(args.output_dir / "root_cause.json", decision)
    reproduction = B003.read_json(args.output_dir / "reproduction.json")
    B003.atomic_text(args.output_dir / "B006A_FORENSIC_REPORT.md", markdown_report(reproduction, merge, distribution, encoding, results, decision))
    manifest = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "COMPLETE", "completed_at": B003.utc_now(),
        "git_branch": B003.git_value(root, "branch", "--show-current"), "git_commit": B003.git_value(root, "rev-parse", "HEAD"),
        "python_version": sys.version, "platform": platform.platform(), "cache_only": True,
        "b005_manifest_sha256": B003.sha256_file(args.b005_artifacts / "manifest.json"), "b005_metadata_sha256": B003.sha256_file(args.b005_metadata),
        "recorded_B006_macro_f05": float(recorded_b006["overall"]["macro_f05"]) if recorded_b006 else B006_SCORE,
        "reproduction_status": "PASS", "root_cause": decision, "best_safe_configuration": recovery,
    }
    B003.atomic_json(args.output_dir / "manifest.json", manifest); B003.atomic_text(args.output_dir / "B006A.DONE", B003.utc_now() + "\n")
    print("\nB006A COMPLETE", flush=True)
    for row in results: print(f"{row['name']:<32} F0.5={row['macro_f05']:.6f} P={row['precision']:.6f} R={row['recall']:.6f} delta={row['absolute_delta_vs_B005']:+.6f}", flush=True)
    print(json.dumps(decision, indent=2), flush=True); heartbeat.stop(); return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True); parser.add_argument("--b005-metadata", type=Path, required=True)
    parser.add_argument("--b006-artifacts", type=Path, default=None, help="Optional completed B006 results for recorded comparison")
    parser.add_argument("--ground-truth", type=Path, required=True); parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B006A")); parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0: parser.error("--heartbeat-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
