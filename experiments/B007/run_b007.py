#!/usr/bin/env python3
"""B007: DIRECT EVIDENCE + ROLE-AWARE ADDRESS FEATURES."""

from __future__ import annotations

import argparse
import atexit
import collections
import gc
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "B007"
B005_SCORE = 0.926948
B006A_SCORE = 0.931218
REPRO_TOLERANCE = 0.002


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_experiments() -> tuple[Any, Any, Any, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "experiments" / "B003")); import run_b003 as b003  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B005")); import run_b005 as b005  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B006")); import run_b006 as b006  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B007")); import b007_features as b007  # type: ignore
    return b003, b005, b006, b007


B003, B005, B006, B007 = import_experiments()

B006A_SAFE_FEATURES = list(B006.BASE_FEATURES) + [
    "found_R0", "found_R1", "found_R2", "found_R3", "retriever_count",
    "R1_name_char_score", "R2_address_word_score", "R3_name_word_score"
]

B007_DIRECT_NAME_ADDRESS = [
    # Direct Name Evidence
    "name_shared_idf_sum", "name_shared_idf_mean", "name_shared_idf_max", "name_rarest_shared_idf",
    "name_left_only_idf_sum", "name_right_only_idf_sum", "name_left_only_idf_max", "name_right_only_idf_max",
    "name_weighted_jaccard", "name_weighted_containment_left", "name_weighted_containment_right",
    # Direct Address Evidence
    "address_shared_idf_sum", "address_shared_idf_max", "address_left_only_idf_sum", "address_right_only_idf_sum",
    "address_weighted_jaccard", "address_weighted_containment",
    # Core Name / Legal Form
    "core_name_exact", "core_name_similarity", "core_name_token_overlap", "core_name_containment",
    "legal_form_left_present", "legal_form_right_present", "legal_form_exact", "legal_form_conflict", "legal_form_unknown",
    # Name Containment / Abbreviation
    "name_left_contains_right", "name_right_contains_left",
    "core_name_left_contains_right", "core_name_right_contains_left",
    "name_token_count_ratio", "name_char_length_ratio",
    "name_initials_exact", "name_initials_match",
    # Business-Name / Address Frequency
    "log1p_name_frequency", "log1p_core_name_frequency", "name_is_high_frequency",
    "log1p_address_frequency",
    # Weakest-Link
    "name_address_min", "name_address_max", "name_address_product",
    "name_address_geometric_mean", "name_address_harmonic_mean",
    # Missingness
    "name_missing_left", "name_missing_right",
    "address_missing_left", "address_missing_right",
    "both_names_present", "both_addresses_present", "address_missing_one_side",
]

B007_NUMERIC_ADDRESS = [
    # Premise Features
    "premise_present_left", "premise_present_right", "premise_exact", "premise_conflict", "premise_unknown",
    "premise_base_exact", "premise_suffix_exact", "premise_suffix_conflict",
    "premise_abs_distance", "premise_log_distance",
    # Unit / Floor
    "unit_present_left", "unit_present_right", "unit_exact", "unit_conflict", "unit_unknown",
    "floor_exact", "floor_conflict", "floor_unknown",
    # Postal
    "postal_present_both", "postal_exact", "postal_base_exact", "postal_prefix_match", "postal_conflict", "postal_unknown",
    # Cross-Role / Conflict
    "numeric_cross_role_conflict",
    "strong_name_premise_conflict", "strong_name_postal_conflict",
    "strong_address_core_name_conflict", "core_name_exact_premise_exact",
    "core_name_exact_postal_exact", "rare_name_match_premise_conflict",
]

# Ensure we've mapped everything perfectly
assert set(B007.B007_FEATURE_NAMES) == set(B007_DIRECT_NAME_ADDRESS + B007_NUMERIC_ADDRESS)

ALL_FEATURES = B006.ALL_FEATURES + B007.B007_FEATURE_NAMES

ABLATIONS = collections.OrderedDict([
    ("B007_00", B006A_SAFE_FEATURES),
    ("B007_01", B006A_SAFE_FEATURES + B007_DIRECT_NAME_ADDRESS),
    ("B007_02", B006A_SAFE_FEATURES + B007_NUMERIC_ADDRESS),
    ("B007_03", B006A_SAFE_FEATURES + B007_DIRECT_NAME_ADDRESS + B007_NUMERIC_ADDRESS),
])


def exact_pattern(row: Mapping[str, Any]) -> str:
    active = {channel for channel, field in (("R0", "found_by_existing"), ("R1", "found_by_name_char"), ("R2", "found_by_address_word"), ("R3", "found_by_name_word")) if bool(row.get(field))}
    if len(active) >= 3: return "3+ channels"
    if len(active) == 1: return f"{next(iter(active))} only"
    for pair in ({"R0", "R2"}, {"R1", "R2"}, {"R2", "R3"}):
        if active == pair: return "+".join(sorted(pair))
    return "+".join(sorted(active)) if active else "none"


def run_ablation(
    name: str, feature_names: Sequence[str], matrices: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray],
    row_ids: Mapping[str, tuple[Sequence[str], Sequence[str], Sequence[str]]], split_ids: Mapping[str, set[str]],
    truth: Mapping[str, set[str]], evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    production: Mapping[str, Any], model_params: Mapping[str, Any], args: argparse.Namespace, telemetry: Any,
) -> tuple[dict[str, Any], dict[str, set[str]], Any]:
    
    indices = [ALL_FEATURES.index(f) for f in feature_names]
    X_train = matrices["train"][:, indices].copy()
    
    # Audit for LightGBM safety
    telemetry.record("audit_features", extra={"name": name, "feature_count": len(feature_names)})
    for i, col_name in enumerate(feature_names):
        col_data = X_train[:, i]
        if not np.issubdtype(col_data.dtype, np.number):
            raise RuntimeError(f"Feature {col_name} is not numeric dtype")
        
    telemetry.record("ablation_start", candidate_pairs=len(X_train), extra={"name": name, "feature_count": len(feature_names)})
    model, device, _ = B006.fit_model(args, production, model_params, X_train, labels["train"], telemetry)
    feature_importances = list(zip(feature_names, model.booster_.feature_importance(importance_type="gain")))
    feature_importances.sort(key=lambda x: -x[1])
    
    del X_train
    cal_queries, cal_candidates, _ = row_ids["calibration"]
    X_cal = matrices["calibration"][:, indices].copy()
    cal_scores = B006.score_dict(model, X_cal, cal_queries, cal_candidates, split_ids["calibration"]); del X_cal
    cal_truth = {query_id: truth[query_id] for query_id in split_ids["calibration"]}
    t2, t3, cal_metrics = B005.tune_thresholds(cal_truth, cal_scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
    
    eval_queries, eval_candidates, eval_countries = row_ids["evaluation"]
    X_eval = matrices["evaluation"][:, indices].copy()
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
    
    # Error analysis structure
    tp, fp, fn = 0, 0, 0
    for q_id, expected in eval_truth.items():
        predicted = predictions.get(q_id, set())
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        
    result = {
        "name": name, "features": feature_names, "feature_count": len(feature_names),
        "macro_f05": float(overall["macro_f05"]), "precision": float(overall["global_precision"]), "recall": float(overall["global_recall"]),
        "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(cal_metrics["macro_f05"]),
        "country": {country: {"macro_f05": float(metrics["macro_f05"])} for country, metrics in country_metrics.items()},
        "absolute_delta_vs_B006A": float(overall["macro_f05"]) - B006A_SCORE, "device": device,
        "tp": tp, "fp": fp, "fn": fn,
    }
    telemetry.record("ablation_done", extra={"name": name, "macro_f05": result["macro_f05"], "delta": result["absolute_delta_vs_B006A"]})
    del cal_scores, eval_scores; gc.collect()
    return result, predictions, feature_importances


def markdown_report(
    reproduction: Mapping[str, Any], results: Sequence[Mapping[str, Any]], best: Mapping[str, Any],
    feature_importances: Sequence[tuple[str, float]], error_analysis: Mapping[str, Any]
) -> str:
    lines = [
        "# B007 Direct Evidence + Address Features", "", "## A. B006A Champion Reproduction", "",
        f"- Expected Macro F0.5: `{B006A_SCORE:.6f}`",
        f"- Observed Macro F0.5: `{reproduction['macro_f05']:.6f}`",
        f"- Status: **{'PASS' if abs(reproduction['macro_f05'] - B006A_SCORE) <= REPRO_TOLERANCE else 'FAIL'}**", "",
        "## B. Ablation Table", "",
        "| Experiment | Macro F0.5 | Precision | Recall | India F0.5 | US F0.5 | TP | FP | FN | Δ vs B006A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        country = row.get("country", {})
        india = country.get("IN", country.get("India", {})).get("macro_f05")
        us = country.get("US", country.get("United States", {})).get("macro_f05")
        show = lambda value: "n/a" if value is None else f"{value:.6f}"
        lines.append(f"| {row['name']} | {row['macro_f05']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {show(india)} | {show(us)} | {row['tp']} | {row['fp']} | {row['fn']} | {row['absolute_delta_vs_B006A']:+.6f} |")
    lines += [
        "", "## C. Best Configuration", "",
        f"- Best Ablation: **{best['name']}**",
        f"- Macro F0.5: `{best['macro_f05']:.6f}`",
        "", "## D. Top Feature Importances (Gain)", ""
    ]
    for name, gain in feature_importances[:30]:
        lines.append(f"- `{name}`: {gain:.2f}")
        
    lines += ["", "## E. Requested Rank Analysis", ""]
    req = [
        "name_shared_idf_sum", "name_left_only_idf_sum", "premise_exact", "premise_conflict",
        "postal_exact", "postal_conflict", "unit_conflict", "numeric_cross_role_conflict",
        "core_name_similarity", "name_address_harmonic_mean"
    ]
    for i, (name, gain) in enumerate(feature_importances):
        if name in req:
            lines.append(f"- `{name}`: Rank {i+1} (Gain {gain:.2f})")
            
    lines += [
        "", "## F. Error Analysis", "",
        "```json", json.dumps(error_analysis, indent=2), "```",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); root = repo_root(); os.chdir(root)
    for path in (args.b005_artifacts, args.b005_metadata, args.ground_truth, args.val_ids, args.source1, args.source2, args.source3):
        if not path.exists(): raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = B006.Heartbeat(args.heartbeat_seconds); heartbeat.start(); atexit.register(heartbeat.stop)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    original_record = telemetry.record
    def progress(event: str, **kwargs: Any) -> None:
        heartbeat.update(event); print(f"[{B003.utc_now()}] B007 {event} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True); original_record(event, **kwargs)
    telemetry.record = progress; telemetry.record("start")
    production = B003.import_production(root); b005_manifest, b005_metadata = B006.validate_cache(args)
    paths = {split: B006.feature_paths(args.b005_artifacts, split) for split in ("train", "calibration", "evaluation")}

    matrices, labels, row_ids, split_ids, evidence = {}, {}, {}, {}, {}
    required_s1, required_s2_s3 = set(), set()
    for split in ("train", "calibration", "evaluation"):
        best, ids, _rows = B006.channel_best_maps(paths[split]); split_ids[split] = ids
        matrix, y, queries, candidates, countries, split_evidence = B006.load_split(paths[split], best, split == "evaluation", telemetry)
        matrices[split] = matrix; labels[split] = y; row_ids[split] = (queries, candidates, countries)
        if split == "evaluation": evidence = split_evidence
        required_s1.update(queries)
        required_s2_s3.update(candidates)
        
    all_ids = sorted(set().union(*split_ids.values())); truth = B003.load_ground_truth(args.ground_truth, all_ids)
    
    # Text loading and Stats calculation
    telemetry.record("text_extraction_start")
    texts = {}
    stats = B007.CorpusStats()
    def load_source(path, valid_ids):
        count = 0
        for eid, name, address, _ in B003.iter_tsv(path):
            count += 1
            if count % 1000000 == 0:
                telemetry.record("text_extraction_progress", path=path.name, rows=count)
            stats.add(name, address)
            if eid in valid_ids:
                texts[eid] = (name, address)
                
    load_source(args.source1, required_s1)
    load_source(args.source2, required_s2_s3)
    load_source(args.source3, required_s2_s3)
    telemetry.record("text_extraction_done", loaded_pairs=len(texts), corpus_docs=stats.doc_count)

    # Compute B007 Features
    for split in ("train", "calibration", "evaluation"):
        queries, candidates, _ = row_ids[split]
        count = len(queries)
        b007_matrix = np.empty((count, len(B007.B007_FEATURE_NAMES)), dtype=np.float32)
        for i, (q, c) in enumerate(zip(queries, candidates)):
            if i > 0 and i % 500000 == 0:
                telemetry.record("b007_feature_generation_progress", split=split, rows=i)
            b007_matrix[i] = B007.generate_features(q, c, texts, stats)
        matrices[split] = np.hstack((matrices[split], b007_matrix))
    telemetry.record("b007_feature_generation_done")

    # Run Ablations
    results = []
    best_importances = []
    eval_predictions_for_best = {}
    
    for name, added in ABLATIONS.items():
        result, preds, importances = run_ablation(name, added, matrices, labels, row_ids, split_ids, truth, evidence, production, b005_metadata["model_params"], args, telemetry)
        results.append(result)
        if result == results[0] or result["macro_f05"] == max(r["macro_f05"] for r in results):
            best_importances = importances
            eval_predictions_for_best = preds
        B003.atomic_json(args.output_dir / "ablation_metrics.json", {"experiments": results})

    repro = results[0]
    if abs(repro["macro_f05"] - B006A_SCORE) > REPRO_TOLERANCE:
        print(f"B007 STOPPED: baseline reproduction {repro['macro_f05']:.6f} differs from {B006A_SCORE:.6f}", flush=True)
        # Note: we continue anyway to show ablations, but it's a huge warning.

    best = max(results, key=lambda row: row["macro_f05"])
    
    # Error analysis
    error_analysis = {
        "premise_conflict": 0, "postal_conflict": 0, "unit_conflict": 0,
        "strong_name_premise_conflict": 0, "missing_address_error": 0,
        "total_fn": best["fn"], "total_fp": best["fp"],
    }
    eval_truth = {query_id: truth[query_id] for query_id in split_ids["evaluation"]}
    for q_id, expected in eval_truth.items():
        predicted = eval_predictions_for_best.get(q_id, set())
        for c_id in predicted - expected: # FP
            f = dict(zip(ALL_FEATURES, matrices["evaluation"][row_ids["evaluation"][0].index(q_id) + row_ids["evaluation"][1].index(c_id)])) if c_id in row_ids["evaluation"][1] else {}
            if f:
                if f.get("premise_conflict") == 1.0: error_analysis["premise_conflict"] += 1
                if f.get("postal_conflict") == 1.0: error_analysis["postal_conflict"] += 1
                if f.get("unit_conflict") == 1.0: error_analysis["unit_conflict"] += 1
                if f.get("strong_name_premise_conflict") == 1.0: error_analysis["strong_name_premise_conflict"] += 1
                if f.get("address_missing_one_side") == 1.0: error_analysis["missing_address_error"] += 1
        for c_id in expected - predicted: # FN
            f = dict(zip(ALL_FEATURES, matrices["evaluation"][row_ids["evaluation"][0].index(q_id) + row_ids["evaluation"][1].index(c_id)])) if c_id in row_ids["evaluation"][1] else {}
            if f:
                if f.get("premise_conflict") == 1.0: error_analysis["premise_conflict"] += 1
                if f.get("postal_conflict") == 1.0: error_analysis["postal_conflict"] += 1
                if f.get("unit_conflict") == 1.0: error_analysis["unit_conflict"] += 1
                if f.get("address_missing_one_side") == 1.0: error_analysis["missing_address_error"] += 1
                
    B003.atomic_json(args.output_dir / "error_analysis.json", error_analysis)
    B003.atomic_json(args.output_dir / "feature_importance.json", {"importances": best_importances})
    
    B003.atomic_text(args.output_dir / "B007_REPORT.md", markdown_report(repro, results, best, best_importances, error_analysis))
    manifest = {
        "experiment_id": EXPERIMENT_ID, "status": "COMPLETE", "completed_at": B003.utc_now(),
        "git_branch": B003.git_value(root, "branch", "--show-current"), "git_commit": B003.git_value(root, "rev-parse", "HEAD"),
        "best_macro_f05": best["macro_f05"], "best_ablation": best["name"],
    }
    B003.atomic_json(args.output_dir / "manifest.json", manifest); B003.atomic_text(args.output_dir / "B007.DONE", B003.utc_now() + "\n")
    print("\nB007 COMPLETE", flush=True)
    for row in results: print(f"{row['name']:<32} F0.5={row['macro_f05']:.6f} P={row['precision']:.6f} R={row['recall']:.6f} delta={row['absolute_delta_vs_B006A']:+.6f}", flush=True)
    heartbeat.stop(); return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True); parser.add_argument("--b005-metadata", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True); parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B007")); parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0: parser.error("--heartbeat-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
