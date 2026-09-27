#!/usr/bin/env python3
"""GPU2 final cache-only challenger: C1 decision rules, C2 source models, C3 tiny LGBM search."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "GPU2_CHALLENGER"
CHAMPION = {"macro_f05": 0.931218, "precision": 0.974840, "recall": 0.862974}
SAFE_RETRIEVAL_FEATURES = [
    "found_R0", "found_R1", "found_R2", "found_R3", "retriever_count",
    "R1_name_char_score", "R2_address_word_score", "R3_name_word_score",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "experiments" / "B003")); import run_b003 as b003  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B005")); import run_b005 as b005  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B006")); import run_b006 as b006  # type: ignore
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import decision_rules  # type: ignore
    return b003, b005, b006, decision_rules


B003, B005, B006, DECISION = import_dependencies()
SAFE_FEATURES = list(B006.BASE_FEATURES) + SAFE_RETRIEVAL_FEATURES
SAFE_INDICES = [B006.ALL_FEATURES.index(name) for name in SAFE_FEATURES]


class DeadlineReached(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float) -> None:
        self.started = time.monotonic()
        self.limit_seconds = minutes * 60.0

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return self.limit_seconds - self.elapsed_seconds

    def check(self, stage: str, reserve_seconds: float = 0.0) -> None:
        if self.remaining_seconds <= reserve_seconds:
            raise DeadlineReached(
                f"Hard deadline reached before {stage}; remaining={self.remaining_seconds:.1f}s, "
                f"required_reserve={reserve_seconds:.1f}s"
            )


class FrozenEvaluationGate:
    """Enforce at most one frozen evaluation call for a named challenger."""
    def __init__(self, name: str, evaluator: Callable[..., Mapping[str, Any]]) -> None:
        self.name = name
        self.evaluator = evaluator
        self.calls = 0

    def evaluate(self, truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]]) -> dict[str, Any]:
        if self.calls:
            raise RuntimeError(f"Frozen evaluation leakage: {self.name} evaluated more than once")
        self.calls += 1
        return dict(self.evaluator(truth, predictions))


def atomic_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    B003.atomic_json(path, value)


def country_metrics(
    truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]],
    countries: Mapping[str, str], evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for country in sorted(set(countries.values())):
        ids = [query_id for query_id in truth if countries[query_id] == country]
        result[country] = dict(evaluator(
            {query_id: truth[query_id] for query_id in ids},
            {query_id: predictions.get(query_id, set()) for query_id in ids},
        ))
    return result


def metric_row(
    name: str, overall: Mapping[str, Any], calibration_macro: float,
    thresholds: Mapping[str, float], countries: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], **extra: Any,
) -> dict[str, Any]:
    row = {
        "configuration": name,
        "macro_f05": float(overall["macro_f05"]),
        "precision": float(overall["global_precision"]),
        "recall": float(overall["global_recall"]),
        "delta_vs_B006A_safe": float(overall["macro_f05"]) - CHAMPION["macro_f05"],
        "calibration_macro_f05": float(calibration_macro),
        "thresholds": dict(thresholds),
        "country": {key: {"macro_f05": float(value["macro_f05"])} for key, value in countries.items()},
        "S2": B006.source_metrics(truth, predictions, "S2-"),
        "S3": B006.source_metrics(truth, predictions, "S3-"),
    }
    row.update(extra)
    return row


def summarize_competition(
    query_stats: Mapping[tuple[str, str], Mapping[str, float]],
    target_stats: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    def summarize(rows: Iterable[Mapping[str, float]]) -> dict[str, Any]:
        rows = list(rows)
        if not rows:
            return {"count": 0}
        result: dict[str, Any] = {"count": len(rows)}
        for name in rows[0]:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            result[name] = {
                "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
            }
        return result
    return {
        "query_candidate_source": summarize(query_stats.values()),
        "target_claimants": summarize(target_stats.values()),
    }


def tune_source_independently(
    source: str, truth: Mapping[str, set[str]], scores: Mapping[str, Sequence[tuple[str, float]]],
    evaluator: Callable[..., Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    prefix = source + "-"
    source_truth = {query_id: {target for target in targets if target.startswith(prefix)} for query_id, targets in truth.items()}
    source_scores = {
        query_id: [(target, score) for target, score in rows if target.startswith(prefix)]
        for query_id, rows in scores.items()
    }
    cache: dict[float, dict[str, Any]] = {}

    def trial(value: float) -> dict[str, Any]:
        threshold = round(float(value), 6)
        if threshold not in cache:
            if source == "S2":
                predictions = DECISION.apply_decision_rule(source_scores, threshold, 1.01, {"name": "baseline"})
            else:
                predictions = DECISION.apply_decision_rule(source_scores, 1.01, threshold, {"name": "baseline"})
            cache[threshold] = dict(evaluator(source_truth, predictions))
        return cache[threshold]

    coarse = np.arange(0.05, 0.951, 0.05)
    best = max(coarse, key=lambda value: (float(trial(value)["macro_f05"]), -float(value)))
    fine = np.arange(max(0.0, best - 0.05), min(1.0, best + 0.05) + 0.001, 0.01)
    best = max(fine, key=lambda value: (float(trial(value)["macro_f05"]), -float(value)))
    metrics = dict(trial(best)); metrics["threshold_trials"] = len(cache)
    return round(float(best), 6), metrics


def safe_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, SAFE_INDICES].copy()


def score_source_models(
    model_s2: Any, model_s3: Any, matrix: np.ndarray,
    queries: Sequence[str], candidates: Sequence[str], all_queries: Iterable[str],
) -> dict[str, list[tuple[str, float]]]:
    result = {query_id: [] for query_id in all_queries}
    source2 = np.fromiter((candidate.startswith("S2-") for candidate in candidates), dtype=bool, count=len(candidates))
    for mask, model in ((source2, model_s2), (~source2, model_s3)):
        indices = np.flatnonzero(mask)
        probabilities = model.predict_proba(matrix[indices])
        for index, probability in zip(indices, probabilities):
            result[queries[index]].append((candidates[index], float(probability)))
    return result


def fit_source_model(
    source: str, matrix: np.ndarray, labels: np.ndarray, candidates: Sequence[str],
    args: argparse.Namespace, production: Mapping[str, Any], params: Mapping[str, Any], telemetry: Any,
) -> tuple[Any, str]:
    mask = np.fromiter((candidate.startswith(source + "-") for candidate in candidates), dtype=bool, count=len(candidates))
    model, device, _ = B006.fit_model(args, production, params, matrix[mask], labels[mask], telemetry)
    return model, device


def c3_configurations(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(params)
    candidates = [
        ("current", {}),
        ("leaves_63", {"num_leaves": 63}),
        ("leaves_127", {"num_leaves": 127}),
        ("min_child_40", {"min_child_samples": 40}),
        ("min_child_80", {"min_child_samples": 80}),
        ("depth_8_lr_0025", {"max_depth": 8, "learning_rate": 0.025}),
    ]
    return [{"name": name, "params": {**base, **updates}, "changes": updates} for name, updates in candidates]


def validate_inputs(args: argparse.Namespace) -> None:
    required = [args.b005_artifacts / "manifest.json", args.b005_metadata, args.ground_truth, args.val_ids]
    missing = [str(path) for path in required if not path.exists()]
    for split in ("train", "calibration", "evaluation"):
        if not list((args.b005_artifacts / "features" / split).glob("*.parquet")):
            missing.append(str(args.b005_artifacts / "features" / split / "*.parquet"))
    if missing:
        raise FileNotFoundError("Required B005/B006A cache is unavailable; STOP and use GPU2 for inference. Missing: " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root(); os.chdir(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    deadline = Deadline(args.deadline_minutes)
    heartbeat = B006.Heartbeat(args.heartbeat_seconds); heartbeat.start(); atexit.register(heartbeat.stop)
    original_record = telemetry.record

    def record(event: str, **kwargs: Any) -> None:
        heartbeat.update(event)
        kwargs.setdefault("extra", {})
        kwargs["extra"] = {**kwargs["extra"], "elapsed_seconds": deadline.elapsed_seconds, "remaining_seconds": deadline.remaining_seconds}
        print(f"[{B003.utc_now()}] GPU2 {event} | elapsed={deadline.elapsed_seconds:.0f}s | remaining={deadline.remaining_seconds:.0f}s", flush=True)
        original_record(event, **kwargs)

    telemetry.record = record
    record("start", extra={"deadline_minutes": args.deadline_minutes})
    manifest: dict[str, Any] = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "RUNNING",
        "created_at": B003.utc_now(), "git_branch": B003.git_value(root, "branch", "--show-current"),
        "git_commit": B003.git_value(root, "rev-parse", "HEAD"), "cache_only": True,
        "retrieval_regeneration": False, "safe_features": SAFE_FEATURES,
        "deadline_minutes": args.deadline_minutes,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)

    try:
        validate_inputs(args)
        b005_manifest, b005_metadata = B006.validate_cache(args)
    except Exception as exc:
        manifest.update({"status": "STOPPED_CACHE_UNAVAILABLE", "stopped_at": B003.utc_now(), "error": repr(exc), "recommendation": "Reassign GPU2 to final test inference."})
        atomic_json(args.output_dir / "manifest.json", manifest)
        record("cache_unavailable_stop", extra={"error": repr(exc)})
        print(str(exc), flush=True); heartbeat.stop(); return 2

    production = B003.import_production(root)
    paths = {split: B006.feature_paths(args.b005_artifacts, split) for split in ("train", "calibration", "evaluation")}
    matrices: dict[str, np.ndarray] = {}; labels: dict[str, np.ndarray] = {}
    row_ids: dict[str, tuple[list[str], list[str], list[str]]] = {}; split_ids: dict[str, set[str]] = {}
    try:
        for split in ("train", "calibration", "evaluation"):
            deadline.check(f"load_{split}")
            best, ids, rows = B006.channel_best_maps(paths[split]); split_ids[split] = ids
            matrix, y, queries, candidates, countries, _evidence = B006.load_split(paths[split], best, False, telemetry)
            matrices[split] = matrix; labels[split] = y; row_ids[split] = (queries, candidates, countries)
            record("cache_split_loaded", candidate_pairs=rows, processed_s1=len(ids), extra={"split": split})
    except DeadlineReached as exc:
        stopped = {"status": "SKIPPED_DEADLINE", "reason": str(exc)}
        atomic_json(args.output_dir / "c1_decision_metrics.json", stopped)
        atomic_json(args.output_dir / "c2_source_models_metrics.json", stopped)
        comparison = {
            "baseline": CHAMPION,
            "rows": [{"configuration": "baseline B006A-safe", **CHAMPION, "delta_vs_B006A_safe": 0.0}],
            "best_gpu2_challenger": None, "worth_considering": False, "promoted_automatically": False,
        }
        atomic_json(args.output_dir / "comparison.json", comparison)
        manifest.update({"status": "STOPPED_DEADLINE", "stopped_at": B003.utc_now(), "reason": str(exc), "comparison": comparison})
        atomic_json(args.output_dir / "manifest.json", manifest)
        record("deadline_stop_during_cache_load", extra={"reason": str(exc)})
        heartbeat.stop(); return 0
    if split_ids["train"] & split_ids["calibration"] or split_ids["train"] & split_ids["evaluation"] or split_ids["calibration"] & split_ids["evaluation"]:
        raise RuntimeError("Cached train/calibration/evaluation query IDs overlap")
    frozen_ids, _ = B003.load_selected_ids(args.val_ids, 5000)
    if split_ids["evaluation"] != set(frozen_ids):
        raise RuntimeError("Evaluation cache is not the frozen 5,000-ID split")
    truth = B003.load_ground_truth(args.ground_truth, sorted(set().union(*split_ids.values())))
    for split in matrices:
        q, c, _ = row_ids[split]
        mismatches = sum(int(int(label) != int(candidate in truth[query])) for label, query, candidate in zip(labels[split], q, c))
        if mismatches:
            raise RuntimeError(f"{split} cache has {mismatches} label mismatches")
    manifest["cache"] = {
        "status": "AVAILABLE", "b005_manifest_sha256": B003.sha256_file(args.b005_artifacts / "manifest.json"),
        "b005_metadata_sha256": B003.sha256_file(args.b005_metadata),
        "rows": {split: int(len(labels[split])) for split in labels},
        "queries": {split: len(split_ids[split]) for split in split_ids},
        "frozen_evaluation_ids_verified": True,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)

    cal_truth = {query: truth[query] for query in split_ids["calibration"]}
    eval_truth = {query: truth[query] for query in split_ids["evaluation"]}
    eval_countries = {query: country for query, country in zip(row_ids["evaluation"][0], row_ids["evaluation"][2])}
    completed: list[dict[str, Any]] = []
    c1_result: dict[str, Any] = {"status": "NOT_RUN"}
    c2_result: dict[str, Any] = {"status": "NOT_RUN"}
    c3_result: dict[str, Any] = {"status": "NOT_REQUESTED"}
    safe_model = None

    # C1: one safe model, calibration-only rule selection, one frozen evaluation.
    try:
        deadline.check("C1", args.stage_reserve_seconds)
        record("c1_start")
        X_train = safe_matrix(matrices["train"])
        safe_model, safe_device, safe_params = B006.fit_model(args, production, b005_metadata["model_params"], X_train, labels["train"], telemetry)
        del X_train; gc.collect()
        cal_scores = B006.score_dict(safe_model, safe_matrix(matrices["calibration"]), row_ids["calibration"][0], row_ids["calibration"][1], split_ids["calibration"])
        t2, t3, safe_cal_metrics = B005.tune_thresholds(cal_truth, cal_scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
        winner_rule, trials = DECISION.select_rule_on_calibration(cal_truth, cal_scores, t2, t3, DECISION.default_rule_family(), production["evaluate_predictions"])
        qstats, tstats = DECISION.competition_features(cal_scores)
        deadline.check("C1 frozen evaluation")
        eval_scores = B006.score_dict(safe_model, safe_matrix(matrices["evaluation"]), row_ids["evaluation"][0], row_ids["evaluation"][1], split_ids["evaluation"])
        predictions = DECISION.apply_decision_rule(eval_scores, t2, t3, winner_rule)
        gate = FrozenEvaluationGate("C1", production["evaluate_predictions"])
        overall = gate.evaluate(eval_truth, predictions)
        countries = country_metrics(eval_truth, predictions, eval_countries, production["evaluate_predictions"])
        winning_cal = next(row for row in trials if row["rule"] == winner_rule)
        c1_result = metric_row(
            "C1 decision", overall, winning_cal["metrics"]["macro_f05"], {"S2": t2, "S3": t3}, countries,
            eval_truth, predictions, status="COMPLETE", selected_rule=winner_rule, calibration_trials=trials,
            competition_summary=summarize_competition(qstats, tstats), frozen_evaluation_calls=gate.calls,
            device=safe_device, model_params=safe_params,
        )
        atomic_json(args.output_dir / "c1_decision_metrics.json", c1_result); completed.append(c1_result)
        safe_model.save(str(args.output_dir / "c1_safe_model.joblib"))
        record("c1_complete", extra={"macro_f05": c1_result["macro_f05"], "delta": c1_result["delta_vs_B006A_safe"]})
        del eval_scores, predictions; gc.collect()
    except DeadlineReached as exc:
        c1_result = {"status": "SKIPPED_DEADLINE", "reason": str(exc)}
        atomic_json(args.output_dir / "c1_decision_metrics.json", c1_result)

    # C2: source-specific fits and independent source calibration.
    if c1_result.get("status") == "COMPLETE":
        try:
            deadline.check("C2", args.stage_reserve_seconds)
            record("c2_start")
            train_matrix = safe_matrix(matrices["train"])
            model_s2, device_s2 = fit_source_model("S2", train_matrix, labels["train"], row_ids["train"][1], args, production, b005_metadata["model_params"], telemetry)
            deadline.check("C2 S3 fit", args.stage_reserve_seconds / 2)
            model_s3, device_s3 = fit_source_model("S3", train_matrix, labels["train"], row_ids["train"][1], args, production, b005_metadata["model_params"], telemetry)
            del train_matrix; gc.collect()
            cal_scores_c2 = score_source_models(model_s2, model_s3, safe_matrix(matrices["calibration"]), row_ids["calibration"][0], row_ids["calibration"][1], split_ids["calibration"])
            c2_t2, c2_cal_s2 = tune_source_independently("S2", cal_truth, cal_scores_c2, production["evaluate_predictions"])
            c2_t3, c2_cal_s3 = tune_source_independently("S3", cal_truth, cal_scores_c2, production["evaluate_predictions"])
            cal_predictions_c2 = DECISION.apply_decision_rule(cal_scores_c2, c2_t2, c2_t3, {"name": "baseline"})
            c2_cal_combined = dict(production["evaluate_predictions"](cal_truth, cal_predictions_c2))
            deadline.check("C2 frozen evaluation")
            eval_scores_c2 = score_source_models(model_s2, model_s3, safe_matrix(matrices["evaluation"]), row_ids["evaluation"][0], row_ids["evaluation"][1], split_ids["evaluation"])
            predictions_c2 = DECISION.apply_decision_rule(eval_scores_c2, c2_t2, c2_t3, {"name": "baseline"})
            gate = FrozenEvaluationGate("C2", production["evaluate_predictions"])
            overall = gate.evaluate(eval_truth, predictions_c2)
            countries = country_metrics(eval_truth, predictions_c2, eval_countries, production["evaluate_predictions"])
            c2_result = metric_row(
                "C2 source-specific", overall, c2_cal_combined["macro_f05"], {"S2": c2_t2, "S3": c2_t3}, countries,
                eval_truth, predictions_c2, status="COMPLETE", independent_calibration={"S2": c2_cal_s2, "S3": c2_cal_s3},
                devices={"S2": device_s2, "S3": device_s3}, frozen_evaluation_calls=gate.calls,
            )
            atomic_json(args.output_dir / "c2_source_models_metrics.json", c2_result); completed.append(c2_result)
            model_s2.save(str(args.output_dir / "c2_model_s2.joblib")); model_s3.save(str(args.output_dir / "c2_model_s3.joblib"))
            record("c2_complete", extra={"macro_f05": c2_result["macro_f05"], "delta": c2_result["delta_vs_B006A_safe"]})
            del model_s2, model_s3, cal_scores_c2, eval_scores_c2, predictions_c2; gc.collect()
        except DeadlineReached as exc:
            c2_result = {"status": "SKIPPED_DEADLINE", "reason": str(exc)}
            atomic_json(args.output_dir / "c2_source_models_metrics.json", c2_result)
    else:
        c2_result = {"status": "SKIPPED", "reason": "C1 did not complete cleanly"}
        atomic_json(args.output_dir / "c2_source_models_metrics.json", c2_result)

    # C3 is explicitly opt-in and only starts after clean C1/C2 completion.
    if args.run_c3 and c1_result.get("status") == "COMPLETE" and c2_result.get("status") == "COMPLETE":
        try:
            deadline.check("C3", args.c3_minimum_remaining_seconds)
            record("c3_start")
            configurations = c3_configurations(b005_metadata["model_params"])
            trials = []
            best_model = safe_model
            best_record = {
                "name": "current", "params": configurations[0]["params"], "changes": {},
                "threshold_s2": c1_result["thresholds"]["S2"], "threshold_s3": c1_result["thresholds"]["S3"],
                "calibration_macro_f05": float(safe_cal_metrics["macro_f05"]), "device": c1_result["device"],
            }
            trials.append(dict(best_record))
            for configuration in configurations[1:]:
                deadline.check(f"C3 {configuration['name']}", args.stage_reserve_seconds)
                model, device, _ = B006.fit_model(args, production, configuration["params"], safe_matrix(matrices["train"]), labels["train"], telemetry)
                scores = B006.score_dict(model, safe_matrix(matrices["calibration"]), row_ids["calibration"][0], row_ids["calibration"][1], split_ids["calibration"])
                hp_t2, hp_t3, hp_metrics = B005.tune_thresholds(cal_truth, scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
                record = {
                    "name": configuration["name"], "params": configuration["params"], "changes": configuration["changes"],
                    "threshold_s2": hp_t2, "threshold_s3": hp_t3, "calibration_macro_f05": float(hp_metrics["macro_f05"]), "device": device,
                }
                trials.append(record)
                if record["calibration_macro_f05"] > best_record["calibration_macro_f05"]:
                    if best_model is not safe_model: del best_model
                    best_model = model; best_record = record
                else:
                    del model
                del scores; gc.collect()
            deadline.check("C3 frozen evaluation")
            scores = B006.score_dict(best_model, safe_matrix(matrices["evaluation"]), row_ids["evaluation"][0], row_ids["evaluation"][1], split_ids["evaluation"])
            predictions = production["apply_threshold_and_deduplication"](scores, best_record["threshold_s2"], best_record["threshold_s3"])
            gate = FrozenEvaluationGate("C3", production["evaluate_predictions"])
            overall = gate.evaluate(eval_truth, predictions)
            countries = country_metrics(eval_truth, predictions, eval_countries, production["evaluate_predictions"])
            c3_result = metric_row(
                "C3 tuned LGBM", overall, best_record["calibration_macro_f05"], {"S2": best_record["threshold_s2"], "S3": best_record["threshold_s3"]},
                countries, eval_truth, predictions, status="COMPLETE", selected_configuration=best_record,
                calibration_trials=trials, frozen_evaluation_calls=gate.calls,
            )
            atomic_json(args.output_dir / "c3_tuning_metrics.json", c3_result); completed.append(c3_result)
            best_model.save(str(args.output_dir / "c3_winner_model.joblib"))
            record("c3_complete", extra={"macro_f05": c3_result["macro_f05"], "delta": c3_result["delta_vs_B006A_safe"]})
            if best_model is not safe_model: del best_model
            del scores, predictions; gc.collect()
        except DeadlineReached as exc:
            c3_result = {"status": "SKIPPED_DEADLINE", "reason": str(exc)}
            atomic_json(args.output_dir / "c3_tuning_metrics.json", c3_result)

    comparison_rows = [{"configuration": "baseline B006A-safe", **CHAMPION, "delta_vs_B006A_safe": 0.0}]
    comparison_rows.extend({key: row[key] for key in ("configuration", "macro_f05", "precision", "recall", "delta_vs_B006A_safe")} for row in completed)
    best = max(completed, key=lambda row: row["macro_f05"]) if completed else None
    comparison = {
        "baseline": CHAMPION, "rows": comparison_rows,
        "best_gpu2_challenger": None if best is None else best["configuration"],
        "best_macro_f05": None if best is None else best["macro_f05"],
        "best_delta": None if best is None else best["delta_vs_B006A_safe"],
        "worth_considering": bool(best and best["macro_f05"] > CHAMPION["macro_f05"]),
        "material_improvement": bool(best and best["delta_vs_B006A_safe"] >= 0.001),
        "promoted_automatically": False,
    }
    atomic_json(args.output_dir / "comparison.json", comparison)
    manifest.update({
        "status": "COMPLETE", "completed_at": B003.utc_now(), "elapsed_seconds": deadline.elapsed_seconds,
        "c1_status": c1_result.get("status"), "c2_status": c2_result.get("status"), "c3_status": c3_result.get("status"),
        "comparison": comparison, "production_handoff": {
            "decision_module": "experiments/GPU2_CHALLENGER/decision_rules.py",
            "safe_feature_names": SAFE_FEATURES,
            "artifacts_to_reuse": [
                "c1_decision_metrics.json", "c1_safe_model.joblib", "c2_model_s2.joblib", "c2_model_s3.joblib",
                "c3_winner_model.joblib", "comparison.json",
            ],
        },
        "recommendation": "Consider only a completed challenger with frozen Macro F0.5 above 0.931218; do not auto-promote.",
    })
    atomic_json(args.output_dir / "manifest.json", manifest)
    heartbeat.stop()
    print("\nGPU2 CHALLENGER STOP", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True)
    parser.add_argument("--b005-metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/GPU2_CHALLENGER"))
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--deadline-minutes", type=float, default=90.0)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--stage-reserve-seconds", type=float, default=300.0)
    parser.add_argument("--c3-minimum-remaining-seconds", type=float, default=1800.0)
    parser.add_argument("--run-c3", action="store_true", help="Run the optional six-configuration C3 search if C1/C2 finish and time remains")
    args = parser.parse_args(argv)
    if args.deadline_minutes <= 0 or args.heartbeat_seconds <= 0:
        parser.error("deadline and heartbeat must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
