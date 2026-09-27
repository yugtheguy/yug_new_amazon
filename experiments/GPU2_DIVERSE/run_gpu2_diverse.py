#!/usr/bin/env python3
"""Cache-only CatBoost/XGBoost diversity challenge with optional B007 blending."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EXPERIMENT_ID = "GPU2_DIVERSE"
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
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import diverse_utils  # type: ignore
    return b003, b005, b006, diverse_utils


B003, B005, B006, UTILS = import_dependencies()
SAFE_FEATURES = list(B006.BASE_FEATURES) + SAFE_RETRIEVAL_FEATURES
SAFE_INDICES = [B006.ALL_FEATURES.index(name) for name in SAFE_FEATURES]


class DeadlineReached(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float) -> None:
        self.started = time.monotonic(); self.limit = minutes * 60.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def require(self, stage: str, reserve: float = 0.0) -> None:
        if self.remaining <= reserve:
            raise DeadlineReached(f"deadline before {stage}: remaining={self.remaining:.1f}s reserve={reserve:.1f}s")


def safe_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix[:, SAFE_INDICES], columns=SAFE_FEATURES, copy=True)


def write_probability_parquet(
    path: Path, queries: Sequence[str], candidates: Sequence[str], countries: Sequence[str],
    labels: np.ndarray, probabilities: Sequence[float], probability_name: str,
) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    rows = UTILS.probability_rows(queries, candidates, countries, labels, probabilities, probability_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.table(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def country_metrics(
    truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], countries: Mapping[str, str], evaluator: Any,
) -> dict[str, dict[str, Any]]:
    result = {}
    for country in sorted(set(countries.values())):
        ids = [query_id for query_id in truth if countries[query_id] == country]
        result[country] = dict(evaluator(
            {query_id: truth[query_id] for query_id in ids},
            {query_id: predictions.get(query_id, set()) for query_id in ids},
        ))
    return result


def evaluate_probabilities(
    probabilities: Sequence[float], queries: Sequence[str], candidates: Sequence[str],
    truth: Mapping[str, set[str]], evaluator: Any, apply: Any, thresholds: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    scores = UTILS.scores_from_columns(queries, candidates, probabilities)
    predictions = apply(scores, thresholds[0], thresholds[1])
    return dict(evaluator(truth, predictions)), predictions


def tune_probabilities(
    probabilities: Sequence[float], queries: Sequence[str], candidates: Sequence[str],
    truth: Mapping[str, set[str]], evaluator: Any, apply: Any,
) -> tuple[float, float, dict[str, Any]]:
    scores = UTILS.scores_from_columns(queries, candidates, probabilities)
    return B005.tune_thresholds(truth, scores, apply, evaluator)


def metric_payload(
    model: str, overall: Mapping[str, Any], calibration: Mapping[str, Any], thresholds: tuple[float, float],
    country: Mapping[str, Mapping[str, Any]], distribution: Mapping[str, Any], runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "COMPLETE", "model": model, "macro_f05": float(overall["macro_f05"]),
        "precision": float(overall["global_precision"]), "recall": float(overall["global_recall"]),
        "delta_vs_B006A": float(overall["macro_f05"]) - UTILS.BASELINE_B006A["macro_f05"],
        "delta_vs_B007": float(overall["macro_f05"]) - UTILS.BASELINE_B007["macro_f05"],
        "threshold_s2": thresholds[0], "threshold_s3": thresholds[1],
        "calibration_macro_f05": float(calibration["macro_f05"]),
        "country": {name: {"macro_f05": float(value["macro_f05"])} for name, value in country.items()},
        "inference_probability_distribution": dict(distribution), "runtime_seconds": runtime_seconds,
    }


def fit_catboost(
    configuration: Mapping[str, Any], train: pd.DataFrame, y_train: np.ndarray,
    calibration: pd.DataFrame, y_calibration: np.ndarray,
) -> Any:
    from catboost import CatBoostClassifier  # type: ignore
    model = CatBoostClassifier(**configuration)
    model.fit(
        train, y_train, eval_set=(calibration, y_calibration),
        early_stopping_rounds=75, verbose=50,
    )
    return model


def fit_xgboost(train: pd.DataFrame, y_train: np.ndarray, calibration: pd.DataFrame, y_calibration: np.ndarray) -> tuple[Any, dict[str, Any]]:
    from xgboost import XGBClassifier  # type: ignore
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 750,
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.90,
        "colsample_bytree": 0.90, "min_child_weight": 5, "reg_lambda": 5.0,
        "reg_alpha": 0.1, "random_state": 42, "n_jobs": -1,
        "tree_method": "hist", "device": "cuda", "early_stopping_rounds": 75,
    }
    model = XGBClassifier(**params)
    try:
        model.fit(train, y_train, eval_set=[(calibration, y_calibration)], verbose=50)
        return model, params
    except Exception as first_error:
        # Compatibility fallback for older Kaggle XGBoost; this is the same
        # configuration and not a second accuracy experiment.
        params.pop("device", None); params["tree_method"] = "gpu_hist"
        model = XGBClassifier(**params)
        try:
            model.fit(train, y_train, eval_set=[(calibration, y_calibration)], verbose=50)
            return model, params
        except Exception as second_error:
            raise RuntimeError(f"XGBoost GPU failed with modern and legacy GPU selectors: {first_error!r}; {second_error!r}")


def read_prediction(path: Path, probability_column: str) -> pd.DataFrame:
    import pyarrow.parquet as pq  # type: ignore
    required = UTILS.KEY_COLUMNS + [probability_column]
    frame = pq.read_table(path, columns=required).to_pandas()
    frame["query_id"] = frame["query_id"].astype(str); frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["candidate_source"] = frame["candidate_source"].astype(str)
    return frame


def run_blend(
    args: argparse.Namespace, output_dir: Path, cal_truth: Mapping[str, set[str]], eval_truth: Mapping[str, set[str]],
    production: Mapping[str, Any], cat_metrics: Mapping[str, Any], xgb_metrics: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.b007_calibration_predictions or not args.b007_evaluation_predictions:
        result = {"status": "NOT_AVAILABLE", "reason": "B007 calibration/evaluation predictions were not supplied"}
        B003.atomic_json(output_dir / "blend_search.json", result)
        return result, {"status": "NOT_AVAILABLE"}
    cat_cal = read_prediction(output_dir / "catboost_calibration_predictions.parquet", "catboost_probability")
    cat_eval = read_prediction(output_dir / "catboost_evaluation_predictions.parquet", "catboost_probability")
    b007_cal = read_prediction(args.b007_calibration_predictions, args.b007_probability_column)
    b007_eval = read_prediction(args.b007_evaluation_predictions, args.b007_probability_column)
    b007_cal, cat_cal = UTILS.assert_canonical_alignment(b007_cal, cat_cal, "B007 calibration", "CatBoost calibration")
    b007_eval, cat_eval = UTILS.assert_canonical_alignment(b007_eval, cat_eval, "B007 evaluation", "CatBoost evaluation")
    trials = []
    for cat_weight in (0.0, 0.05, 0.10, 0.20, 0.30):
        probabilities = UTILS.blend_probabilities(b007_cal[args.b007_probability_column], cat_cal["catboost_probability"], cat_weight)
        t2, t3, metrics = tune_probabilities(probabilities, b007_cal["query_id"], b007_cal["candidate_id"], cal_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"])
        trials.append({"name": f"B007_{1-cat_weight:.2f}_Cat_{cat_weight:.2f}", "catboost_weight": cat_weight, "xgboost_weight": 0.0, "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(metrics["macro_f05"])})
    xgb_cal = xgb_eval = None
    if xgb_metrics and float(xgb_metrics.get("macro_f05", 0.0)) >= 0.93:
        xgb_cal = read_prediction(output_dir / "xgboost_calibration_predictions.parquet", "xgboost_probability")
        xgb_eval = read_prediction(output_dir / "xgboost_evaluation_predictions.parquet", "xgboost_probability")
        b007_cal, xgb_cal = UTILS.assert_canonical_alignment(b007_cal, xgb_cal, "B007 calibration", "XGBoost calibration")
        b007_eval, xgb_eval = UTILS.assert_canonical_alignment(b007_eval, xgb_eval, "B007 evaluation", "XGBoost evaluation")
        for weight in (0.05, 0.10, 0.20):
            probabilities = UTILS.blend_probabilities(b007_cal[args.b007_probability_column], xgb_cal["xgboost_probability"], weight)
            t2, t3, metrics = tune_probabilities(probabilities, b007_cal["query_id"], b007_cal["candidate_id"], cal_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"])
            trials.append({"name": f"B007_{1-weight:.2f}_XGB_{weight:.2f}", "catboost_weight": 0.0, "xgboost_weight": weight, "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(metrics["macro_f05"])})
        probabilities = 0.80 * b007_cal[args.b007_probability_column].to_numpy() + 0.10 * cat_cal["catboost_probability"].to_numpy() + 0.10 * xgb_cal["xgboost_probability"].to_numpy()
        t2, t3, metrics = tune_probabilities(probabilities, b007_cal["query_id"], b007_cal["candidate_id"], cal_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"])
        trials.append({"name": "B007_0.80_Cat_0.10_XGB_0.10", "catboost_weight": 0.10, "xgboost_weight": 0.10, "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(metrics["macro_f05"])})
    winner = max(trials, key=lambda row: (row["calibration_macro_f05"], -row["catboost_weight"] - row["xgboost_weight"]))
    primary_weight = 1.0 - winner["catboost_weight"] - winner["xgboost_weight"]
    eval_probabilities = primary_weight * b007_eval[args.b007_probability_column].to_numpy() + winner["catboost_weight"] * cat_eval["catboost_probability"].to_numpy()
    if winner["xgboost_weight"]:
        assert xgb_eval is not None
        eval_probabilities += winner["xgboost_weight"] * xgb_eval["xgboost_probability"].to_numpy()
    overall, predictions = evaluate_probabilities(eval_probabilities, b007_eval["query_id"], b007_eval["candidate_id"], eval_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"], (winner["threshold_s2"], winner["threshold_s3"]))
    result = {
        "status": "COMPLETE", "calibration_trials": trials, "selected": winner,
        "frozen_evaluation": overall, "delta_vs_B007": float(overall["macro_f05"]) - UTILS.BASELINE_B007["macro_f05"],
        "promote": float(overall["macro_f05"]) > UTILS.BASELINE_B007["macro_f05"],
        "prefer_promotion": float(overall["macro_f05"]) - UTILS.BASELINE_B007["macro_f05"] >= 0.0005,
    }
    B003.atomic_json(output_dir / "blend_search.json", result)
    baseline = next(row for row in trials if row["catboost_weight"] == 0 and row["xgboost_weight"] == 0)
    b007_overall, b007_predictions = evaluate_probabilities(b007_eval[args.b007_probability_column], b007_eval["query_id"], b007_eval["candidate_id"], eval_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"], (baseline["threshold_s2"], baseline["threshold_s3"]))
    _cat_overall, cat_predictions = evaluate_probabilities(cat_eval["catboost_probability"], cat_eval["query_id"], cat_eval["candidate_id"], eval_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"], (cat_metrics["threshold_s2"], cat_metrics["threshold_s3"]))
    diversity = UTILS.diversity_report(eval_truth, b007_predictions, cat_predictions, b007_eval[args.b007_probability_column], cat_eval["catboost_probability"])
    diversity["b007_recomputed_metrics"] = b007_overall
    return result, diversity


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); root = repo_root(); os.chdir(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(args.deadline_minutes)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    heartbeat = B006.Heartbeat(args.heartbeat_seconds); heartbeat.start(); atexit.register(heartbeat.stop)
    original_record = telemetry.record
    def record(event: str, **kwargs: Any) -> None:
        heartbeat.update(event); kwargs.setdefault("extra", {}); kwargs["extra"].update({"elapsed_seconds": deadline.elapsed, "remaining_seconds": deadline.remaining})
        print(f"[{B003.utc_now()}] GPU2_DIVERSE {event} elapsed={deadline.elapsed:.0f}s remaining={deadline.remaining:.0f}s", flush=True); original_record(event, **kwargs)
    telemetry.record = record; record("start")
    manifest = {"experiment_id": EXPERIMENT_ID, "status": "RUNNING", "created_at": B003.utc_now(), "git_branch": B003.git_value(root, "branch", "--show-current"), "git_commit": B003.git_value(root, "rev-parse", "HEAD"), "cache_only": True, "safe_features": SAFE_FEATURES, "deadline_minutes": args.deadline_minutes}
    B003.atomic_json(args.output_dir / "manifest.json", manifest)
    B006.validate_cache(args)
    production = B003.import_production(root)
    paths = {split: B006.feature_paths(args.b005_artifacts, split) for split in ("train", "calibration", "evaluation")}
    matrices = {}; labels = {}; rows = {}; split_ids = {}
    for split in ("train", "calibration", "evaluation"):
        deadline.require(f"load {split}")
        best, ids, count = B006.channel_best_maps(paths[split]); split_ids[split] = ids
        matrix, y, queries, candidates, countries, _ = B006.load_split(paths[split], best, False, telemetry)
        matrices[split] = safe_frame(matrix); labels[split] = y; rows[split] = (queries, candidates, countries)
        del matrix; UTILS.validate_numeric_matrix(matrices[split], y, SAFE_FEATURES, split)
        record("cache_split_validated", candidate_pairs=count, processed_s1=len(ids), extra={"split": split})
    if split_ids["train"] & split_ids["calibration"] or split_ids["train"] & split_ids["evaluation"] or split_ids["calibration"] & split_ids["evaluation"]:
        raise RuntimeError("split overlap")
    frozen_ids, _ = B003.load_selected_ids(args.val_ids, 5000)
    if split_ids["evaluation"] != set(frozen_ids): raise RuntimeError("evaluation IDs differ from frozen 5k")
    truth = B003.load_ground_truth(args.ground_truth, sorted(set().union(*split_ids.values())))
    for split in rows:
        q, c, _ = rows[split]
        if any(int(label) != int(candidate in truth[query]) for label, query, candidate in zip(labels[split], q, c)):
            raise RuntimeError(f"{split} row-label alignment failed")
    cal_truth = {query: truth[query] for query in split_ids["calibration"]}; eval_truth = {query: truth[query] for query in split_ids["evaluation"]}
    eval_countries = {query: country for query, country in zip(rows["evaluation"][0], rows["evaluation"][2])}
    manifest["cache_validation"] = {"status": "PASS", "feature_count": len(SAFE_FEATURES), "rows": {split: len(labels[split]) for split in labels}, "queries": {split: len(split_ids[split]) for split in split_ids}, "numeric_finite_aligned": True, "frozen_5000_verified": True}
    B003.atomic_json(args.output_dir / "manifest.json", manifest)

    cat_start = time.monotonic(); deadline.require("CatBoost fit", args.stage_reserve_seconds)
    cat_configs = [{"task_type": "GPU", "loss_function": "Logloss", "eval_metric": "Logloss", "iterations": 800, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 5, "random_seed": 42, "allow_writing_files": False}]
    if args.try_second_catboost and deadline.remaining >= args.second_catboost_minimum_seconds:
        cat_configs.append({"task_type": "GPU", "loss_function": "Logloss", "eval_metric": "Logloss", "iterations": 700, "depth": 7, "learning_rate": 0.06, "l2_leaf_reg": 8, "random_seed": 42, "allow_writing_files": False})
    cat_trials = []; best_cat = None; best_cat_trial = None; best_cat_cal_probs = None
    for index, configuration in enumerate(cat_configs, start=1):
        deadline.require(f"CatBoost configuration {index}", args.stage_reserve_seconds)
        record("catboost_fit_start", extra={"configuration": index, "params": configuration})
        model = fit_catboost(configuration, matrices["train"], labels["train"], matrices["calibration"], labels["calibration"])
        probabilities = model.predict_proba(matrices["calibration"])[:, 1]
        t2, t3, metrics = tune_probabilities(probabilities, rows["calibration"][0], rows["calibration"][1], cal_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"])
        trial = {"configuration": index, "params": configuration, "best_iteration": model.get_best_iteration(), "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(metrics["macro_f05"])}; cat_trials.append(trial)
        if best_cat_trial is None or trial["calibration_macro_f05"] > best_cat_trial["calibration_macro_f05"]:
            if best_cat is not None: del best_cat
            best_cat = model; best_cat_trial = trial; best_cat_cal_probs = probabilities
        else: del model
        gc.collect(); record("catboost_fit_done", extra=trial)
    assert best_cat is not None and best_cat_trial is not None and best_cat_cal_probs is not None
    deadline.require("CatBoost frozen evaluation")
    cat_eval_probs = best_cat.predict_proba(matrices["evaluation"])[:, 1]
    cat_overall, cat_predictions = evaluate_probabilities(cat_eval_probs, rows["evaluation"][0], rows["evaluation"][1], eval_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"], (best_cat_trial["threshold_s2"], best_cat_trial["threshold_s3"]))
    cat_country = country_metrics(eval_truth, cat_predictions, eval_countries, production["evaluate_predictions"])
    cat_metrics = metric_payload("CatBoost", cat_overall, {"macro_f05": best_cat_trial["calibration_macro_f05"]}, (best_cat_trial["threshold_s2"], best_cat_trial["threshold_s3"]), cat_country, UTILS.probability_distribution(cat_eval_probs), time.monotonic() - cat_start)
    cat_metrics.update({"selected_configuration": best_cat_trial, "calibration_trials": cat_trials, "stop_rule_triggered": float(cat_overall["macro_f05"]) < 0.93})
    best_cat.save_model(str(args.output_dir / "catboost_model.cbm"))
    write_probability_parquet(args.output_dir / "catboost_calibration_predictions.parquet", *rows["calibration"], labels["calibration"], best_cat_cal_probs, "catboost_probability")
    write_probability_parquet(args.output_dir / "catboost_evaluation_predictions.parquet", *rows["evaluation"], labels["evaluation"], cat_eval_probs, "catboost_probability")
    B003.atomic_json(args.output_dir / "catboost_metrics.json", cat_metrics); record("catboost_complete", extra={"macro_f05": cat_metrics["macro_f05"]})

    xgb_metrics = None
    if args.run_xgboost and not cat_metrics["stop_rule_triggered"] and deadline.remaining >= args.xgboost_minimum_seconds:
        xgb_start = time.monotonic(); record("xgboost_fit_start")
        xgb_model, xgb_params = fit_xgboost(matrices["train"], labels["train"], matrices["calibration"], labels["calibration"])
        xgb_cal_probs = xgb_model.predict_proba(matrices["calibration"])[:, 1]
        x2, x3, xcal = tune_probabilities(xgb_cal_probs, rows["calibration"][0], rows["calibration"][1], cal_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"])
        deadline.require("XGBoost frozen evaluation")
        xgb_eval_probs = xgb_model.predict_proba(matrices["evaluation"])[:, 1]
        xoverall, xpred = evaluate_probabilities(xgb_eval_probs, rows["evaluation"][0], rows["evaluation"][1], eval_truth, production["evaluate_predictions"], production["apply_threshold_and_deduplication"], (x2, x3))
        xcountry = country_metrics(eval_truth, xpred, eval_countries, production["evaluate_predictions"])
        xgb_metrics = metric_payload("XGBoost", xoverall, xcal, (x2, x3), xcountry, UTILS.probability_distribution(xgb_eval_probs), time.monotonic() - xgb_start); xgb_metrics["configuration"] = xgb_params
        xgb_model.save_model(str(args.output_dir / "xgboost_model.json"))
        write_probability_parquet(args.output_dir / "xgboost_calibration_predictions.parquet", *rows["calibration"], labels["calibration"], xgb_cal_probs, "xgboost_probability")
        write_probability_parquet(args.output_dir / "xgboost_evaluation_predictions.parquet", *rows["evaluation"], labels["evaluation"], xgb_eval_probs, "xgboost_probability")
        B003.atomic_json(args.output_dir / "xgboost_metrics.json", xgb_metrics); record("xgboost_complete", extra={"macro_f05": xgb_metrics["macro_f05"]})
    elif args.run_xgboost:
        xgb_metrics = {"status": "SKIPPED", "reason": "CatBoost stop rule or insufficient time"}; B003.atomic_json(args.output_dir / "xgboost_metrics.json", xgb_metrics)

    deadline.require("blend/report finalization")
    blend, diversity = run_blend(args, args.output_dir, cal_truth, eval_truth, production, cat_metrics, xgb_metrics)
    rows_out = [
        {"model": "B006A LGBM", **UTILS.BASELINE_B006A, "delta_vs_B006A": 0.0, "delta_vs_B007": UTILS.BASELINE_B006A["macro_f05"] - UTILS.BASELINE_B007["macro_f05"]},
        {key: cat_metrics[key] for key in ("model", "macro_f05", "precision", "recall", "delta_vs_B006A", "delta_vs_B007")},
    ]
    if xgb_metrics and xgb_metrics.get("status") == "COMPLETE": rows_out.append({key: xgb_metrics[key] for key in ("model", "macro_f05", "precision", "recall", "delta_vs_B006A", "delta_vs_B007")})
    rows_out.append({"model": "B007_03", **UTILS.BASELINE_B007, "delta_vs_B006A": UTILS.BASELINE_B007["macro_f05"] - UTILS.BASELINE_B006A["macro_f05"], "delta_vs_B007": 0.0})
    if blend.get("status") == "COMPLETE":
        bm = blend["frozen_evaluation"]; rows_out.append({"model": "Best Blend", "macro_f05": bm["macro_f05"], "precision": bm["global_precision"], "recall": bm["global_recall"], "delta_vs_B006A": bm["macro_f05"] - UTILS.BASELINE_B006A["macro_f05"], "delta_vs_B007": bm["macro_f05"] - UTILS.BASELINE_B007["macro_f05"]})
    comparison = {"rows": rows_out, "error_diversity": diversity, "recommendation": "Promote only a blend above B007_03; prefer >= +0.0005. Otherwise retain B007_03.", "stop": True}
    B003.atomic_json(args.output_dir / "comparison.json", comparison)
    manifest.update({"status": "COMPLETE", "completed_at": B003.utc_now(), "runtime_seconds": deadline.elapsed, "catboost": cat_metrics, "xgboost": xgb_metrics, "blend": blend, "recommendation": comparison["recommendation"], "gpu2_released_for_production": True})
    B003.atomic_json(args.output_dir / "manifest.json", manifest); print("\nGPU2_DIVERSE STOP", flush=True); print(json.dumps(comparison, indent=2), flush=True); heartbeat.stop(); return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True); parser.add_argument("--b005-metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True); parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/GPU2_DIVERSE"))
    parser.add_argument("--deadline-minutes", type=float, default=45.0); parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--stage-reserve-seconds", type=float, default=180.0); parser.add_argument("--second-catboost-minimum-seconds", type=float, default=1800.0)
    parser.add_argument("--xgboost-minimum-seconds", type=float, default=900.0); parser.add_argument("--try-second-catboost", action="store_true"); parser.add_argument("--run-xgboost", action="store_true")
    parser.add_argument("--b007-calibration-predictions", type=Path); parser.add_argument("--b007-evaluation-predictions", type=Path)
    parser.add_argument("--b007-probability-column", default="b007_probability")
    args = parser.parse_args(argv)
    if args.deadline_minutes <= 0 or args.heartbeat_seconds <= 0: parser.error("deadline and heartbeat must be positive")
    if bool(args.b007_calibration_predictions) != bool(args.b007_evaluation_predictions): parser.error("supply both B007 prediction files or neither")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeadlineReached as exc:
        parsed = parse_args()
        parsed.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = parsed.output_dir / "manifest.json"
        manifest = B003.read_json(manifest_path) if manifest_path.exists() else {"experiment_id": EXPERIMENT_ID}
        manifest.update({
            "status": "STOPPED_DEADLINE", "stopped_at": B003.utc_now(), "reason": str(exc),
            "gpu2_released_for_production": True,
            "recommendation": "Use the best completed artifact only; do not count unfinished experiments.",
        })
        B003.atomic_json(manifest_path, manifest)
        print(f"GPU2_DIVERSE STOPPED AT DEADLINE: {exc}", flush=True)
        raise SystemExit(0)
