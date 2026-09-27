#!/usr/bin/env python3
"""B006: add cached retrieval provenance/rank/score features to B005.

This runner never generates candidates or pairwise string features. It consumes the
completed B005 feature Parquets verbatim, validates their split and retrieval
invariants, appends retrieval-only features, and retrains/calibrates/evaluates.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "B006"
B005_MACRO_F05 = 0.926948
EXPECTED_RECALL = 0.9746268657
EXPECTED_ORACLE = 0.9910711081
EXPECTED_EVAL_PAIRS = 875_027
RRF_CONSTANT = 60.0
BASE_PREFIX = "f_"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_previous() -> tuple[Any, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "experiments" / "B003"))
    import run_b003 as b003  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B005"))
    import run_b005 as b005  # type: ignore
    return b003, b005


B003, B005 = import_previous()
BASE_FEATURES = list(B003.FEATURE_NAMES)
PROVENANCE_COLUMNS = [
    "found_by_existing", "found_by_name_char", "found_by_address_word", "found_by_name_word",
    "existing_rank", "name_char_rank", "address_word_rank", "name_word_rank",
    "existing_shared_keys", "name_char_score", "address_word_score", "name_word_score",
    "retriever_count", "rrf_score",
]
RETRIEVAL_FEATURES = [
    "found_R0", "found_R1", "found_R2", "found_R3", "retriever_count",
    "R0_R1", "R0_R2", "R0_R3", "R1_R2", "R1_R3", "R2_R3",
    "retrieved_by_3plus", "retrieved_by_all4",
    "R0_rank", "R1_rank", "R2_rank", "R3_rank",
    "best_rank", "mean_available_rank", "worst_available_rank",
    "R0_shared_keys", "R1_name_char_score", "R2_address_word_score", "R3_name_word_score",
    "reciprocal_rank_R0", "reciprocal_rank_R1", "reciprocal_rank_R2", "reciprocal_rank_R3",
    "reciprocal_rank_sum",
    "R0_best_shared_keys", "R0_shared_keys_gap_from_best",
    "R1_name_char_best_score", "R1_name_char_gap_from_best",
    "R2_address_word_best_score", "R2_address_word_gap_from_best",
    "R3_name_word_best_score", "R3_name_word_gap_from_best",
]
ALL_FEATURES = BASE_FEATURES + RETRIEVAL_FEATURES


class Heartbeat:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds; self.started = time.monotonic(); self.stage = "initializing"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="b006-heartbeat")

    def start(self) -> None:
        self.thread.start()

    def update(self, stage: str) -> None:
        self.stage = stage

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)

    def _run(self) -> None:
        while not self.stop_event.wait(self.seconds):
            print(
                f"[{B003.utc_now()}] B006 HEARTBEAT | stage={self.stage} | "
                f"elapsed={time.monotonic() - self.started:.0f}s",
                flush=True,
            )


def feature_paths(b005_dir: Path, split: str) -> list[Path]:
    paths = sorted((b005_dir / "features" / split).glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No cached B005 {split} feature Parquets under {b005_dir}")
    return paths


def canonical_pair_key(query_id: Any, candidate_id: Any) -> tuple[str, str]:
    return str(query_id), str(candidate_id)


def register_pair_key(seen: set[tuple[str, str]], query_id: Any, candidate_id: Any) -> tuple[str, str]:
    key = canonical_pair_key(query_id, candidate_id)
    if key in seen:
        raise ValueError(f"Duplicate cached B005 candidate pair: {key}")
    seen.add(key)
    return key


def channel_best_maps(paths: Sequence[Path]) -> tuple[dict[str, dict[str, float]], set[str], int]:
    """First pass: validate pair keys and find full query-relative channel maxima."""
    import pyarrow.parquet as pq  # type: ignore
    columns = ["query_id", "candidate_id"] + PROVENANCE_COLUMNS
    best: dict[str, dict[str, float]] = collections.defaultdict(dict)
    query_ids: set[str] = set(); keys: set[tuple[str, str]] = set(); rows = 0
    score_fields = {
        "R0": "existing_shared_keys", "R1": "name_char_score",
        "R2": "address_word_score", "R3": "name_word_score",
    }
    found_fields = {
        "R0": "found_by_existing", "R1": "found_by_name_char",
        "R2": "found_by_address_word", "R3": "found_by_name_word",
    }
    for path in paths:
        table = pq.read_table(path, columns=columns)
        data = table.to_pydict()
        for index, query_value in enumerate(data["query_id"]):
            query_id = str(query_value); candidate_id = str(data["candidate_id"][index])
            key = register_pair_key(keys, query_id, candidate_id)
            query_ids.add(query_id); rows += 1
            for channel in ("R0", "R1", "R2", "R3"):
                if bool(data[found_fields[channel]][index]):
                    raw = data[score_fields[channel]][index]
                    if raw is None:
                        raise RuntimeError(f"{channel} found flag has missing strength for {key}")
                    score = float(raw)
                    best[query_id][channel] = max(best[query_id].get(channel, -np.inf), score)
    return dict(best), query_ids, rows


def retrieval_pattern(row: Mapping[str, Any]) -> str:
    active = [
        channel for channel, field in (
            ("R0", "found_by_existing"), ("R1", "found_by_name_char"),
            ("R2", "found_by_address_word"), ("R3", "found_by_name_word"),
        ) if bool(row.get(field))
    ]
    return f"{active[0]}-only" if len(active) == 1 else "multi-channel"


def encode_rank(rank: Any, sentinel: int) -> float:
    return float(sentinel if rank is None else int(rank))


def reciprocal(rank: Any) -> float:
    return 0.0 if rank is None else 1.0 / (RRF_CONSTANT + float(rank))


def retrieval_feature_values(row: Mapping[str, Any], best: Mapping[str, float]) -> list[float]:
    flags = [
        int(bool(row.get("found_by_existing"))), int(bool(row.get("found_by_name_char"))),
        int(bool(row.get("found_by_address_word"))), int(bool(row.get("found_by_name_word"))),
    ]
    raw_ranks = [row.get("existing_rank"), row.get("name_char_rank"), row.get("address_word_rank"), row.get("name_word_rank")]
    if int(row["retriever_count"]) != sum(flags):
        raise RuntimeError("Cached retriever_count disagrees with R0-R3 found flags")
    for channel, flag, rank, maximum in zip(("R0", "R1", "R2", "R3"), flags, raw_ranks, (50, 30, 30, 30)):
        if bool(flag) != (rank is not None):
            raise RuntimeError(f"Cached {channel} found flag/rank state is inconsistent")
        if rank is not None and not 1 <= int(rank) <= maximum:
            raise RuntimeError(f"Cached {channel} rank is outside 1..{maximum}: {rank}")
    ranks = [encode_rank(raw_ranks[0], 51)] + [encode_rank(value, 31) for value in raw_ranks[1:]]
    available = [float(value) for value in raw_ranks if value is not None]
    rank_aggregates = (
        [min(available), float(np.mean(available)), max(available)]
        if available else [51.0, 51.0, 51.0]
    )
    raw_scores = [row.get("existing_shared_keys"), row.get("name_char_score"), row.get("address_word_score"), row.get("name_word_score")]
    scores = [0.0 if value is None else float(value) for value in raw_scores]
    reciprocal_ranks = [reciprocal(value) for value in raw_ranks]
    expected_rrf = sum(reciprocal_ranks)
    if abs(expected_rrf - float(row.get("rrf_score") or 0.0)) > 1e-9:
        raise RuntimeError("Cached B004 RRF differs from fixed 1/(60+rank) formula")
    relative = []
    for channel, score, flag in zip(("R0", "R1", "R2", "R3"), scores, flags):
        channel_best = float(best.get(channel, 0.0))
        relative.extend((channel_best, score - channel_best if flag else 0.0))
    interactions = [
        flags[0] * flags[1], flags[0] * flags[2], flags[0] * flags[3],
        flags[1] * flags[2], flags[1] * flags[3], flags[2] * flags[3],
        int(sum(flags) >= 3), int(sum(flags) == 4),
    ]
    values = (
        [*flags, float(row["retriever_count"]), *interactions]
        + ranks + rank_aggregates + scores + reciprocal_ranks + [expected_rrf] + relative
    )
    if len(values) != len(RETRIEVAL_FEATURES):
        raise AssertionError((len(values), len(RETRIEVAL_FEATURES)))
    return [float(value) for value in values]


def load_split(
    paths: Sequence[Path],
    best_by_query: Mapping[str, Mapping[str, float]],
    collect_evidence: bool,
    telemetry: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], dict[tuple[str, str], dict[str, Any]]]:
    import pyarrow.parquet as pq  # type: ignore
    columns = ["query_id", "candidate_id", "country", "label"] + PROVENANCE_COLUMNS + [BASE_PREFIX + name for name in BASE_FEATURES]
    matrices, labels, query_ids, candidate_ids, countries = [], [], [], [], []
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for index, path in enumerate(paths, start=1):
        table = pq.read_table(path, columns=columns); data = table.to_pydict(); count = table.num_rows
        base = np.column_stack([table[BASE_PREFIX + name].to_numpy() for name in BASE_FEATURES]).astype(np.float32)
        extra = np.empty((count, len(RETRIEVAL_FEATURES)), dtype=np.float32)
        for row_index in range(count):
            row = {name: data[name][row_index] for name in PROVENANCE_COLUMNS}
            query_id = str(data["query_id"][row_index]); candidate_id = str(data["candidate_id"][row_index])
            extra[row_index] = retrieval_feature_values(row, best_by_query[query_id])
            query_ids.append(query_id); candidate_ids.append(candidate_id); countries.append(str(data["country"][row_index]))
            if collect_evidence:
                evidence[(query_id, candidate_id)] = row
        matrices.append(np.hstack((base, extra)))
        labels.append(table["label"].to_numpy().astype(np.int8))
        telemetry.record("cache_file_loaded", candidate_pairs=count, extra={"file": path.name, "file_index": index, "file_count": len(paths)})
    return np.vstack(matrices), np.concatenate(labels), query_ids, candidate_ids, countries, evidence


def score_dict(
    model: Any, matrix: np.ndarray, query_ids: Sequence[str], candidate_ids: Sequence[str], all_query_ids: Iterable[str]
) -> dict[str, list[tuple[str, float]]]:
    probabilities = model.predict_proba(matrix)
    result: dict[str, list[tuple[str, float]]] = {query_id: [] for query_id in all_query_ids}
    for query_id, candidate_id, probability in zip(query_ids, candidate_ids, probabilities):
        result[query_id].append((candidate_id, float(probability)))
    return result


def fit_model(args: argparse.Namespace, production: Mapping[str, Any], params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, telemetry: Any) -> tuple[Any, str, dict[str, Any]]:
    frozen = dict(params); frozen.pop("device_type", None)
    devices = ["gpu", "cpu"] if args.device == "auto" else [args.device]
    last_error = None
    for device in devices:
        try:
            current = dict(frozen); current["device_type"] = device
            telemetry.record("model_fit_start", candidate_pairs=len(y), extra={"device": device, "features": X.shape[1], "positives": int(y.sum())})
            model = production["EntityMatcherModel"]("lightgbm", **current); model.fit(X, y)
            telemetry.record("model_fit_done", candidate_pairs=len(y), extra={"device": device})
            return model, device, current
        except Exception as exc:
            last_error = repr(exc); telemetry.record("model_fit_failed", extra={"device": device, "error": last_error})
            if device != "gpu" or "cpu" not in devices:
                raise
    raise RuntimeError(last_error or "Model fit failed")


def importance_report(model: Any) -> dict[str, Any]:
    booster = model.model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    rows = [
        {"feature": name, "gain": float(gain), "split_count": int(split)}
        for name, gain, split in zip(ALL_FEATURES, gains, splits)
    ]
    ranked = sorted(rows, key=lambda row: (-row["gain"], -row["split_count"], row["feature"]))
    for rank, row in enumerate(ranked, start=1):
        row["gain_rank"] = rank
    requested = {
        name: next(row for row in ranked if row["feature"] == name)
        for name in ("found_R2", "retriever_count", "R2_rank", "R2_address_word_score", "R2_address_word_gap_from_best", "reciprocal_rank_sum")
    }
    return {"top_25": ranked[:25], "all": ranked, "requested_feature_ranks": requested}


def source_metrics(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], prefix: str) -> dict[str, Any]:
    tp = fp = fn = 0
    for query_id, expected_all in truth.items():
        expected = {value for value in expected_all if value.startswith(prefix)}
        actual = {value for value in predictions.get(query_id, set()) if value.startswith(prefix)}
        tp += len(expected & actual); fp += len(actual - expected); fn += len(expected - actual)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def error_slices(
    truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]],
    evidence: Mapping[tuple[str, str], Mapping[str, Any]], evaluate: Any,
) -> dict[str, Any]:
    fn = collections.Counter(); fp = collections.Counter()
    patterns = ("R0-only", "R1-only", "R2-only", "R3-only", "multi-channel")
    for query_id, expected in truth.items():
        for target_id in expected - predictions.get(query_id, set()):
            row = evidence.get((query_id, target_id))
            fn["GT not retrieved" if row is None else retrieval_pattern(row)] += 1
        for target_id in predictions.get(query_id, set()) - expected:
            fp[retrieval_pattern(evidence[(query_id, target_id)])] += 1
    slices = {}
    for pattern in patterns:
        sliced_truth = {query_id: set() for query_id in truth}; sliced_predictions = {query_id: set() for query_id in truth}
        for (query_id, target_id), row in evidence.items():
            if retrieval_pattern(row) == pattern:
                if target_id in truth[query_id]: sliced_truth[query_id].add(target_id)
                if target_id in predictions.get(query_id, set()): sliced_predictions[query_id].add(target_id)
        metrics = dict(evaluate(sliced_truth, sliced_predictions))
        slices[pattern] = {"candidate_restricted": True, **metrics}
    return {"false_negatives": dict(fn), "false_positives": dict(fp), "retrieval_pattern_slices": slices}


def validate_cache(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = args.b005_artifacts / "manifest.json"
    if not manifest_path.exists() or not args.b005_metadata.exists():
        raise RuntimeError("B006 requires B005 manifest.json, features/, and model metadata.json")
    manifest = B003.read_json(manifest_path); metadata = B003.read_json(args.b005_metadata)
    verification = manifest.get("frozen_b004_verification", {})
    if manifest.get("status") != "COMPLETE" or verification.get("status") != "PASS":
        raise RuntimeError("B005 cache is not marked COMPLETE with frozen retrieval PASS")
    if abs(float(verification.get("recall", 0)) - EXPECTED_RECALL) > 5e-4 or abs(float(verification.get("oracle_f05", 0)) - EXPECTED_ORACLE) > 5e-4 or int(verification.get("pairs", 0)) != EXPECTED_EVAL_PAIRS:
        raise RuntimeError("B005 cached retrieval does not match frozen B004 expectations")
    if metadata.get("feature_count") != 32 or metadata.get("feature_names") != BASE_FEATURES:
        raise RuntimeError("B005 metadata does not preserve the production 32-feature schema")
    return manifest, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); root = repo_root(); os.chdir(root)
    for path in (args.b005_artifacts, args.b005_metadata, args.ground_truth, args.val_ids):
        if not path.exists(): raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True); args.model_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = Heartbeat(args.heartbeat_seconds); heartbeat.start(); atexit.register(heartbeat.stop)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl"); original_record = telemetry.record
    def progress(event: str, **kwargs: Any) -> None:
        heartbeat.update(event); print(f"[{B003.utc_now()}] B006 {event} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True); original_record(event, **kwargs)
    telemetry.record = progress; telemetry.record("start")
    production = B003.import_production(root); b005_manifest, b005_metadata = validate_cache(args)

    split_info: dict[str, dict[str, Any]] = {}
    for split in ("train", "calibration", "evaluation"):
        paths = feature_paths(args.b005_artifacts, split)
        best, ids, rows = channel_best_maps(paths)
        expected_count = int(b005_manifest[split]["s1_count"])
        if len(ids) != expected_count:
            raise RuntimeError(f"B005 {split} cache has {len(ids)} query IDs, expected {expected_count}")
        split_info[split] = {"paths": paths, "best": best, "ids": ids, "rows": rows}
        telemetry.record("cache_split_validated", processed_s1=len(ids), candidate_pairs=rows, extra={"split": split})
    train_ids = split_info["train"]["ids"]; calibration_ids = split_info["calibration"]["ids"]; eval_ids = split_info["evaluation"]["ids"]
    if train_ids & calibration_ids or train_ids & eval_ids or calibration_ids & eval_ids:
        raise RuntimeError("B005 cached train/calibration/evaluation query IDs overlap")
    frozen_ids, _ = B003.load_selected_ids(args.val_ids, 5000)
    if set(frozen_ids) != eval_ids:
        raise RuntimeError("B005 evaluation cache does not equal the frozen 5,000 IDs")
    all_ids = sorted(train_ids | calibration_ids | eval_ids)
    truth = B003.load_ground_truth(args.ground_truth, all_ids)

    schema = {
        "schema_version": 1, "base_feature_count": 32, "retrieval_feature_count": len(RETRIEVAL_FEATURES),
        "final_feature_count": len(ALL_FEATURES), "base_features": BASE_FEATURES,
        "retrieval_features": RETRIEVAL_FEATURES, "all_features": ALL_FEATURES,
        "missing_representation": {
            "R0_rank": "51 (R0 K50 + 1)", "R1_R2_R3_rank": "31 (per-source K30 + 1)",
            "scores": "0.0 plus the corresponding found_R* flag as explicit missing indicator",
            "score_gap": "0.0 when channel missing; found_R* disambiguates missing from an exact tie",
        },
        "rrf_formula": "0 if missing else 1 / (60 + rank); sum across R0-R3",
        "query_relative_best": "maximum retrieval strength among cached B005 rows for the query; B005 mining retained every per-channel strongest negative and all positives",
    }
    B003.atomic_json(args.output_dir / "feature_schema.json", schema)
    manifest = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "RUNNING", "created_at": B003.utc_now(),
        "git_branch": B003.git_value(root, "branch", "--show-current"), "git_commit": B003.git_value(root, "rev-parse", "HEAD"),
        "python_version": sys.version, "platform": platform.platform(),
        "cache_only": True, "b005_artifacts": str(args.b005_artifacts.resolve()), "b005_metadata": str(args.b005_metadata.resolve()),
        "b005_manifest_sha256": B003.sha256_file(args.b005_artifacts / "manifest.json"),
        "b005_metadata_sha256": B003.sha256_file(args.b005_metadata),
        "cached_rows": {split: split_info[split]["rows"] for split in split_info},
        "cached_query_counts": {split: len(split_info[split]["ids"]) for split in split_info},
        "frozen_retrieval": b005_manifest["frozen_b004_verification"], "feature_count": len(ALL_FEATURES),
    }
    B003.atomic_json(args.output_dir / "manifest.json", manifest)

    X_train, y_train, train_queries, train_candidates, _train_countries, _ = load_split(split_info["train"]["paths"], split_info["train"]["best"], False, telemetry)
    if len(y_train) != 2_401_548 or int(y_train.sum()) != 151_548:
        raise RuntimeError(f"B006 training rows/positives changed from measured B005: rows={len(y_train)}, positives={int(y_train.sum())}")
    if any(int(label) != int(candidate_id in truth[query_id]) for label, query_id, candidate_id in zip(y_train, train_queries, train_candidates)):
        raise RuntimeError("Cached B005 training labels do not match ground truth")
    del train_queries, train_candidates
    model, device, model_params = fit_model(args, production, b005_metadata["model_params"], X_train, y_train, telemetry)
    del X_train, y_train
    model_path = args.model_dir / "lightgbm.joblib"; model.save(str(model_path))

    X_cal, y_cal, cal_queries, cal_candidates, _cal_countries, _ = load_split(split_info["calibration"]["paths"], split_info["calibration"]["best"], False, telemetry)
    if any(int(label) != int(candidate_id in truth[query_id]) for label, query_id, candidate_id in zip(y_cal, cal_queries, cal_candidates)):
        raise RuntimeError("Cached B005 calibration labels do not match ground truth")
    del y_cal
    calibration_scores = score_dict(model, X_cal, cal_queries, cal_candidates, calibration_ids); del X_cal
    calibration_truth = {query_id: truth[query_id] for query_id in calibration_ids}
    t2, t3, calibration_metrics = B005.tune_thresholds(calibration_truth, calibration_scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
    calibration_metrics.update({"threshold_s2": t2, "threshold_s3": t3, "query_count": len(calibration_ids)})
    B003.atomic_json(args.output_dir / "calibration_metrics.json", calibration_metrics)

    X_eval, y_eval, eval_queries, eval_candidates, eval_countries, evidence = load_split(split_info["evaluation"]["paths"], split_info["evaluation"]["best"], True, telemetry)
    if len(eval_candidates) != EXPECTED_EVAL_PAIRS:
        raise RuntimeError("Frozen evaluation candidate-pair count changed")
    if any(int(label) != int(candidate_id in truth[query_id]) for label, query_id, candidate_id in zip(y_eval, eval_queries, eval_candidates)):
        raise RuntimeError("Cached B005 evaluation labels do not match ground truth")
    del y_eval
    eval_truth = {query_id: truth[query_id] for query_id in eval_ids}
    oracle = {query_id: set() for query_id in eval_ids}
    for query_id, candidate_id in evidence:
        if candidate_id in eval_truth[query_id]:
            oracle[query_id].add(candidate_id)
    oracle_metrics = production["evaluate_predictions"](eval_truth, oracle)
    if abs(float(oracle_metrics["macro_f05"]) - EXPECTED_ORACLE) > 5e-4:
        raise RuntimeError("Candidate-restricted oracle changed")
    evaluation_scores = score_dict(model, X_eval, eval_queries, eval_candidates, eval_ids); del X_eval
    predictions = production["apply_threshold_and_deduplication"](evaluation_scores, t2, t3)
    overall = dict(production["evaluate_predictions"](eval_truth, predictions))
    countries_by_query = {query_id: country for query_id, country in zip(eval_queries, eval_countries)}
    country_metrics = {
        country: production["evaluate_predictions"](
            {q: eval_truth[q] for q in eval_ids if countries_by_query[q] == country},
            {q: predictions.get(q, set()) for q in eval_ids if countries_by_query[q] == country},
        ) for country in sorted(set(countries_by_query.values()))
    }
    delta = float(overall["macro_f05"]) - B005_MACRO_F05
    evaluation_metrics = {
        "overall": overall, "country": country_metrics,
        "S2": source_metrics(eval_truth, predictions, "S2-"), "S3": source_metrics(eval_truth, predictions, "S3-"),
        "threshold_s2": t2, "threshold_s3": t3, "candidate_pairs": len(eval_candidates),
        "candidate_recall": EXPECTED_RECALL, "candidate_oracle_f05": float(oracle_metrics["macro_f05"]),
        "B005_macro_f05": B005_MACRO_F05, "B006_macro_f05": float(overall["macro_f05"]),
        "absolute_delta": delta, "remaining_gap_to_oracle": EXPECTED_ORACLE - float(overall["macro_f05"]),
        "acceptance": "PASS" if delta > 0 else "FAIL",
    }
    B003.atomic_json(args.output_dir / "evaluation_metrics.json", evaluation_metrics)
    B003.atomic_json(args.output_dir / "error_slices.json", error_slices(eval_truth, predictions, evidence, production["evaluate_predictions"]))
    importance = importance_report(model); B003.atomic_json(args.output_dir / "feature_importance.json", importance)
    metadata = {
        "experiment_id": EXPERIMENT_ID, "feature_names": ALL_FEATURES, "feature_count": len(ALL_FEATURES),
        "model_params": model_params, "device_used": device, "optimal_s2_threshold": t2, "optimal_s3_threshold": t3,
        "calibration_macro_f05": calibration_metrics["macro_f05"], "frozen_evaluation_macro_f05": overall["macro_f05"],
        "absolute_delta_vs_B005": delta,
    }
    B003.atomic_json(args.model_dir / "metadata.json", metadata)
    manifest.update({"status": "COMPLETE", "completed_at": B003.utc_now(), "device_used": device, "thresholds": {"S2": t2, "S3": t3}, "acceptance": evaluation_metrics["acceptance"]})
    B003.atomic_json(args.output_dir / "manifest.json", manifest); B003.atomic_text(args.output_dir / "B006.DONE", B003.utc_now() + "\n")
    print(f"\nB006 COMPLETE | Macro F0.5={overall['macro_f05']:.6f} | delta={delta:+.6f} | {evaluation_metrics['acceptance']}", flush=True)
    print(f"Precision={overall['global_precision']:.6f} Recall={overall['global_recall']:.6f} Oracle gap={evaluation_metrics['remaining_gap_to_oracle']:.6f}", flush=True)
    print("Top retrieval feature ranks:", json.dumps(importance["requested_feature_ranks"], indent=2), flush=True)
    heartbeat.stop(); return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True, help="Completed B005 artifacts directory containing manifest.json and features/")
    parser.add_argument("--b005-metadata", type=Path, required=True, help="B005 models/experiments/B005/metadata.json")
    parser.add_argument("--ground-truth", type=Path, required=True); parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B006")); parser.add_argument("--model-dir", type=Path, default=Path("models/experiments/B006"))
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto"); parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0: parser.error("--heartbeat-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
