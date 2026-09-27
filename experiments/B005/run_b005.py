#!/usr/bin/env python3
"""B005: retrain the frozen 32-feature matcher on B004 hard negatives.

The retriever is imported directly from B004 and is not reimplemented here.
Ground truth labels candidates only after retrieval. Unretrieved positive links are
reported but never inserted into the primary training matrix.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "B005"
EXPECTED_B004 = {"recall": 0.974627, "oracle_f05": 0.991071, "pairs": 875_027}
FEATURE_PREFIX = "f_"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_experiments() -> tuple[Any, Any]:
    root = repo_root()
    sys.path.insert(0, str(root / "experiments" / "B003"))
    import run_b003 as b003  # type: ignore
    sys.path.insert(0, str(root / "experiments" / "B004"))
    import run_b004 as b004  # type: ignore
    return b003, b004


B003, B004 = import_experiments()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def match_bucket(count: int) -> str:
    if count == 0:
        return "singleton"
    if count == 1:
        return "one_link"
    if count == 2:
        return "two_links"
    return "three_plus_links"


def select_training_ids(
    ground_truth_path: Path,
    validation_ids: set[str],
    train_limit: int,
) -> list[str]:
    """Preserve repository comparability: first N non-validation GT rows."""
    selected: list[str] = []
    for entity_id, _name, _address, _country in B003.iter_tsv(ground_truth_path):
        if entity_id not in validation_ids:
            selected.append(entity_id)
            if len(selected) == train_limit:
                break
    if len(selected) != train_limit:
        raise ValueError(f"Requested {train_limit:,} training S1 IDs, found {len(selected):,}")
    return selected


def deterministic_calibration_split(
    ids: Sequence[str],
    truth: Mapping[str, set[str]],
    countries: Mapping[str, str],
    fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    strata: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for entity_id in ids:
        strata[(countries[entity_id], match_bucket(len(truth[entity_id])))].append(entity_id)
    calibration: set[str] = set()
    for values in strata.values():
        ranked = sorted(values, key=lambda value: (stable_key(seed, value), value))
        count = int(round(len(ranked) * fraction))
        if fraction > 0 and len(ranked) > 1:
            count = max(1, min(count, len(ranked) - 1))
        calibration.update(ranked[:count])
    train = [value for value in ids if value not in calibration]
    calibrate = [value for value in ids if value in calibration]
    if set(train) & set(calibrate) or set(train) | set(calibrate) != set(ids):
        raise RuntimeError("Invalid train/calibration partition")
    return train, calibrate


def candidate_reason_sets(row: Mapping[str, Any]) -> set[str]:
    reasons = set()
    if row.get("found_by_existing"):
        reasons.add("R0")
    if row.get("found_by_name_char"):
        reasons.add("R1")
    if row.get("found_by_address_word"):
        reasons.update(("R2", "near_address"))
    if row.get("found_by_name_word"):
        reasons.update(("R3", "near_name"))
    if row.get("found_by_name_char"):
        reasons.add("near_name")
    if int(row.get("retriever_count", 0)) >= 2:
        reasons.add("multi_channel")
    reasons.add("top_rrf")
    return reasons


def negative_sort_key(row: Mapping[str, Any], reason: str) -> tuple[Any, ...]:
    if reason == "top_rrf":
        return (-float(row.get("rrf_score", 0.0)), str(row["candidate_id"]))
    rank_field = {
        "R0": "existing_rank", "R1": "name_char_rank", "R2": "address_word_rank",
        "R3": "name_word_rank", "near_address": "address_word_rank",
    }.get(reason)
    if reason == "near_name":
        ranks = [row.get("name_char_rank"), row.get("name_word_rank")]
        rank = min((int(v) for v in ranks if v is not None), default=10**9)
    elif rank_field:
        rank = int(row[rank_field]) if row.get(rank_field) is not None else 10**9
    else:
        rank = -int(row.get("retriever_count", 0))
    return (rank, -float(row.get("rrf_score", 0.0)), str(row["candidate_id"]))


def select_hard_negatives(
    negatives: Sequence[Mapping[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Deterministic round-robin union retaining every requested hard category."""
    if budget <= 0:
        return []
    categories = ("multi_channel", "R0", "R2", "R1", "R3", "near_address", "near_name", "top_rrf")
    queues: dict[str, list[Mapping[str, Any]]] = {
        reason: sorted(
            [row for row in negatives if reason in candidate_reason_sets(row)],
            key=lambda row, current=reason: negative_sort_key(row, current),
        )
        for reason in categories
    }
    chosen: dict[str, dict[str, Any]] = {}
    offsets = collections.Counter()
    while len(chosen) < min(budget, len(negatives)):
        progressed = False
        for reason in categories:
            queue = queues[reason]
            while offsets[reason] < len(queue):
                row = queue[offsets[reason]]
                offsets[reason] += 1
                candidate_id = str(row["candidate_id"])
                if candidate_id in chosen:
                    continue
                selected = dict(row)
                selected["selection_reasons"] = sorted(candidate_reason_sets(row))
                chosen[candidate_id] = selected
                progressed = True
                break
            if len(chosen) >= min(budget, len(negatives)):
                break
        if not progressed:
            break
    return list(chosen.values())


def partition_candidates(
    candidates: Sequence[Mapping[str, Any]], true_targets: set[str]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], set[str]]:
    """Label only retrieved candidates and return unreachable GT separately."""
    positives = [row for row in candidates if str(row["candidate_id"]) in true_targets]
    negatives = [row for row in candidates if str(row["candidate_id"]) not in true_targets]
    retrieved_ids = {str(row["candidate_id"]) for row in candidates}
    return positives, negatives, true_targets - retrieved_ids


def tune_thresholds(
    truth: Mapping[str, set[str]],
    scores: Mapping[str, Sequence[tuple[str, float]]],
    apply_uniqueness: Any,
    evaluate_predictions: Any,
) -> tuple[float, float, dict[str, Any]]:
    """Coordinate coarse-to-fine search; every trial applies target uniqueness."""
    cache: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate(t2: float, t3: float) -> dict[str, Any]:
        key = (round(float(t2), 6), round(float(t3), 6))
        if key not in cache:
            predictions = apply_uniqueness(scores, key[0], key[1])
            cache[key] = dict(evaluate_predictions(truth, predictions))
        return cache[key]

    def best_axis(values: Iterable[float], fixed: float, source: str) -> tuple[float, dict[str, Any]]:
        candidates = []
        for value in values:
            t2, t3 = (value, fixed) if source == "S2" else (fixed, value)
            metrics = evaluate(float(t2), float(t3))
            candidates.append((float(metrics["macro_f05"]), -float(value), float(value), metrics))
        _score, _tie, value, metrics = max(candidates, key=lambda item: (item[0], item[1]))
        return value, metrics

    coarse = np.arange(0.05, 0.951, 0.05)
    t2, _ = best_axis(coarse, 0.5, "S2")
    t3, _ = best_axis(coarse, t2, "S3")
    t2, _ = best_axis(np.arange(max(0, t2 - 0.05), min(1, t2 + 0.05) + 0.001, 0.01), t3, "S2")
    t3, metrics = best_axis(np.arange(max(0, t3 - 0.05), min(1, t3 + 0.05) + 0.001, 0.01), t2, "S3")
    metrics = dict(metrics)
    metrics["threshold_trials"] = len(cache)
    return round(t2, 6), round(t3, 6), metrics


def preprocess_record(name: str, address: str, norm: Any) -> tuple[str, str, str, set[str]]:
    clean_name, core_name, _ = norm.normalize_name(name)
    clean_address, numbers, _ = norm.normalize_address(address)
    return clean_name, core_name, clean_address, numbers


def retrieval_chunk_paths(
    retrieval_dir: Path, country: str, chunk_index: int
) -> list[Path]:
    paths = [
        retrieval_dir / "channels" / "R0_EXISTING" / country / "S2_S3" / f"chunk_{chunk_index:05d}.parquet"
    ]
    for channel in ("R1_NAME_CHAR", "R2_ADDRESS_WORD", "R3_NAME_WORD"):
        for source in ("S2", "S3"):
            paths.append(retrieval_dir / "channels" / channel / country / source / f"chunk_{chunk_index:05d}.parquet")
    return paths


def load_merged_chunk(
    retrieval_dir: Path, country: str, chunk_index: int, rrf_constant: float
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = []
    for path in retrieval_chunk_paths(retrieval_dir, country, chunk_index):
        if not path.exists() or not path.with_suffix(".DONE").exists():
            raise RuntimeError(f"Incomplete B004 retrieval checkpoint: {path}")
        rows.extend(B003.read_parquet_rows(path))
    return B004.merge_provenance(rows, rrf_constant)


def write_feature_checkpoint(
    path: Path,
    query_ids: Sequence[str],
    split_by_id: Mapping[str, str],
    truth: Mapping[str, set[str]],
    query_records: Mapping[str, tuple[str, str, str]],
    target_preprocessed: Mapping[str, tuple[str, str, str, set[str]]],
    merged: Mapping[tuple[str, str], Mapping[str, Any]],
    production: Mapping[str, Any],
    max_negatives: int,
) -> dict[str, Any]:
    norm = production["norm"]
    extract = production["extract_features_for_pair"]
    rows_by_query: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for (query_id, _candidate_id), evidence in merged.items():
        rows_by_query[query_id].append(evidence)
    output_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "queries": len(query_ids), "candidate_pairs": 0, "retrieved_positives": 0,
        "candidate_negatives": 0, "selected_negatives": 0, "missed_positives": 0,
        "negative_counts": [], "reason_counts": collections.Counter(),
        "retriever_count": collections.Counter(),
    }
    for query_id in query_ids:
        split = split_by_id[query_id]
        candidates = rows_by_query.get(query_id, [])
        positives, negatives, missed_positives = partition_candidates(candidates, truth[query_id])
        selected_negatives = negatives if split != "train" else select_hard_negatives(negatives, max_negatives)
        selected = positives + selected_negatives
        stats["candidate_pairs"] += len(candidates)
        stats["retrieved_positives"] += len(positives)
        stats["candidate_negatives"] += len(negatives)
        stats["selected_negatives"] += len(selected_negatives)
        stats["missed_positives"] += len(missed_positives)
        stats["negative_counts"].append(len(selected_negatives))
        query_tuple = preprocess_record(query_records[query_id][0], query_records[query_id][1], norm)
        for evidence in selected:
            target_id = str(evidence["candidate_id"])
            if target_id not in target_preprocessed:
                raise KeyError(f"Candidate target missing from country population: {target_id}")
            label = int(target_id in truth[query_id])
            shared_keys = int(evidence.get("existing_shared_keys") or 0)
            features = extract(query_tuple, target_preprocessed[target_id], target_id, shared_keys)
            if len(features) != len(B003.FEATURE_NAMES):
                raise RuntimeError("Production feature schema is no longer 32 columns")
            row = {
                "query_id": query_id, "candidate_id": target_id, "split": split,
                "country": query_records[query_id][2], "label": label,
                **{name: evidence.get(name) for name in B004.PROVENANCE_FIELDS},
            }
            row.update({FEATURE_PREFIX + name: float(value) for name, value in zip(B003.FEATURE_NAMES, features)})
            if not label:
                reasons = evidence.get("selection_reasons", candidate_reason_sets(evidence))
                for reason in reasons:
                    stats["reason_counts"][reason] += 1
                stats["retriever_count"][str(evidence["retriever_count"])] += 1
            output_rows.append(row)
    B003.write_parquet(path, output_rows)
    B004.atomic_done(path.with_suffix(".DONE"), {"rows": len(output_rows), "queries": len(query_ids)})
    stats["reason_counts"] = dict(stats["reason_counts"])
    stats["retriever_count"] = dict(stats["retriever_count"])
    B003.atomic_json(path.with_suffix(".metrics.json"), stats)
    return stats


def generate_retrieval(
    args: argparse.Namespace,
    all_ids: Sequence[str],
    records: Mapping[str, tuple[str, str, str]],
    country_to_ids: Mapping[str, Sequence[str]],
    production: Mapping[str, Any],
    telemetry: Any,
) -> None:
    retrieval_args = argparse.Namespace(**vars(args))
    retrieval_args.output_dir = args.output_dir / "retrieval"
    for country in sorted(country_to_ids, key=lambda value: len(country_to_ids[value])):
        query_ids = country_to_ids[country]
        B004.run_r0_country(
            retrieval_args, country, query_ids, records, production["norm"],
            production["get_blocking_keys"], telemetry,
        )
        for source, path in (("S2", args.source2), ("S3", args.source3)):
            B004.run_tfidf_source_country(
                retrieval_args, country, source, path, query_ids, records,
                production["norm"], telemetry,
            )


def combine_stats(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = collections.Counter()
    reasons = collections.Counter()
    retrievers = collections.Counter()
    negative_counts: list[int] = []
    for part in parts:
        for key in ("queries", "candidate_pairs", "retrieved_positives", "candidate_negatives", "selected_negatives", "missed_positives"):
            totals[key] += int(part[key])
        reasons.update(part["reason_counts"])
        retrievers.update(part["retriever_count"])
        negative_counts.extend(part["negative_counts"])
    array = np.asarray(negative_counts or [0])
    result = dict(totals)
    result.update({
        "negative_sampling_rate": totals["selected_negatives"] / totals["candidate_negatives"] if totals["candidate_negatives"] else 0.0,
        "average_negatives_per_s1": float(array.mean()),
        "p50_negatives_per_s1": float(np.percentile(array, 50)),
        "p95_negatives_per_s1": float(np.percentile(array, 95)),
        "max_negatives_per_s1": int(array.max()),
        "selection_reason_overlap_counts": dict(reasons),
        "retriever_count_distribution": dict(retrievers),
    })
    return result


def feature_paths(output_dir: Path, split: str | None = None) -> list[Path]:
    paths = sorted((output_dir / "features").glob("*/*.parquet"))
    if split is None:
        return paths
    return [path for path in paths if path.parent.name == split]


def load_matrix(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq  # type: ignore
    columns = [FEATURE_PREFIX + name for name in B003.FEATURE_NAMES] + ["label"]
    matrices, labels = [], []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        matrices.append(np.column_stack([table[column].to_numpy() for column in columns[:-1]]).astype(np.float32))
        labels.append(table["label"].to_numpy().astype(np.int8))
    if not matrices:
        raise RuntimeError("No feature checkpoints found")
    return np.vstack(matrices), np.concatenate(labels)


def fit_model(args: argparse.Namespace, production: Mapping[str, Any], telemetry: Any) -> tuple[Any, str, dict[str, Any]]:
    X, y = load_matrix(feature_paths(args.output_dir, "train"))
    params = {
        "objective": "binary", "metric": ["binary_logloss", "auc"],
        "n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves, "max_depth": args.max_depth,
        "subsample": 0.85, "colsample_bytree": 0.85, "min_child_samples": 30,
        "random_state": args.seed, "n_jobs": -1, "verbose": -1,
    }
    devices = ["gpu", "cpu"] if args.device == "auto" else [args.device]
    error = None
    for device in devices:
        try:
            current = dict(params)
            current["device_type"] = device
            telemetry.record("model_fit_start", extra={"device": device, "rows": len(y), "positives": int(y.sum())})
            model = production["EntityMatcherModel"]("lightgbm", **current)
            model.fit(X, y)
            telemetry.record("model_fit_done", extra={"device": device})
            return model, device, current
        except Exception as exc:  # GPU availability differs across Kaggle images.
            error = repr(exc)
            telemetry.record("model_fit_failed", extra={"device": device, "error": error})
            if device != "gpu" or "cpu" not in devices:
                raise
    raise RuntimeError(error or "LightGBM training failed")


def score_paths(model: Any, paths: Sequence[Path], query_ids: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    import pyarrow.parquet as pq  # type: ignore
    scores: dict[str, list[tuple[str, float]]] = {query_id: [] for query_id in query_ids}
    feature_columns = [FEATURE_PREFIX + name for name in B003.FEATURE_NAMES]
    for path in paths:
        table = pq.read_table(path, columns=["query_id", "candidate_id"] + feature_columns)
        matrix = np.column_stack([table[column].to_numpy() for column in feature_columns]).astype(np.float32)
        probabilities = model.predict_proba(matrix)
        for query_id, target_id, probability in zip(table["query_id"].to_pylist(), table["candidate_id"].to_pylist(), probabilities):
            scores[str(query_id)].append((str(target_id), float(probability)))
    return scores


def subset_metrics(ids: Sequence[str], truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], evaluate: Any) -> dict[str, Any]:
    selected_truth = {entity_id: truth[entity_id] for entity_id in ids}
    selected_predictions = {entity_id: predictions.get(entity_id, set()) for entity_id in ids}
    return dict(evaluate(selected_truth, selected_predictions))


def source_metrics(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], prefix: str) -> dict[str, Any]:
    tp = fp = fn = 0
    for query_id, expected in truth.items():
        actual = predictions.get(query_id, set())
        expected = {value for value in expected if value.startswith(prefix)}
        actual = {value for value in actual if value.startswith(prefix)}
        tp += len(expected & actual); fp += len(actual - expected); fn += len(expected - actual)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def verify_b004(
    args: argparse.Namespace,
    eval_ids: Sequence[str],
    truth: Mapping[str, set[str]],
    records: Mapping[str, tuple[str, str, str]],
    country_to_ids: Mapping[str, Sequence[str]],
    evaluate_predictions: Any,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    evidence_by_query = {query_id: [] for query_id in eval_ids}
    if args.b004_eval_dir:
        b004_manifest_path = args.b004_eval_dir / "manifest.json"
        union_path = args.b004_eval_dir / "union" / "uncapped_provenance.parquet"
        if not b004_manifest_path.exists() or not union_path.exists():
            raise RuntimeError("--b004-eval-dir must contain manifest.json and union/uncapped_provenance.parquet")
        b004_manifest = B003.read_json(b004_manifest_path)
        expected_hash = B003.sha256_ids(eval_ids)
        retrieval = b004_manifest.get("retrieval", {})
        cache_valid = (
            b004_manifest.get("status") == "COMPLETE"
            and b004_manifest.get("validation", {}).get("selected_sha256") == expected_hash
            and retrieval.get("R0", {}).get("k") == 50
            and retrieval.get("R0", {}).get("name_prune") == 300
            and retrieval.get("R0", {}).get("address_prune") == 150
            and retrieval.get("R1", {}).get("k_per_source") == 30
            and retrieval.get("R1", {}).get("analyzer") == "char_wb"
            and retrieval.get("R1", {}).get("ngram_range") == [3, 5]
            and retrieval.get("R1", {}).get("max_features") == 1_000_000
            and retrieval.get("R2", {}).get("k_per_source") == 30
            and retrieval.get("R2", {}).get("max_features") == 750_000
            and retrieval.get("R3", {}).get("k_per_source") == 30
            and retrieval.get("R3", {}).get("max_features") == 500_000
            and retrieval.get("tfidf_min_df") == 2
            and retrieval.get("tfidf_min_score") == 1e-6
        )
        if not cache_valid:
            raise RuntimeError("B004 cache manifest/hash/configuration does not match frozen B005 evaluation")
        selected = set(eval_ids)
        for evidence in B003.read_parquet_rows(union_path):
            query_id = str(evidence["query_id"])
            if query_id in selected:
                evidence_by_query[query_id].append(evidence)
        missing = [query_id for query_id in eval_ids if not evidence_by_query[query_id]]
        if missing:
            raise RuntimeError(f"B004 cache has no union candidates for {len(missing)} evaluation IDs")
    else:
        for country, all_country_ids in country_to_ids.items():
            for chunk_index, chunk_ids in B003.chunked(all_country_ids, args.query_chunk_size):
                if not any(query_id in evidence_by_query for query_id in chunk_ids):
                    continue
                merged = load_merged_chunk(args.output_dir / "retrieval", country, chunk_index, args.rrf_constant)
                for evidence in merged.values():
                    if evidence["query_id"] in evidence_by_query:
                        evidence_by_query[evidence["query_id"]].append(evidence)
    countries = {query_id: records[query_id][2] for query_id in eval_ids}
    result, _ranked = B004.evaluate_candidate_set(
        "R0+R1+R2+R3", B004.CHANNELS, evidence_by_query, eval_ids, truth,
        countries, evaluate_predictions, None, args.r0_guarantee,
    )
    observed = {
        "recall": float(result["gt_link_recall"]),
        "oracle_f05": float(result["candidate_restricted_oracle"]["macro_f05"]),
        "pairs": int(result["candidate_pairs"]),
    }
    observed["status"] = "PASS" if (
        abs(observed["recall"] - EXPECTED_B004["recall"]) <= args.retrieval_tolerance
        and abs(observed["oracle_f05"] - EXPECTED_B004["oracle_f05"]) <= args.retrieval_tolerance
        and observed["pairs"] == EXPECTED_B004["pairs"]
    ) else "FAIL"
    observed["expected"] = dict(EXPECTED_B004)
    observed["tolerance"] = args.retrieval_tolerance
    return observed, evidence_by_query


def error_analysis(
    truth: Mapping[str, set[str]],
    scores: Mapping[str, Sequence[tuple[str, float]]],
    predictions: Mapping[str, set[str]],
    evidence_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    t2: float,
    t3: float,
) -> dict[str, Any]:
    evidence = {(q, str(row["candidate_id"])): row for q, rows in evidence_by_query.items() for row in rows}
    score_lookup = {q: dict(values) for q, values in scores.items()}
    pre = {
        q: {tid for tid, score in values if score >= (t2 if tid.startswith("S2-") else t3)}
        for q, values in scores.items()
    }
    fn = collections.Counter(); fp = collections.Counter()
    for query_id, expected in truth.items():
        candidate_ids = set(score_lookup.get(query_id, {}))
        for target_id in expected - predictions.get(query_id, set()):
            if target_id not in candidate_ids:
                fn["not_retrieved"] += 1
            elif target_id not in pre.get(query_id, set()):
                fn["retrieved_but_rejected"] += 1
            else:
                fn["lost_by_uniqueness"] += 1
        for target_id in predictions.get(query_id, set()) - expected:
            row = evidence[(query_id, target_id)]
            if int(row["retriever_count"]) >= 2:
                fp["multi_channel"] += 1
            else:
                fp["single_channel"] += 1
            if row.get("found_by_address_word"):
                fp["R2_address_driven"] += 1
            if row.get("found_by_name_char") or row.get("found_by_name_word"):
                fp["R1_R3_name_driven"] += 1
    fn["total"] = sum(fn.values()); fp["total_false_positives"] = sum(1 for q in truth for tid in predictions.get(q, set()) - truth[q])
    return {"false_negatives": dict(fn), "false_positives": dict(fp)}


def dataset_summary(ids: Sequence[str], truth: Mapping[str, set[str]], records: Mapping[str, tuple[str, str, str]]) -> dict[str, Any]:
    countries = collections.Counter(records[value][2] for value in ids)
    singleton = sum(not truth[value] for value in ids)
    return {
        "s1_count": len(ids), "country_counts": dict(countries),
        "singleton_count": singleton, "non_singleton_count": len(ids) - singleton,
        "ground_truth_link_count": sum(len(truth[value]) for value in ids),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root(); os.chdir(root)
    for path in (args.source1, args.source2, args.source3, args.ground_truth, args.val_ids):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    original_record = telemetry.record
    def progress(event: str, **kwargs: Any) -> None:
        print(f"[{B003.utc_now()}] B005 {event} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True)
        original_record(event, **kwargs)
    telemetry.record = progress
    production = B003.import_production(root)

    eval_ids, total_val_ids = B003.load_selected_ids(args.val_ids, args.val_limit)
    eval_set = set(eval_ids)
    pool_ids = select_training_ids(args.ground_truth, eval_set, args.train_limit)
    all_initial_ids = pool_ids + eval_ids
    records, _ = B003.load_validation_records(args.source1, all_initial_ids)
    truth = B003.load_ground_truth(args.ground_truth, all_initial_ids)
    countries = {value: records[value][2] for value in all_initial_ids}
    train_ids, calibration_ids = deterministic_calibration_split(
        pool_ids, truth, countries, args.calibration_fraction, args.seed,
    )
    if eval_set & (set(train_ids) | set(calibration_ids)):
        raise RuntimeError("Frozen evaluation IDs leaked into B005 training/calibration")
    all_ids = train_ids + calibration_ids + eval_ids
    split_by_id = {value: "train" for value in train_ids}
    split_by_id.update({value: "calibration" for value in calibration_ids})
    split_by_id.update({value: "evaluation" for value in eval_ids})
    country_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    for value in all_ids:
        country_to_ids[records[value][2]].append(value)

    manifest = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "RUNNING",
        "created_at": B003.utc_now(), "git_branch": B003.git_value(root, "branch", "--show-current"),
        "git_commit": B003.git_value(root, "rev-parse", "HEAD"),
        "python_version": sys.version, "platform": platform.platform(),
        "sampling": "first N non-validation ground-truth rows; deterministic stratified calibration split",
        "train": dataset_summary(train_ids, truth, records),
        "calibration": dataset_summary(calibration_ids, truth, records),
        "evaluation": dataset_summary(eval_ids, truth, records),
        "validation_file_total_ids": total_val_ids,
        "feature_names": list(B003.FEATURE_NAMES), "feature_count": len(B003.FEATURE_NAMES),
        "retrieval": {
            "channels": list(B004.CHANNELS), "frozen_from": "B004",
            "external_evaluation_cache": str(args.b004_eval_dir.resolve()) if args.b004_eval_dir else None,
        },
        "inputs": {name: str(getattr(args, name).resolve()) for name in ("source1", "source2", "source3", "ground_truth", "val_ids")},
    }
    fingerprint_value = {
        "inputs": manifest["inputs"],
        "input_sizes": {name: getattr(args, name).stat().st_size for name in ("source1", "source2", "source3", "ground_truth", "val_ids")},
        "validation_ids_sha256": B003.sha256_file(args.val_ids),
        "train_limit": args.train_limit, "calibration_fraction": args.calibration_fraction,
        "seed": args.seed, "max_negatives": args.max_negatives_per_s1,
        "retrieval": manifest["retrieval"],
    }
    manifest["config_fingerprint"] = hashlib.sha256(json.dumps(fingerprint_value, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        old = B003.read_json(manifest_path)
        if old.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise RuntimeError("Refusing resume because B005 configuration changed")
    elif manifest_path.exists() and any((args.output_dir / "retrieval").glob("channels/*/*/*/*.DONE")):
        raise RuntimeError("B005 checkpoints exist; pass --resume or use a new output directory")
    B003.atomic_json(manifest_path, manifest)
    generated_ids = train_ids + calibration_ids if args.b004_eval_dir else all_ids
    generated_country_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    for value in generated_ids:
        generated_country_to_ids[records[value][2]].append(value)
    telemetry.record("retrieval_start", extra={"queries": len(generated_ids), "external_b004_eval_cache": bool(args.b004_eval_dir)})
    generate_retrieval(args, generated_ids, records, generated_country_to_ids, production, telemetry)

    eval_truth = {value: truth[value] for value in eval_ids}
    retrieval_verification, eval_evidence = verify_b004(
        args, eval_ids, eval_truth, records, country_to_ids,
        production["evaluate_predictions"],
    )
    manifest["frozen_b004_verification"] = retrieval_verification
    B003.atomic_json(manifest_path, manifest)
    if retrieval_verification["status"] != "PASS":
        manifest["status"] = "STOPPED_RETRIEVAL_MISMATCH"; B003.atomic_json(manifest_path, manifest)
        raise RuntimeError(f"Frozen B004 retrieval mismatch: {retrieval_verification}")

    stats_parts: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for country in sorted(country_to_ids):
        targets = B003.load_targets_for_country(args.source2, args.source3, country)
        target_preprocessed = {
            target_id: preprocess_record(name, address, production["norm"])
            for target_id, (name, address, _record_country) in targets.items()
        }
        del targets
        query_ids = generated_country_to_ids.get(country, [])
        for chunk_index, chunk_ids in B003.chunked(query_ids, args.query_chunk_size):
            merged = load_merged_chunk(args.output_dir / "retrieval", country, chunk_index, args.rrf_constant)
            chunks_by_split: dict[str, list[str]] = collections.defaultdict(list)
            for query_id in chunk_ids:
                chunks_by_split[split_by_id[query_id]].append(query_id)
            for split, split_ids in chunks_by_split.items():
                path = args.output_dir / "features" / split / f"{country}_chunk_{chunk_index:05d}.parquet"
                if args.resume and path.exists() and path.with_suffix(".DONE").exists():
                    metrics_path = path.with_suffix(".metrics.json")
                    if not metrics_path.exists():
                        raise RuntimeError(f"Resume checkpoint lacks metrics: {metrics_path}")
                    stats_parts[split].append(B003.read_json(metrics_path))
                    continue
                subset_merged = {key: value for key, value in merged.items() if key[0] in set(split_ids)}
                part = write_feature_checkpoint(
                    path, split_ids, split_by_id, truth, records, target_preprocessed,
                    subset_merged, production, args.max_negatives_per_s1,
                )
                stats_parts[split].append(part)
                telemetry.record("feature_chunk_done", country=country, processed_s1=len(split_ids), candidate_pairs=part["candidate_pairs"], extra={"split": split, "chunk": chunk_index})
        if args.b004_eval_dir:
            external_eval_ids = [value for value in eval_ids if records[value][2] == country]
            for chunk_index, chunk_ids in B003.chunked(external_eval_ids, args.query_chunk_size):
                path = args.output_dir / "features" / "evaluation" / f"{country}_external_chunk_{chunk_index:05d}.parquet"
                if args.resume and path.exists() and path.with_suffix(".DONE").exists():
                    metrics_path = path.with_suffix(".metrics.json")
                    if not metrics_path.exists():
                        raise RuntimeError(f"Resume checkpoint lacks metrics: {metrics_path}")
                    stats_parts["evaluation"].append(B003.read_json(metrics_path))
                    continue
                chunk_set = set(chunk_ids)
                subset_merged = {
                    (query_id, str(evidence["candidate_id"])): evidence
                    for query_id in chunk_ids for evidence in eval_evidence[query_id]
                    if query_id in chunk_set
                }
                part = write_feature_checkpoint(
                    path, chunk_ids, split_by_id, truth, records, target_preprocessed,
                    subset_merged, production, args.max_negatives_per_s1,
                )
                stats_parts["evaluation"].append(part)
                telemetry.record("feature_chunk_done", country=country, processed_s1=len(chunk_ids), candidate_pairs=part["candidate_pairs"], extra={"split": "evaluation", "chunk": chunk_index, "external_b004_cache": True})
        del target_preprocessed

    training_metrics = combine_stats(stats_parts["train"])
    training_metrics["gt_retrieval_recall"] = (
        training_metrics.get("retrieved_positives", 0) /
        max(1, training_metrics.get("retrieved_positives", 0) + training_metrics.get("missed_positives", 0))
    )
    B003.atomic_json(args.output_dir / "training_candidate_metrics.json", training_metrics)
    B003.atomic_json(args.output_dir / "hard_negative_metrics.json", training_metrics)

    model, device, model_params = fit_model(args, production, telemetry)
    model_path = args.model_dir / "lightgbm.joblib"; model.save(str(model_path))
    calibration_scores = score_paths(model, feature_paths(args.output_dir, "calibration"), calibration_ids)
    calibration_truth = {value: truth[value] for value in calibration_ids}
    t2, t3, calibration_metrics = tune_thresholds(
        calibration_truth, calibration_scores, production["apply_threshold_and_deduplication"],
        production["evaluate_predictions"],
    )
    calibration_metrics.update({"threshold_s2": t2, "threshold_s3": t3, "split": dataset_summary(calibration_ids, truth, records)})
    B003.atomic_json(args.output_dir / "calibration_metrics.json", calibration_metrics)

    eval_scores = score_paths(model, feature_paths(args.output_dir, "evaluation"), eval_ids)
    predictions = production["apply_threshold_and_deduplication"](eval_scores, t2, t3)
    overall = dict(production["evaluate_predictions"](eval_truth, predictions))
    country_metrics = {
        country: subset_metrics([value for value in eval_ids if records[value][2] == country], eval_truth, predictions, production["evaluate_predictions"])
        for country in sorted({records[value][2] for value in eval_ids})
    }
    evaluation_metrics = {
        "overall": overall, "country": country_metrics,
        "S2": source_metrics(eval_truth, predictions, "S2-"),
        "S3": source_metrics(eval_truth, predictions, "S3-"),
        "threshold_s2": t2, "threshold_s3": t3,
        "b004_retrieval": retrieval_verification,
        "candidate_oracle_f05": EXPECTED_B004["oracle_f05"],
        "gap_to_oracle": EXPECTED_B004["oracle_f05"] - float(overall["macro_f05"]),
        "comparison": {
            "Historical Model A on B003": 0.746790,
            "Historical Model B on B003": 0.768838,
            "B005 new model on B004 union": float(overall["macro_f05"]),
            "Candidate oracle": EXPECTED_B004["oracle_f05"],
        },
    }
    evaluation_metrics["acceptance_diagnostics"] = {
        "retrieval_reproduced": retrieval_verification["status"] == "PASS",
        "improves_historical_model_b": float(overall["macro_f05"]) > 0.768838,
        "precision_above_historical_model_b": float(overall["global_precision"]) > 0.784835,
        "note": "Materiality is reported for review rather than encoded as an invented score target.",
    }
    errors = error_analysis(eval_truth, eval_scores, predictions, eval_evidence, t2, t3)
    B003.atomic_json(args.output_dir / "evaluation_metrics.json", evaluation_metrics)
    B003.atomic_json(args.output_dir / "errors.json", errors)
    metadata = {
        "experiment_id": EXPERIMENT_ID, "feature_names": list(B003.FEATURE_NAMES),
        "feature_count": len(B003.FEATURE_NAMES), "model_params": model_params,
        "device_used": device, "optimal_s2_threshold": t2, "optimal_s3_threshold": t3,
        "calibration_macro_f05": calibration_metrics["macro_f05"],
        "frozen_evaluation_macro_f05": overall["macro_f05"],
    }
    B003.atomic_json(args.model_dir / "metadata.json", metadata)
    manifest.update({"status": "COMPLETE", "completed_at": B003.utc_now(), "device_used": device, "model": str(model_path.resolve()), "thresholds": {"S2": t2, "S3": t3}})
    B003.atomic_json(manifest_path, manifest); B003.atomic_text(args.output_dir / "B005.DONE", B003.utc_now() + "\n")
    print("\nB005 COMPLETE", flush=True)
    print(f"Frozen retrieval: {retrieval_verification}", flush=True)
    print(f"Device: {device} | thresholds S2={t2:.3f} S3={t3:.3f}", flush=True)
    print(f"Frozen Macro F0.5={overall['macro_f05']:.6f} precision={overall['global_precision']:.6f} recall={overall['global_recall']:.6f}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True); parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True); parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt")); parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=50_000); parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--max-negatives-per-s1", type=int, default=50); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B005")); parser.add_argument("--model-dir", type=Path, default=Path("models/experiments/B005"))
    parser.add_argument("--b004-eval-dir", type=Path, default=None, help="Optional completed B004 artifact directory for the frozen 5k evaluation cache")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--query-chunk-size", type=int, default=250); parser.add_argument("--progress-every-targets", type=int, default=500_000)
    parser.add_argument("--r0-k", type=int, default=50); parser.add_argument("--r0-name-prune", type=int, default=300); parser.add_argument("--r0-addr-prune", type=int, default=150)
    parser.add_argument("--tfidf-k", type=int, default=30); parser.add_argument("--tfidf-min-score", type=float, default=1e-6); parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--name-char-analyzer", choices=("char", "char_wb"), default="char_wb"); parser.add_argument("--name-char-ngram-min", type=int, default=3); parser.add_argument("--name-char-ngram-max", type=int, default=5)
    parser.add_argument("--name-char-max-features", type=int, default=1_000_000); parser.add_argument("--address-word-max-features", type=int, default=750_000); parser.add_argument("--name-word-max-features", type=int, default=500_000)
    parser.add_argument("--sparse-threads", type=int, default=max(1, os.cpu_count() or 1)); parser.add_argument("--rrf-constant", type=float, default=60.0); parser.add_argument("--r0-guarantee", type=int, default=10)
    parser.add_argument("--retrieval-tolerance", type=float, default=0.0005)
    parser.add_argument("--n-estimators", type=int, default=500); parser.add_argument("--learning-rate", type=float, default=0.035); parser.add_argument("--num-leaves", type=int, default=63); parser.add_argument("--max-depth", type=int, default=7)
    args = parser.parse_args(argv)
    if args.r0_k != 50 or args.tfidf_k != 30 or args.rrf_constant != 60.0:
        parser.error("B005 freezes B004 retrieval (R0 K50, TF-IDF K30/source, RRF C60)")
    frozen = {
        "r0_name_prune": 300, "r0_addr_prune": 150,
        "tfidf_min_score": 1e-6, "tfidf_min_df": 2,
        "name_char_analyzer": "char_wb", "name_char_ngram_min": 3,
        "name_char_ngram_max": 5, "name_char_max_features": 1_000_000,
        "address_word_max_features": 750_000, "name_word_max_features": 500_000,
    }
    changed = [name for name, expected in frozen.items() if getattr(args, name) != expected]
    if changed:
        parser.error("B005 freezes all B004 retrieval parameters; changed: " + ", ".join(changed))
    if args.train_limit <= 0 or args.max_negatives_per_s1 <= 0 or args.query_chunk_size <= 0:
        parser.error("limits and chunk size must be positive")
    if not 0 < args.calibration_fraction < 0.5:
        parser.error("--calibration-fraction must be between 0 and 0.5")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
