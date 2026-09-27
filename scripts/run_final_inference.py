#!/usr/bin/env python3
"""Frozen B004 retrieval + B007 production scoring with country checkpoints."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/B003", "experiments/B004", "experiments/B005", "experiments/B006", "experiments/B007"):
    sys.path.insert(0, str(ROOT / rel))
import run_b003 as B003  # type: ignore  # noqa: E402
import run_b004 as B004  # type: ignore  # noqa: E402
import run_b005 as B005  # type: ignore  # noqa: E402
import run_b006 as B006  # type: ignore  # noqa: E402
import run_b007 as B007R  # type: ignore  # noqa: E402
import b007_features as B007F  # type: ignore  # noqa: E402


class Progress:
    def __init__(self, path: Path):
        self.telemetry = B003.Telemetry(path)
        self.started = time.monotonic()

    def record(self, event: str, **kwargs: Any) -> None:
        print(f"[{B003.utc_now()}] FINAL {event} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True)
        self.telemetry.record(event, **kwargs)


def done_valid(done: Path, artifact: Path) -> bool:
    if not done.exists() or not artifact.exists():
        return False
    payload = B003.read_json(done)
    return payload.get("sha256") == B003.sha256_file(artifact) and payload.get("size") == artifact.stat().st_size


def mark_done(done: Path, artifact: Path, extra: Mapping[str, Any] | None = None) -> None:
    B003.atomic_json(done, {"completed_at": B003.utc_now(), "path": str(artifact.resolve()), "size": artifact.stat().st_size, "sha256": B003.sha256_file(artifact), **dict(extra or {})})


def retrieval_args(args: argparse.Namespace, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output, source2=args.source2, source3=args.source3,
        query_chunk_size=args.query_chunk_size, progress_every_targets=args.progress_every_targets,
        r0_k=50, r0_name_prune=300, r0_addr_prune=150, tfidf_k=30,
        tfidf_min_score=1e-6, tfidf_min_df=2, name_char_analyzer="char_wb",
        name_char_ngram_min=3, name_char_ngram_max=5, name_char_max_features=1_000_000,
        address_word_max_features=750_000, name_word_max_features=500_000,
        sparse_threads=args.sparse_threads, rrf_constant=60.0,
    )


def country_queries(source1: Path, country: str, limit: int | None = None) -> tuple[list[str], dict[str, tuple[str, str, str]]]:
    ids, records = [], {}
    for entity_id, name, address, record_country in B003.iter_tsv(source1):
        if record_country != country:
            continue
        ids.append(entity_id); records[entity_id] = (name, address, record_country)
        if limit and len(ids) >= limit:
            break
    if not ids:
        raise RuntimeError(f"No Source1 rows found for country {country!r}")
    return ids, records


def channel_best(merged: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = collections.defaultdict(dict)
    mapping = {"R0": ("found_by_existing", "existing_shared_keys"), "R1": ("found_by_name_char", "name_char_score"), "R2": ("found_by_address_word", "address_word_score"), "R3": ("found_by_name_word", "name_word_score")}
    for (query_id, _), row in merged.items():
        for channel, (flag, score_name) in mapping.items():
            if row.get(flag):
                score = float(row[score_name]); result[query_id][channel] = max(result[query_id].get(channel, -np.inf), score)
    return result


def build_rows(
    chunk_ids: Sequence[str], merged: Mapping[tuple[str, str], Mapping[str, Any]],
    query_records: Mapping[str, tuple[str, str, str]], targets: Mapping[str, tuple[str, str, str]],
    stats: Any, production: Mapping[str, Any], feature_schema: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    norm, extract = production["norm"], production["extract_features_for_pair"]
    best = channel_best(merged)
    target_preprocessed = {}
    texts = {qid: (query_records[qid][0], query_records[qid][1]) for qid in chunk_ids}
    candidate_rows, feature_rows = [], []
    parser = collections.Counter()
    seen = set()
    for (query_id, candidate_id), evidence in sorted(merged.items()):
        source = "S2" if candidate_id.startswith("S2-") else "S3" if candidate_id.startswith("S3-") else None
        if source is None or str(evidence["candidate_source"]) != source:
            raise RuntimeError(f"Candidate source mismatch: {(query_id, candidate_id)}")
        key = (query_id, candidate_id, source)
        if key in seen:
            raise RuntimeError(f"Duplicate canonical candidate key: {key}")
        seen.add(key)
        if candidate_id not in targets:
            raise KeyError(f"Candidate absent from country target population: {candidate_id}")
        t_name, t_address, _ = targets[candidate_id]
        texts[candidate_id] = (t_name, t_address)
        if candidate_id not in target_preprocessed:
            clean_name, core_name, _ = norm.normalize_name(t_name)
            clean_addr, nums, _ = norm.normalize_address(t_address)
            target_preprocessed[candidate_id] = (clean_name, core_name, clean_addr, nums)
        q_name, q_address, country = query_records[query_id]
        clean_name, core_name, _ = norm.normalize_name(q_name)
        clean_addr, nums, _ = norm.normalize_address(q_address)
        base = extract((clean_name, core_name, clean_addr, nums), target_preprocessed[candidate_id], candidate_id, int(evidence.get("existing_shared_keys") or 0))
        retrieval = B006.retrieval_feature_values(evidence, best[query_id])
        b007 = B007F.generate_features(query_id, candidate_id, texts, stats)
        all_values = dict(zip(B007R.ALL_FEATURES, [*base, *retrieval, *b007]))
        if list(feature_schema) != list(B007R.ABLATIONS["B007_03"]):
            raise RuntimeError("Inference feature schema differs from frozen B007_03 schema")
        selected = [float(all_values[name]) for name in feature_schema]
        if not np.isfinite(selected).all():
            raise RuntimeError(f"NaN/inf feature for {key}")
        candidate_rows.append({**dict(evidence), "candidate_source": source})
        feature_rows.append({"query_id": query_id, "candidate_id": candidate_id, "candidate_source": source, "country": country, **{name: value for name, value in zip(feature_schema, selected)}})
        for address in (q_address, t_address):
            parsed = B007F.parse_numeric_address(address)
            parser["records"] += 1
            parser["premise"] += int("premise_base" in parsed); parser["unit"] += int("unit" in parsed)
            parser["floor"] += int("floor" in parsed); parser["postal"] += int("postal" in parsed)
    return candidate_rows, feature_rows, dict(parser)


def combine_parquets(paths: Sequence[Path], output: Path) -> None:
    import pyarrow.parquet as pq  # type: ignore
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No prediction chunks to combine")
    os.replace(temporary, output)


def run_country(args: argparse.Namespace, country: str, smoke_limit: int | None = None) -> Path:
    suffix = f"{country}_smoke_{smoke_limit}" if smoke_limit else country
    country_dir = args.output_dir / suffix
    country_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(country_dir / "telemetry.jsonl")
    ids, records = country_queries(args.source1, country, smoke_limit)
    run_config = {
        "country": country, "smoke_limit": smoke_limit, "query_ids_sha256": B003.sha256_ids(ids),
        "inputs": {path.name: {"size": path.stat().st_size, "sha256": B003.sha256_file(path)} for path in (args.source1, args.source2, args.source3)},
        "model_id": args.model_id, "query_chunk_size": args.query_chunk_size,
        "retrieval": "B004:R0_K50+R1/R2/R3_K30:uncapped_union",
    }
    run_config["fingerprint"] = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    config_path = country_dir / "run_config.json"
    if config_path.exists():
        previous = B003.read_json(config_path)
        if previous.get("fingerprint") != run_config["fingerprint"]:
            raise RuntimeError(f"Refusing {country} resume because inputs/configuration changed")
        if not args.resume:
            raise RuntimeError(f"{country} checkpoints already exist; pass --resume or use a new output directory")
    B003.atomic_json(config_path, run_config)
    production = B003.import_production(ROOT)
    norm, blocker, _ = B004.import_retrieval_modules()
    rdir = country_dir / "_retrieval"
    rargs = retrieval_args(args, rdir)
    progress.record("country_start", country=country, processed_s1=len(ids), extra={"smoke_limit": smoke_limit})
    B004.run_r0_country(rargs, country, ids, records, norm, blocker, progress)
    B004.run_tfidf_source_country(rargs, country, "S2", args.source2, ids, records, norm, progress)
    B004.run_tfidf_source_country(rargs, country, "S3", args.source3, ids, records, norm, progress)

    metadata = B003.read_json(args.model_dir / "metadata.json")
    frozen = metadata["models"][args.model_id]
    model_feature_schema = frozen["feature_names"]
    feature_schema = metadata["models"]["B007_03"]["feature_names"]
    expected_hash = frozen["feature_schema_sha256"]
    observed_hash = hashlib.sha256(json.dumps(model_feature_schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError("Frozen feature schema hash mismatch")
    stats = joblib.load(args.model_dir / "training_corpus_stats.joblib")
    model_file = "lightgbm.joblib" if args.model_id == "B007_03" else "lightgbm_b007_01.joblib"
    model = production["EntityMatcherModel"].load(str(args.model_dir / model_file))
    targets = B003.load_targets_for_country(args.source2, args.source3, country)
    candidate_counts = collections.Counter(); retriever_counts = collections.Counter(); parser_totals = collections.Counter(); semantic_missing = collections.Counter()
    prediction_chunks = []
    for chunk_index, chunk_ids in B003.chunked(ids, args.query_chunk_size):
        candidate_path = country_dir / "candidates" / f"chunk_{chunk_index:05d}.parquet"
        feature_path = country_dir / "features" / f"chunk_{chunk_index:05d}.parquet"
        prediction_path = country_dir / "prediction_chunks" / f"chunk_{chunk_index:05d}.parquet"
        if args.resume and done_valid(prediction_path.with_suffix(".DONE"), prediction_path):
            prediction_chunks.append(prediction_path)
            metrics_path = prediction_path.with_suffix(".metrics.json")
            if not metrics_path.exists():
                raise RuntimeError(f"Resume checkpoint lacks metrics: {metrics_path}")
            metrics = B003.read_json(metrics_path)
            candidate_counts.update(metrics["candidate_counts"])
            retriever_counts.update(metrics["retriever_counts"])
            parser_totals.update(metrics["parser"])
            semantic_missing.update(metrics.get("semantic_missing", {}))
            continue
        merged = B005.load_merged_chunk(rdir, country, chunk_index, 60.0)
        candidate_rows, feature_rows, parser = build_rows(chunk_ids, merged, records, targets, stats, production, feature_schema)
        B003.write_parquet(candidate_path, candidate_rows); mark_done(candidate_path.with_suffix(".DONE"), candidate_path, {"rows": len(candidate_rows)})
        B003.write_parquet(feature_path, feature_rows); mark_done(feature_path.with_suffix(".DONE"), feature_path, {"rows": len(feature_rows)})
        X = np.asarray([[row[name] for name in model_feature_schema] for row in feature_rows], dtype=np.float32)
        probabilities = model.predict_proba(X) if len(X) else np.asarray([], dtype=np.float32)
        prediction_rows = [{"query_id": row["query_id"], "candidate_id": row["candidate_id"], "candidate_source": row["candidate_source"], "country": country, "probability": float(p)} for row, p in zip(feature_rows, probabilities)]
        B003.write_parquet(prediction_path, prediction_rows); mark_done(prediction_path.with_suffix(".DONE"), prediction_path, {"rows": len(prediction_rows)})
        prediction_chunks.append(prediction_path)
        chunk_candidate_counts = collections.Counter()
        chunk_retriever_counts = collections.Counter()
        for row in candidate_rows:
            candidate_counts[row["query_id"]] += 1; chunk_candidate_counts[row["query_id"]] += 1
            for channel, field in (("R0", "found_by_existing"), ("R1", "found_by_name_char"), ("R2", "found_by_address_word"), ("R3", "found_by_name_word")):
                value = int(bool(row[field])); retriever_counts[channel] += value; chunk_retriever_counts[channel] += value
        parser_totals.update(parser)
        chunk_semantic_missing = {
            name: int(sum(float(row[name]) for row in feature_rows))
            for name in ("name_missing_left", "name_missing_right", "address_missing_left", "address_missing_right")
            if name in feature_schema
        }
        semantic_missing.update(chunk_semantic_missing)
        B003.atomic_json(prediction_path.with_suffix(".metrics.json"), {"candidate_counts": dict(chunk_candidate_counts), "retriever_counts": dict(chunk_retriever_counts), "parser": parser, "semantic_missing": chunk_semantic_missing})
        progress.record("chunk_complete", country=country, processed_s1=min((chunk_index + 1) * args.query_chunk_size, len(ids)), candidate_pairs=len(candidate_rows), extra={"chunk": chunk_index})
    final_predictions = country_dir / "predictions.parquet"
    combine_parquets(prediction_chunks, final_predictions); mark_done(country_dir / "DONE", final_predictions, {"queries": len(ids), "model_id": args.model_id})
    counts = [candidate_counts.get(q, 0) for q in ids]
    # Read all probability chunks so reports remain correct on --resume.
    import pyarrow.parquet as pq  # type: ignore
    probability_values = []
    for path in prediction_chunks:
        probability_values.extend(float(v) for v in pq.read_table(path, columns=["probability"])["probability"].to_pylist())
    probs = np.asarray(probability_values, dtype=float)
    report = {
        "country": country, "smoke": smoke_limit is not None, "model_id": args.model_id,
        "s1_queries": len(ids), "s2_targets": sum(1 for key in targets if key.startswith("S2-")), "s3_targets": sum(1 for key in targets if key.startswith("S3-")),
        "candidate_pairs": int(sum(counts)), "avg_candidates_per_s1": float(np.mean(counts)),
        "candidate_p50": float(np.percentile(counts, 50)), "candidate_p95": float(np.percentile(counts, 95)), "candidate_p99": float(np.percentile(counts, 99)),
        "retriever_pair_hits": dict(retriever_counts),
        "retriever_hit_rates_pct": {channel: 100.0 * retriever_counts.get(channel, 0) / max(1, sum(counts)) for channel in ("R0", "R1", "R2", "R3")},
        "probability": {name: float(value) for name, value in zip(("mean", "p50", "p90", "p95", "p99"), ([probs.mean(), *np.percentile(probs, [50, 90, 95, 99])] if len(probs) else [0, 0, 0, 0, 0]))},
        "parser": {name + "_pct": 100.0 * parser_totals.get(name, 0) / max(1, parser_totals.get("records", 0)) for name in ("premise", "unit", "floor", "postal")},
        "b007_feature_nan_pct": 0.0,
        "semantic_missingness_pct": {name: 100.0 * semantic_missing.get(name, 0) / max(1, sum(counts)) for name in ("name_missing_left", "name_missing_right", "address_missing_left", "address_missing_right")},
        "evaluation_distribution_reference": frozen.get("evaluation_distribution_reference"),
        "feature_schema_sha256": expected_hash, "elapsed_seconds": time.monotonic() - progress.started,
    }
    # Stream threshold statistic without retaining all prediction rows.
    above = total = 0
    for path in prediction_chunks:
        data = pq.read_table(path, columns=["candidate_source", "probability"]).to_pydict()
        for source, probability in zip(data["candidate_source"], data["probability"]):
            total += 1; above += int(float(probability) >= (frozen["threshold_s2"] if source == "S2" else frozen["threshold_s3"]))
    report["percentage_above_final_threshold"] = 100.0 * above / max(1, total)
    reference = frozen.get("evaluation_distribution_reference") or {}
    anomalies = []
    reference_avg = float(reference.get("avg_candidates_per_s1") or 0.0)
    if reference_avg and (report["avg_candidates_per_s1"] > 3.0 * reference_avg or report["avg_candidates_per_s1"] < 0.2 * reference_avg):
        anomalies.append("candidate_count_per_s1_extreme_vs_frozen_evaluation")
    for channel, rate in report["retriever_hit_rates_pct"].items():
        reference_rate = float(reference.get("retriever_hit_rates_pct", {}).get(channel, rate))
        if abs(rate - reference_rate) >= 30.0:
            anomalies.append(f"{channel}_hit_rate_shift_ge_30pp")
    reference_p95 = float(reference.get("probability", {}).get("p95") or 0.0)
    if reference_p95 and abs(report["probability"]["p95"] - reference_p95) >= 0.5:
        anomalies.append("model_probability_p95_extreme_shift")
    report["extreme_anomaly_flags"] = anomalies
    B003.atomic_json(country_dir / ("france_smoke_report.json" if smoke_limit else "country_report.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    return final_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True); parser.add_argument("--source2", type=Path, required=True); parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/final_b007")); parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_test"))
    parser.add_argument("--country", action="append", help="Repeat for independently runnable countries; default discovers all countries")
    parser.add_argument("--smoke-france", type=int, default=0); parser.add_argument("--model-id", choices=("B007_03", "B007_01"), default="B007_03")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--query-chunk-size", type=int, default=250)
    parser.add_argument("--progress-every-targets", type=int, default=500_000); parser.add_argument("--sparse-threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); os.chdir(ROOT)
    for path in (args.source1, args.source2, args.source3, args.model_dir / "metadata.json", args.model_dir / "training_corpus_stats.joblib"):
        if not path.exists(): raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_france:
        run_country(args, "France", args.smoke_france); return 0
    countries = args.country or sorted({country for _id, _name, _address, country in B003.iter_tsv(args.source1)})
    for country in countries:
        run_country(args, country)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
