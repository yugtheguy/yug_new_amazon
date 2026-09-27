#!/usr/bin/env python3
"""Apply frozen thresholds/target uniqueness to raw scores and validate submission."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/B003"))
import run_b003 as B003  # type: ignore  # noqa: E402


def read_predictions(paths: list[Path]) -> tuple[dict[str, list[tuple[str, float]]], set[tuple[str, str]]]:
    import pyarrow.parquet as pq  # type: ignore
    scores: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    pairs = set()
    for path in paths:
        data = pq.read_table(path, columns=["query_id", "candidate_id", "candidate_source", "probability"]).to_pydict()
        for q, c, source, probability in zip(data["query_id"], data["candidate_id"], data["candidate_source"], data["probability"]):
            expected = "S2" if str(c).startswith("S2-") else "S3" if str(c).startswith("S3-") else None
            if source != expected: raise RuntimeError(f"Candidate source mismatch: {(q, c, source)}")
            key = (str(q), str(c))
            if key in pairs: raise RuntimeError(f"Duplicate raw prediction pair: {key}")
            pairs.add(key); scores[str(q)].append((str(c), float(probability)))
    return scores, pairs


def write_lists(path: Path, header: tuple[str, str], ordered_ids: list[str], mapping: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for query_id in ordered_ids:
            writer.writerow([query_id, ",".join(sorted(mapping.get(query_id, set())))])
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/final_b007"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_submission"))
    parser.add_argument("--model-id", choices=("B007_03", "B007_01"), default="B007_03")
    parser.add_argument("--train-source2", type=Path); parser.add_argument("--train-source3", type=Path)
    args = parser.parse_args(); os.chdir(ROOT)
    metadata = B003.read_json(args.model_dir / "metadata.json"); frozen = metadata["models"][args.model_id]
    source1 = args.test_dir / "test_source1.tsv"; source2 = args.test_dir / "test_source2.tsv"; source3 = args.test_dir / "test_source3.tsv"
    for path in [*args.predictions, source1, source2, source3]:
        if not path.exists(): raise FileNotFoundError(path)
    ordered_ids = [entity_id for entity_id, _name, _address, _country in B003.iter_tsv(source1)]
    if len(ordered_ids) != len(set(ordered_ids)): raise RuntimeError("Duplicate test Source1 IDs")
    scores, pair_keys = read_predictions(args.predictions)
    unknown_queries = set(scores) - set(ordered_ids)
    if unknown_queries: raise RuntimeError(f"Predictions contain unknown Source1 IDs: {sorted(unknown_queries)[:5]}")
    for query_id in ordered_ids: scores.setdefault(query_id, [])
    production = B003.import_production(ROOT)
    matches = production["apply_threshold_and_deduplication"](scores, frozen["threshold_s2"], frozen["threshold_s3"])
    candidate_map = {query_id: {candidate for candidate, _ in values} for query_id, values in scores.items()}
    valid_targets = {entity_id for path in (source2, source3) for entity_id, _n, _a, _c in B003.iter_tsv(path)}
    selected = set().union(*matches.values()) if matches else set()
    if not selected <= valid_targets: raise RuntimeError(f"Predicted unknown test targets: {sorted(selected - valid_targets)[:5]}")
    if any("nan" in value.lower() for values in matches.values() for value in values): raise RuntimeError("NaN string in predictions")
    if args.train_source2 and args.train_source3:
        train_targets = {entity_id for path in (args.train_source2, args.train_source3) for entity_id, _n, _a, _c in B003.iter_tsv(path)}
        overlap = selected & train_targets
        if overlap: raise RuntimeError(f"Train target IDs leaked into submission: {sorted(overlap)[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matching = args.output_dir / "matching_results.tsv"; candidates = args.output_dir / "candidate_pairs.tsv"
    write_lists(matching, ("source1_entity_id", "matched_entity_ids"), ordered_ids, matches)
    write_lists(candidates, ("source1_entity_id", "candidate_entity_ids"), ordered_ids, candidate_map)
    alias = args.output_dir / f"submission_{args.model_id.lower()}.tsv"
    alias.write_bytes(matching.read_bytes())
    command = [sys.executable, str(ROOT / "utils/validate_submission.py"), "--matching", str(matching), "--candidate", str(candidates), "--test-dir", str(args.test_dir), "--check-ids"]
    validation = subprocess.run(command, text=True, capture_output=True)
    print(validation.stdout, end=""); print(validation.stderr, end="", file=sys.stderr)
    if validation.returncode: raise RuntimeError("Official submission validator failed")
    manifest = {
        "status": "VALIDATED", "generated_at": B003.utc_now(), "git_commit": B003.git_value(ROOT, "rev-parse", "HEAD"),
        "model_id": args.model_id, "frozen_metrics": frozen["declared_frozen_metrics"], "feature_schema_sha256": frozen["feature_schema_sha256"],
        "thresholds": {"S2": frozen["threshold_s2"], "S3": frozen["threshold_s3"]},
        "retrieval": metadata["retrieval_version"], "input_files": {path.name: {"size": path.stat().st_size, "sha256": B003.sha256_file(path)} for path in (source1, source2, source3)},
        "prediction_artifacts": [str(path.resolve()) for path in args.predictions], "submission_path": str(matching.resolve()),
        "candidate_path": str(candidates.resolve()), "row_count": len(ordered_ids), "matched_target_count": len(selected),
        "validator": {"command": command, "returncode": validation.returncode, "stdout": validation.stdout, "stderr": validation.stderr},
    }
    B003.atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"VALID SUBMISSION: {matching} ({len(ordered_ids):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
