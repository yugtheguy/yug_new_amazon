#!/usr/bin/env python3
"""Materialize the already-selected B007_03 and B007_01 models for production."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/B003", "experiments/B005", "experiments/B006", "experiments/B007"):
    sys.path.insert(0, str(ROOT / rel))
import run_b003 as B003  # type: ignore  # noqa: E402
import run_b005 as B005  # type: ignore  # noqa: E402
import run_b006 as B006  # type: ignore  # noqa: E402
import run_b007 as B007R  # type: ignore  # noqa: E402
import b007_features as B007F  # type: ignore  # noqa: E402

FROZEN_METRICS = {
    "B007_01": {"macro_f05": 0.9611574772786582, "precision": 0.985741, "recall": 0.916705},
    "B007_03": {"macro_f05": 0.9617940138448521, "precision": 0.987924, "recall": 0.915786},
}


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_training_stats(paths: list[Path], output: Path) -> B007F.CorpusStats:
    if output.exists():
        print(f"[freeze] loading frozen corpus statistics: {output}", flush=True)
        return joblib.load(output)
    stats = B007F.CorpusStats()
    for path in paths:
        for index, (_entity_id, name, address, _country) in enumerate(B003.iter_tsv(path), start=1):
            stats.add(name, address)
            if index % 1_000_000 == 0:
                print(f"[freeze] corpus stats {path.name}: {index:,} rows", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, output, compress=3)
    print(f"[freeze] saved corpus stats for {stats.doc_count:,} documents", flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b005-artifacts", type=Path, required=True)
    parser.add_argument("--b005-metadata", type=Path, required=True)
    parser.add_argument("--b007-artifacts", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/final_b007"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/final_b007"))
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args()
    os.chdir(ROOT)
    for path in (args.b005_artifacts, args.b005_metadata, args.b007_artifacts, args.source1, args.source2, args.source3, args.ground_truth):
        if not path.exists():
            raise FileNotFoundError(path)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    production = B003.import_production(ROOT)
    metadata = B003.read_json(args.b005_metadata)
    paths = {split: B006.feature_paths(args.b005_artifacts, split) for split in ("train", "calibration", "evaluation")}
    matrices, labels, row_ids, split_ids = {}, {}, {}, {}
    for split in paths:
        best, ids, _ = B006.channel_best_maps(paths[split])
        matrix, y, queries, candidates, countries, _ = B006.load_split(paths[split], best, False, B003.Telemetry(args.artifact_dir / "freeze_telemetry.jsonl"))
        cached = np.load(args.b007_artifacts / f"b007_features_{split}.npy")
        if len(cached) != len(matrix):
            raise RuntimeError(f"B007 {split} cache row mismatch")
        matrices[split] = np.hstack((matrix, cached))
        labels[split] = y
        row_ids[split] = (queries, candidates, countries)
        split_ids[split] = ids

    all_ids = sorted(set().union(*split_ids.values()))
    truth = B003.load_ground_truth(args.ground_truth, all_ids)
    fit_args = SimpleNamespace(device=args.device)
    frozen = {}
    for experiment in ("B007_01", "B007_03"):
        feature_names = list(B007R.ABLATIONS[experiment])
        indices = [B007R.ALL_FEATURES.index(name) for name in feature_names]
        model, device, fitted_params = B006.fit_model(
            fit_args, production, metadata["model_params"], matrices["train"][:, indices], labels["train"],
            B003.Telemetry(args.artifact_dir / "freeze_telemetry.jsonl"),
        )
        cal_q, cal_c, _ = row_ids["calibration"]
        cal_scores = B006.score_dict(model, matrices["calibration"][:, indices], cal_q, cal_c, split_ids["calibration"])
        cal_truth = {q: truth[q] for q in split_ids["calibration"]}
        t2, t3, cal_metrics = B005.tune_thresholds(cal_truth, cal_scores, production["apply_threshold_and_deduplication"], production["evaluate_predictions"])
        eval_q, eval_c, _ = row_ids["evaluation"]
        eval_scores = B006.score_dict(model, matrices["evaluation"][:, indices], eval_q, eval_c, split_ids["evaluation"])
        predictions = production["apply_threshold_and_deduplication"](eval_scores, t2, t3)
        observed = production["evaluate_predictions"]({q: truth[q] for q in split_ids["evaluation"]}, predictions)
        expected = FROZEN_METRICS[experiment]
        if abs(float(observed["macro_f05"]) - expected["macro_f05"]) > 0.002:
            raise RuntimeError(f"{experiment} reproduction failed: {observed}")
        model_name = "lightgbm.joblib" if experiment == "B007_03" else "lightgbm_b007_01.joblib"
        model.save(str(args.model_dir / model_name))
        probability_reference = None
        if experiment == "B007_03":
            flat_probabilities = np.asarray([probability for values in eval_scores.values() for _candidate, probability in values], dtype=float)
            evaluation_matrix = matrices["evaluation"]
            probability_reference = {
                "candidate_pairs": int(len(flat_probabilities)),
                "avg_candidates_per_s1": float(len(flat_probabilities) / max(1, len(split_ids["evaluation"]))),
                "retriever_hit_rates_pct": {
                    channel: float(100.0 * evaluation_matrix[:, B007R.ALL_FEATURES.index(f"found_{channel}")].mean())
                    for channel in ("R0", "R1", "R2", "R3")
                },
                "probability": {
                    "mean": float(flat_probabilities.mean()),
                    **{f"p{percentile}": float(np.percentile(flat_probabilities, percentile)) for percentile in (50, 90, 95, 99)},
                },
                "percentage_above_final_threshold": float(100.0 * sum(
                    probability >= (t2 if candidate.startswith("S2-") else t3)
                    for values in eval_scores.values() for candidate, probability in values
                ) / max(1, len(flat_probabilities))),
            }
        frozen[experiment] = {
            "model_path": str((args.model_dir / model_name).resolve()), "feature_names": feature_names,
            "feature_schema_sha256": sha256_json(feature_names), "model_params": fitted_params, "device": device,
            "threshold_s2": t2, "threshold_s3": t3, "calibration_macro_f05": float(cal_metrics["macro_f05"]),
            "observed_metrics": {"macro_f05": float(observed["macro_f05"]), "precision": float(observed["global_precision"]), "recall": float(observed["global_recall"])},
            "declared_frozen_metrics": expected,
            "evaluation_distribution_reference": probability_reference,
        }
        print(f"[freeze] {experiment}: F0.5={observed['macro_f05']:.6f} S2={t2:.3f} S3={t3:.3f}", flush=True)

    stats_path = args.model_dir / "training_corpus_stats.joblib"
    stats = build_training_stats([args.source1, args.source2, args.source3], stats_path)
    split_identity = {
        split: {"query_count": len(ids), "query_ids_sha256": B003.sha256_ids(sorted(ids)), "row_count": len(labels[split])}
        for split, ids in split_ids.items()
    }
    normalization_files = [ROOT / "experiments/B007/b007_features.py", ROOT / "code/business_entity_resolution/src/normalization.py"]
    metadata_out = {
        "schema_version": 1, "model_id": "B007_03", "backup_model_id": "B007_01",
        "created_at": B003.utc_now(), "git_commit": B003.git_value(ROOT, "rev-parse", "HEAD"),
        "models": frozen, "split_identity": split_identity,
        "retrieval_version": "B004:R0_K50+R1_NAME_CHAR_K30+R2_ADDRESS_WORD_K30+R3_NAME_WORD_K30:uncapped_union",
        "parser_normalization_version": {path.relative_to(ROOT).as_posix(): B003.sha256_file(path) for path in normalization_files},
        "corpus_statistics": {"path": str(stats_path.resolve()), "doc_count": stats.doc_count, "semantics": "all rows of TRAIN source1+source2+source3; frozen for inference"},
    }
    B003.atomic_json(args.model_dir / "metadata.json", metadata_out)
    B003.atomic_json(args.artifact_dir / "feature_schema.json", {name: frozen[name]["feature_names"] for name in frozen})
    B003.atomic_json(args.artifact_dir / "thresholds.json", {name: {"S2": frozen[name]["threshold_s2"], "S3": frozen[name]["threshold_s3"]} for name in frozen})
    B003.atomic_json(args.artifact_dir / "manifest.json", {"status": "FROZEN", **metadata_out})
    print("[freeze] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
