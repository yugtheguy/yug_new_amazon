#!/usr/bin/env python3
"""B003: frozen, full-target-population K50 baseline evaluation.

This runner intentionally does not change the matching algorithm. It imports the
current production normalization, blocking, feature extraction, metric, and greedy
target-deduplication implementations. Ground truth is loaded only after all raw
candidate checkpoints have been generated and scored.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

EXPERIMENT_ID = "B003"
FEATURE_NAMES = [
    "name_exact_clean", "name_exact_core", "name_lev_sim", "name_jw_sim",
    "name_token_sort", "name_token_set", "name_token_jaccard",
    "name_token_overlap", "name_len_diff", "name_len_ratio",
    "name_prefix_match", "name_char_3gram", "name_first_token_match",
    "name_first_token_conflict", "addr_has_val", "addr_exact_clean",
    "addr_lev_sim", "addr_jw_sim", "addr_token_sort", "addr_token_set",
    "addr_token_jaccard", "num_common_count", "num_has_match",
    "num_has_mismatch", "exact_core_no_addr", "cross_prod", "cross_sum",
    "cross_max", "cross_min", "is_source2", "is_source3",
    "shared_keys_count",
]
THRESHOLD_CONFIGS = {
    "current_s2_0.85_s3_0.80": {"s2": 0.85, "s3": 0.80},
    "historical_s2_0.75_s3_0.85": {"s2": 0.75, "s3": 0.85},
}
RANK_CUTOFFS = (1, 5, 10, 20, 50)
CANDIDATE_COLUMNS = (
    "query_id", "candidate_id", "candidate_source", "country",
    "candidate_rank", "shared_keys_count", "model_probability_A",
    "model_probability_B",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    payload = "".join(f"{item}\n" for item in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def process_memory() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore
        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss), int(psutil.virtual_memory().available)
    except Exception:
        pass

    rss = None
    available = None
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return rss, available


def gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    inventory = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            inventory.append({"name": parts[0], "memory_total_mb": parts[1], "driver": parts[2]})
    return inventory


class Telemetry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.peak_rss = 0

    def record(
        self,
        stage: str,
        *,
        country: str | None = None,
        processed_s1: int = 0,
        candidate_pairs: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        rss, available = process_memory()
        if rss is not None:
            self.peak_rss = max(self.peak_rss, rss)
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "wall_seconds": time.monotonic() - self.started,
            "stage": stage,
            "country": country,
            "processed_s1": processed_s1,
            "candidate_pairs": candidate_pairs,
            "rss_bytes": rss,
            "available_ram_bytes": available,
        }
        if extra:
            event.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event


def load_selected_ids(path: Path, limit: int) -> tuple[list[str], int]:
    all_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entity_id = line.strip()
            if entity_id:
                all_ids.append(entity_id)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"Validation ID file contains duplicates: {path}")
    selected = all_ids[:limit]
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} validation IDs, but {path} has {len(all_ids)}")
    return selected, len(all_ids)


def iter_tsv(path: Path) -> Iterator[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line_number, line in enumerate(handle, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if not parts[0]:
                raise ValueError(f"Blank entity ID at {path}:{line_number}")
            yield (
                parts[0],
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "",
                parts[3] if len(parts) > 3 else "",
            )


def load_validation_records(
    source1: Path, selected_ids: Sequence[str]
) -> tuple[dict[str, tuple[str, str, str]], dict[str, list[str]]]:
    selected = set(selected_ids)
    records: dict[str, tuple[str, str, str]] = {}
    country_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    for entity_id, name, address, country in iter_tsv(source1):
        if entity_id in selected:
            if entity_id in records:
                raise ValueError(f"Duplicate selected S1 ID in source file: {entity_id}")
            records[entity_id] = (name, address, country)
    missing = selected - records.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected validation IDs missing from S1; examples: {sorted(missing)[:5]}")
    for entity_id in selected_ids:
        country_to_ids[records[entity_id][2]].append(entity_id)
    return records, dict(country_to_ids)


def load_targets_for_country(
    source2: Path, source3: Path, country: str
) -> dict[str, tuple[str, str, str]]:
    """Match production semantics: S2 then S3 into one insertion-ordered dict."""
    targets: dict[str, tuple[str, str, str]] = {}
    for source_path in (source2, source3):
        for entity_id, name, address, record_country in iter_tsv(source_path):
            if record_country == country:
                targets[entity_id] = (name, address, record_country)
    return targets


def prune_limit(key: tuple[str, str], name_prune: int, addr_prune: int) -> int:
    key_type = key[0]
    return name_prune if (
        key_type.startswith("n") or key_type.startswith("core") or key_type.startswith("compact")
    ) else addr_prune


def build_production_index(
    targets: Mapping[str, tuple[str, str, str]],
    normalize_name: Callable[[str], tuple[str, str, str]],
    normalize_address: Callable[[str], tuple[str, set[str], str]],
    get_blocking_keys: Callable[[str, str, str], set[tuple[str, str]]],
    name_prune: int,
    addr_prune: int,
) -> tuple[dict[str, tuple[str, str, str, set[str]]], dict[tuple[str, str], list[str]], int]:
    target_preprocessed: dict[str, tuple[str, str, str, set[str]]] = {}
    index: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for target_id, (name, address, country) in targets.items():
        clean_name, core_name, _ = normalize_name(name)
        clean_address, numbers, _ = normalize_address(address)
        target_preprocessed[target_id] = (clean_name, core_name, clean_address, numbers)
        for key in get_blocking_keys(name, address, country):
            index[key].append(target_id)
    pruned = 0
    for key in list(index.keys()):
        if len(index[key]) > prune_limit(key, name_prune, addr_prune):
            del index[key]
            pruned += 1
    return target_preprocessed, dict(index), pruned


def retrieve_production_candidates(
    name: str,
    address: str,
    country: str,
    index: Mapping[tuple[str, str], Sequence[str]],
    get_blocking_keys: Callable[[str, str, str], set[tuple[str, str]]],
    top_k: int,
) -> list[tuple[str, int]]:
    """Exact production logic, including set iteration and Counter tie behavior."""
    counts: collections.Counter[str] = collections.Counter()
    for key in get_blocking_keys(name, address, country):
        if key in index:
            counts.update(index[key])
    return counts.most_common(top_k) if counts else []


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "B003 Parquet output requires pyarrow. Install experiments/B003/requirements.txt"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading B003 checkpoints requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def load_ground_truth(path: Path, selected_ids: Sequence[str]) -> dict[str, set[str]]:
    selected = set(selected_ids)
    truth: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0] in selected:
                if parts[0] in truth:
                    raise ValueError(f"Duplicate S1 ID in ground truth: {parts[0]}")
                truth[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
    missing = selected - truth.keys()
    if missing:
        raise ValueError(f"{len(missing)} validation IDs missing from ground truth")
    return {entity_id: truth[entity_id] for entity_id in selected_ids}


def subset_mapping(mapping: Mapping[str, set[str]], ids: Iterable[str]) -> dict[str, set[str]]:
    return {entity_id: set(mapping.get(entity_id, set())) for entity_id in ids}


def source_link_metrics(
    truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], prefix: str
) -> dict[str, Any]:
    tp = fp = fn = 0
    for entity_id, true_values in truth.items():
        true_source = {value for value in true_values if value.startswith(prefix)}
        pred_source = {value for value in predictions.get(entity_id, set()) if value.startswith(prefix)}
        tp += len(true_source & pred_source)
        fp += len(pred_source - true_source)
        fn += len(true_source - pred_source)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def candidate_recall_report(
    selected_ids: Sequence[str],
    truth: Mapping[str, set[str]],
    ranked_candidates: Mapping[str, Sequence[str]],
    countries: Mapping[str, str],
) -> dict[str, Any]:
    def calculate(ids: Sequence[str], prefix: str | None = None) -> dict[str, Any]:
        denominators = 0
        retrieved = {cutoff: 0 for cutoff in RANK_CUTOFFS}
        for entity_id in ids:
            true_values = truth[entity_id]
            if prefix:
                true_values = {value for value in true_values if value.startswith(prefix)}
            denominators += len(true_values)
            candidates = ranked_candidates.get(entity_id, ())
            for cutoff in RANK_CUTOFFS:
                retrieved[cutoff] += len(true_values & set(candidates[:cutoff]))
        return {
            "ground_truth_links": denominators,
            "retrieved_links": {str(k): retrieved[k] for k in RANK_CUTOFFS},
            "recall": {
                str(k): (retrieved[k] / denominators if denominators else None)
                for k in RANK_CUTOFFS
            },
        }

    report: dict[str, Any] = {"overall": calculate(list(selected_ids))}
    report["by_source"] = {
        "S2": calculate(list(selected_ids), "S2-"),
        "S3": calculate(list(selected_ids), "S3-"),
    }
    country_values = sorted(set(countries.values()))
    report["by_country"] = {
        country: calculate([entity_id for entity_id in selected_ids if countries[entity_id] == country])
        for country in country_values
    }
    report["by_country_and_source"] = {
        country: {
            "S2": calculate([entity_id for entity_id in selected_ids if countries[entity_id] == country], "S2-"),
            "S3": calculate([entity_id for entity_id in selected_ids if countries[entity_id] == country], "S3-"),
        }
        for country in country_values
    }
    return report


def candidate_count_report(selected_ids: Sequence[str], ranked: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    counts = [len(ranked.get(entity_id, ())) for entity_id in selected_ids]
    buckets = {"0": 0, "1": 0, "2-5": 0, "6-20": 0, "21-50": 0, ">50": 0}
    for count in counts:
        if count == 0:
            buckets["0"] += 1
        elif count == 1:
            buckets["1"] += 1
        elif count <= 5:
            buckets["2-5"] += 1
        elif count <= 20:
            buckets["6-20"] += 1
        elif count <= 50:
            buckets["21-50"] += 1
        else:
            buckets[">50"] += 1
    ordered = sorted(counts)
    def percentile(value: float) -> float:
        if not ordered:
            return 0.0
        index = int(round((len(ordered) - 1) * value))
        return float(ordered[index])
    return {
        "total_pairs": sum(counts),
        "average": statistics.fmean(counts) if counts else 0.0,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(counts, default=0),
        "buckets": buckets,
    }


def candidate_set_hash(selected_ids: Sequence[str], rows_by_query: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for query_id in selected_ids:
        rows = sorted(rows_by_query.get(query_id, ()), key=lambda row: int(row["candidate_rank"]))
        for row in rows:
            digest.update(
                f"{query_id}\0{row['candidate_id']}\0{row['candidate_rank']}\0{row['shared_keys_count']}\n".encode("utf-8")
            )
    return digest.hexdigest()


def error_decomposition(
    truth: Mapping[str, set[str]],
    score_dict: Mapping[str, Sequence[tuple[str, float]]],
    pre_dedup: Mapping[str, set[str]],
    post_dedup: Mapping[str, set[str]],
    s2_threshold: float,
    s3_threshold: float,
    near_tie_margin: float,
) -> dict[str, Any]:
    not_retrieved = below_threshold = lost_uniqueness = 0
    candidate_ids = {sid: {tid for tid, _ in scores} for sid, scores in score_dict.items()}
    score_lookup = {sid: dict(scores) for sid, scores in score_dict.items()}
    for sid, true_values in truth.items():
        for tid in true_values:
            if tid not in candidate_ids.get(sid, set()):
                not_retrieved += 1
            elif tid not in pre_dedup.get(sid, set()):
                threshold = s2_threshold if tid.startswith("S2-") else s3_threshold
                if score_lookup[sid][tid] < threshold:
                    below_threshold += 1
            elif tid not in post_dedup.get(sid, set()):
                lost_uniqueness += 1

    target_competitors: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for sid, accepted in pre_dedup.items():
        for tid in accepted:
            target_competitors[tid].append((sid, score_lookup[sid][tid]))

    wrong_non_ambiguous = ambiguity_near_tie = other = 0
    for sid, predicted in post_dedup.items():
        for tid in predicted - truth.get(sid, set()):
            competitors = sorted(target_competitors.get(tid, ()), key=lambda item: item[1], reverse=True)
            if len(competitors) >= 2 and competitors[0][1] - competitors[1][1] <= near_tie_margin:
                ambiguity_near_tie += 1
            elif tid in score_lookup.get(sid, {}):
                wrong_non_ambiguous += 1
            else:
                other += 1
    return {
        "false_negatives": {
            "gt_not_retrieved": not_retrieved,
            "gt_retrieved_below_threshold": below_threshold,
            "gt_lost_by_target_uniqueness": lost_uniqueness,
            "total": not_retrieved + below_threshold + lost_uniqueness,
        },
        "false_positives": {
            "wrong_candidate_accepted_non_ambiguous": wrong_non_ambiguous,
            "ambiguity_near_tie": ambiguity_near_tie,
            "other": other,
            "total": wrong_non_ambiguous + ambiguity_near_tie + other,
            "near_tie_margin": near_tie_margin,
        },
    }


def strict_validate(
    selected_ids: Sequence[str],
    predictions: Mapping[str, set[str]],
    candidates_by_query: Mapping[str, set[str]],
    existence_report: Mapping[str, Any],
) -> dict[str, Any]:
    if set(predictions) != set(selected_ids):
        raise ValueError("Predictions do not contain exactly one mapping for every selected S1")
    predicted_ids: set[str] = set()
    candidate_ids = set().union(*candidates_by_query.values()) if candidates_by_query else set()
    for sid in selected_ids:
        values = predictions[sid]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate prediction IDs for {sid}")
        invalid = {value for value in values if not value.startswith(("S2-", "S3-"))}
        if invalid:
            raise ValueError(f"Invalid target namespace for {sid}: {sorted(invalid)[:5]}")
        if not values <= candidates_by_query.get(sid, set()):
            raise ValueError(f"Prediction outside candidate universe for {sid}")
        predicted_ids.update(values)
    return {
        "selected_s1_count": len(selected_ids),
        "predicted_target_count": len(predicted_ids),
        "candidate_target_count": len(candidate_ids),
        "target_existence": dict(existence_report),
        "status": "PASS",
    }


def validate_candidate_existence(
    candidate_ids: set[str], source2: Path, source3: Path, check_existence: bool
) -> dict[str, Any]:
    invalid_namespace = {
        target_id for target_id in candidate_ids
        if not target_id.startswith(("S2-", "S3-"))
    }
    if invalid_namespace:
        raise ValueError(f"Invalid candidate namespace: {sorted(invalid_namespace)[:5]}")
    if not check_existence:
        return {
            "checked": False,
            "candidate_target_count": len(candidate_ids),
            "missing_target_count": None,
            "status": "SKIPPED_BY_FLAG",
        }
    needed = set(candidate_ids)
    for path in (source2, source3):
        for target_id, _, _, _ in iter_tsv(path):
            needed.discard(target_id)
            if not needed:
                break
        if not needed:
            break
    if needed:
        raise ValueError(f"{len(needed)} candidate target IDs do not exist; examples: {sorted(needed)[:5]}")
    return {
        "checked": True,
        "candidate_target_count": len(candidate_ids),
        "missing_target_count": 0,
        "status": "PASS",
    }


def model_description(model: Any, path: Path) -> dict[str, Any]:
    wrapped = getattr(model, "model", None)
    n_features = getattr(wrapped, "n_features_in_", None)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "wrapper_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_class": f"{wrapped.__class__.__module__}.{wrapped.__class__.__name__}" if wrapped is not None else None,
        "model_type": getattr(model, "model_type", None),
        "feature_count": int(n_features) if n_features is not None else len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "parameters": getattr(model, "params", {}),
    }


def base_manifest(args: argparse.Namespace, repo_root: Path, selected: Sequence[str], total_ids: int) -> dict[str, Any]:
    inputs = {
        "source1": str(args.source1.resolve()),
        "source2": str(args.source2.resolve()),
        "source3": str(args.source3.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
    }
    input_hashes = None
    if args.hash_inputs:
        input_hashes = {name: sha256_file(Path(path)) for name, path in inputs.items()}
    semantic_code_paths = [
        Path(__file__).resolve(),
        repo_root / "code" / "business_entity_resolution" / "src" / "normalization.py",
        repo_root / "code" / "business_entity_resolution" / "src" / "blocking.py",
        repo_root / "code" / "business_entity_resolution" / "src" / "features.py",
        repo_root / "code" / "business_entity_resolution" / "src" / "model.py",
        repo_root / "code" / "business_entity_resolution" / "src" / "thresholding.py",
        repo_root / "code" / "business_entity_resolution" / "src" / "evaluation.py",
    ]
    semantic_code_hashes = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in semantic_code_paths
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "INITIALIZED",
        "created_at": utc_now(),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "git_status_porcelain": git_value(repo_root, "status", "--porcelain=v1", "--untracked-files=all"),
        "semantic_code_sha256": semantic_code_hashes,
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "gpu_inventory": gpu_inventory(),
        "top_k": args.top_k,
        "validation_ids_path": str(args.val_ids.resolve()),
        "validation_ids_sha256": sha256_file(args.val_ids),
        "validation_ids_total_count": total_ids,
        "selected_validation_ids_sha256": sha256_ids(selected),
        "selected_validation_count": len(selected),
        "inputs": inputs,
        "input_sha256": input_hashes,
        "threshold_configs": THRESHOLD_CONFIGS,
        "feature_schema": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "candidate_generator": {
            "implementation": "production typed exact blocking keys + Counter.most_common(top_k)",
            "name_prune": args.name_prune,
            "address_prune": args.addr_prune,
            "country_partition": "exact raw country string",
            "ordering": "Counter.most_common; ties preserve production insertion order",
        },
        "models": {
            "A": {"path": str(args.model_a.resolve()), "sha256": sha256_file(args.model_a)},
            "B": {"path": str(args.model_b.resolve()), "sha256": sha256_file(args.model_b)},
        },
        "output_dir": str(args.output_dir.resolve()),
        "outputs": {},
    }


def config_fingerprint(manifest: Mapping[str, Any]) -> str:
    selected = {
        "experiment_id": manifest["experiment_id"],
        "git_commit": manifest["git_commit"],
        "top_k": manifest["top_k"],
        "validation_ids_sha256": manifest["validation_ids_sha256"],
        "selected_validation_ids_sha256": manifest["selected_validation_ids_sha256"],
        "inputs": manifest["inputs"],
        "input_sha256": manifest["input_sha256"],
        "threshold_configs": manifest["threshold_configs"],
        "feature_schema": manifest["feature_schema"],
        "candidate_generator": manifest["candidate_generator"],
        "models": manifest["models"],
        "semantic_code_sha256": manifest["semantic_code_sha256"],
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest()


def import_production(repo_root: Path) -> dict[str, Any]:
    src = repo_root / "code" / "business_entity_resolution" / "src"
    sys.path.insert(0, str(src))
    import normalization as norm  # type: ignore
    from blocking import get_blocking_keys  # type: ignore
    from evaluation import evaluate_predictions  # type: ignore
    from features import FEATURE_NAMES as production_feature_names  # type: ignore
    from features import extract_features_for_pair  # type: ignore
    from model import EntityMatcherModel  # type: ignore
    from thresholding import apply_threshold_and_deduplication  # type: ignore
    if list(production_feature_names) != FEATURE_NAMES:
        raise RuntimeError("B003 feature schema differs from current production FEATURE_NAMES")
    return {
        "norm": norm,
        "get_blocking_keys": get_blocking_keys,
        "evaluate_predictions": evaluate_predictions,
        "extract_features_for_pair": extract_features_for_pair,
        "EntityMatcherModel": EntityMatcherModel,
        "apply_threshold_and_deduplication": apply_threshold_and_deduplication,
    }


def chunked(values: Sequence[str], size: int) -> Iterator[tuple[int, Sequence[str]]]:
    for start in range(0, len(values), size):
        yield start // size, values[start:start + size]


def generate_country_candidates(
    args: argparse.Namespace,
    country: str,
    query_ids: Sequence[str],
    validation_records: Mapping[str, tuple[str, str, str]],
    production: Mapping[str, Any],
    model_a: Any,
    model_b: Any,
    telemetry: Telemetry,
) -> dict[str, Any]:
    raw_dir = args.output_dir / "raw_candidates" / country
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(chunked(query_ids, args.chunk_size))
    incomplete = [
        (index, ids) for index, ids in chunks
        if not (raw_dir / f"chunk_{index:05d}.DONE").exists()
    ]
    if not incomplete:
        telemetry.record("country_candidates_already_complete", country=country, processed_s1=len(query_ids))
        return {"country": country, "resumed": True, "chunks": len(chunks)}

    telemetry.record("load_targets_start", country=country)
    targets = load_targets_for_country(args.source2, args.source3, country)
    telemetry.record("load_targets_done", country=country, extra={"target_records": len(targets)})
    norm = production["norm"]
    target_preprocessed, index, pruned = build_production_index(
        targets, norm.normalize_name, norm.normalize_address,
        production["get_blocking_keys"], args.name_prune, args.addr_prune,
    )
    telemetry.record(
        "index_built", country=country,
        extra={"target_records": len(targets), "active_keys": len(index), "pruned_keys": pruned},
    )
    del targets
    gc.collect()

    processed = 0
    pair_count = 0
    for chunk_index, ids in incomplete:
        rows: list[dict[str, Any]] = []
        features: list[list[float]] = []
        for query_id in ids:
            name, address, record_country = validation_records[query_id]
            candidates = retrieve_production_candidates(
                name, address, record_country, index,
                production["get_blocking_keys"], args.top_k,
            )
            clean_name, core_name, _ = norm.normalize_name(name)
            clean_address, numbers, _ = norm.normalize_address(address)
            source_tuple = (clean_name, core_name, clean_address, numbers)
            for rank, (target_id, shared_count) in enumerate(candidates, start=1):
                rows.append({
                    "query_id": query_id,
                    "candidate_id": target_id,
                    "candidate_source": "S2" if target_id.startswith("S2-") else "S3",
                    "country": record_country,
                    "candidate_rank": rank,
                    "shared_keys_count": int(shared_count),
                })
                features.append(production["extract_features_for_pair"](
                    source_tuple, target_preprocessed[target_id], target_id, shared_count,
                ))
        if features:
            import numpy as np
            matrix = np.asarray(features, dtype=np.float32)
            if matrix.shape[1] != len(FEATURE_NAMES):
                raise RuntimeError(f"Expected {len(FEATURE_NAMES)} features, got {matrix.shape[1]}")
            probabilities_a = model_a.predict_proba(matrix)
            probabilities_b = model_b.predict_proba(matrix)
            for row, probability_a, probability_b in zip(rows, probabilities_a, probabilities_b):
                row["model_probability_A"] = float(probability_a)
                row["model_probability_B"] = float(probability_b)
        output = raw_dir / f"chunk_{chunk_index:05d}.parquet"
        write_parquet(output, rows)
        atomic_text(
            raw_dir / f"chunk_{chunk_index:05d}.DONE",
            json.dumps({"completed_at": utc_now(), "rows": len(rows)}) + "\n",
        )
        processed += len(ids)
        pair_count += len(rows)
        telemetry.record(
            "candidate_chunk_done", country=country, processed_s1=processed,
            candidate_pairs=pair_count, extra={"chunk": chunk_index, "rows": len(rows)},
        )

    del target_preprocessed, index
    gc.collect()
    return {"country": country, "resumed": False, "chunks": len(chunks), "new_pairs": pair_count}


def collect_candidate_rows(output_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((output_dir / "raw_candidates").glob("*/chunk_*.parquet"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        done = path.with_suffix(".DONE")
        if not done.exists():
            raise RuntimeError(f"Candidate checkpoint lacks DONE marker: {path}")
        rows.extend(read_parquet_rows(path))
    return rows


def evaluate_b003(
    args: argparse.Namespace,
    selected_ids: Sequence[str],
    validation_records: Mapping[str, tuple[str, str, str]],
    production: Mapping[str, Any],
    telemetry: Telemetry,
) -> dict[str, Any]:
    telemetry.record("evaluation_load_candidates_start")
    rows = collect_candidate_rows(args.output_dir)
    rows_by_query: dict[str, list[dict[str, Any]]] = {entity_id: [] for entity_id in selected_ids}
    for row in rows:
        if row["query_id"] not in rows_by_query:
            raise ValueError(f"Candidate checkpoint contains unexpected query ID: {row['query_id']}")
        rows_by_query[row["query_id"]].append(row)
    for query_rows in rows_by_query.values():
        query_rows.sort(key=lambda row: int(row["candidate_rank"]))
        candidate_ids = [row["candidate_id"] for row in query_rows]
        ranks = [int(row["candidate_rank"]) for row in query_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Duplicate target ID in a query's candidate checkpoint")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Candidate ranks are not contiguous and one-based")
    ranked = {
        entity_id: [row["candidate_id"] for row in rows_by_query[entity_id]]
        for entity_id in selected_ids
    }
    countries = {entity_id: validation_records[entity_id][2] for entity_id in selected_ids}

    # Labels are intentionally loaded only after raw candidates exist.
    truth = load_ground_truth(args.ground_truth, selected_ids)
    diagnostic_dir = args.output_dir / "diagnostic_candidates"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted((args.output_dir / "raw_candidates").glob("*/chunk_*.parquet")):
        relative = raw_path.relative_to(args.output_dir / "raw_candidates")
        output = diagnostic_dir / relative
        done = output.with_suffix(".DONE")
        if args.resume and done.exists() and output.exists():
            continue
        diagnostic_rows = read_parquet_rows(raw_path)
        for row in diagnostic_rows:
            row["is_ground_truth"] = row["candidate_id"] in truth[row["query_id"]]
        write_parquet(output, diagnostic_rows)
        done.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(done, json.dumps({"completed_at": utc_now(), "rows": len(diagnostic_rows)}) + "\n")

    evaluate_predictions = production["evaluate_predictions"]
    oracle = {
        entity_id: truth[entity_id] & set(ranked.get(entity_id, ()))
        for entity_id in selected_ids
    }
    oracle_metrics = evaluate_predictions(truth, oracle)
    oracle_by_country = {}
    for country in sorted(set(countries.values())):
        ids = [entity_id for entity_id in selected_ids if countries[entity_id] == country]
        oracle_by_country[country] = evaluate_predictions(
            subset_mapping(truth, ids), subset_mapping(oracle, ids)
        )

    results: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "candidate_set_sha256": candidate_set_hash(selected_ids, rows_by_query),
        "validation_s1_count": len(selected_ids),
        "candidate_counts": candidate_count_report(selected_ids, ranked),
        "candidate_recall": candidate_recall_report(selected_ids, truth, ranked, countries),
        "candidate_restricted_oracle": {
            "overall": oracle_metrics,
            "by_country": oracle_by_country,
            "by_source": {
                "S2": source_link_metrics(truth, oracle, "S2-"),
                "S3": source_link_metrics(truth, oracle, "S3-"),
            },
        },
        "models": {},
    }

    candidates_by_query = {
        entity_id: {row["candidate_id"] for row in query_rows}
        for entity_id, query_rows in rows_by_query.items()
    }
    candidate_universe = set().union(*candidates_by_query.values()) if candidates_by_query else set()
    existence_report = validate_candidate_existence(
        candidate_universe, args.source2, args.source3,
        check_existence=not args.skip_strict_id_check,
    )
    results["strict_candidate_validation"] = existence_report
    apply_dedup = production["apply_threshold_and_deduplication"]
    for model_key, probability_column in (("model_A", "model_probability_A"), ("model_B", "model_probability_B")):
        score_dict: dict[str, list[tuple[str, float]]] = {entity_id: [] for entity_id in selected_ids}
        for row in rows:
            score_dict[row["query_id"]].append((row["candidate_id"], float(row[probability_column])))
        model_results = {}
        for config_name, thresholds in THRESHOLD_CONFIGS.items():
            pre = {
                entity_id: {
                    target_id for target_id, probability in scores
                    if probability >= (thresholds["s2"] if target_id.startswith("S2-") else thresholds["s3"])
                }
                for entity_id, scores in score_dict.items()
            }
            post = apply_dedup(score_dict, thresholds["s2"], thresholds["s3"])
            before_metrics = evaluate_predictions(truth, pre)
            after_metrics = evaluate_predictions(truth, post)
            by_country = {}
            for country in sorted(set(countries.values())):
                ids = [entity_id for entity_id in selected_ids if countries[entity_id] == country]
                by_country[country] = {
                    "before_dedup": evaluate_predictions(subset_mapping(truth, ids), subset_mapping(pre, ids)),
                    "after_dedup": evaluate_predictions(subset_mapping(truth, ids), subset_mapping(post, ids)),
                }
            validation = strict_validate(
                selected_ids, post, candidates_by_query, existence_report,
            )
            prediction_path = args.output_dir / "predictions" / model_key / f"{config_name}.parquet"
            write_parquet(prediction_path, [
                {
                    "query_id": entity_id,
                    "country": countries[entity_id],
                    "before_dedup_target_ids": sorted(pre.get(entity_id, set())),
                    "after_dedup_target_ids": sorted(post.get(entity_id, set())),
                }
                for entity_id in selected_ids
            ])
            model_results[config_name] = {
                "thresholds": thresholds,
                "candidate_set_sha256": results["candidate_set_sha256"],
                "top_k": args.top_k,
                "before_dedup": before_metrics,
                "after_production_dedup": after_metrics,
                "by_country": by_country,
                "by_source_after_dedup": {
                    "S2": source_link_metrics(truth, post, "S2-"),
                    "S3": source_link_metrics(truth, post, "S3-"),
                },
                "error_decomposition": error_decomposition(
                    truth, score_dict, pre, post, thresholds["s2"], thresholds["s3"],
                    args.near_tie_margin,
                ),
                "strict_validation": validation,
                "predictions_path": str(prediction_path.resolve()),
            }
            decomposition = model_results[config_name]["error_decomposition"]
            if decomposition["false_negatives"]["total"] != after_metrics["total_fn"]:
                raise RuntimeError("False-negative decomposition does not sum to official FN")
            if decomposition["false_positives"]["total"] != after_metrics["total_fp"]:
                raise RuntimeError("False-positive decomposition does not sum to official FP")
        results["models"][model_key] = model_results

    oracle_f05 = float(oracle_metrics["macro_f05"])
    if oracle_f05 < 0.985:
        recommendation = "CASE A: candidate retrieval is a severe bottleneck; prioritize retrieval work."
    elif oracle_f05 < 0.995:
        recommendation = "CASE B: retrieval and matching both matter; prioritize K50 hard-negative remine while preparing retrieval improvements."
    else:
        recommendation = "CASE C: retrieval is largely solved; prioritize K50 negatives, provenance, direct evidence, numeric contradictions, competition, and decision work."
    results["heuristic_recommendation"] = recommendation
    telemetry.record("evaluation_complete", processed_s1=len(selected_ids), candidate_pairs=len(rows))
    return results


def print_summary(metrics: Mapping[str, Any], telemetry: Telemetry) -> None:
    counts = metrics["candidate_counts"]
    recall = metrics["candidate_recall"]["overall"]["recall"]
    oracle = metrics["candidate_restricted_oracle"]["overall"]
    print("\nB003 FULL-CORPUS K50")
    print(f"Validation S1: {metrics['validation_s1_count']}")
    print(f"Candidate pairs: {counts['total_pairs']}")
    print("\nCandidate Recall")
    for cutoff in RANK_CUTOFFS:
        value = recall[str(cutoff)]
        print(f"@{cutoff:<2}  {'NA' if value is None else f'{value:.6f}'}")
    print(f"\nCandidate-restricted Oracle Macro F0.5: {oracle['macro_f05']:.6f}")
    for model_key, configurations in metrics["models"].items():
        print(f"\n{model_key.upper()}")
        for config_name, result in configurations.items():
            final = result["after_production_dedup"]
            print(f"  {config_name}")
            print(
                f"    Macro F0.5={final['macro_f05']:.6f} "
                f"Precision={final['global_precision']:.6f} "
                f"Recall={final['global_recall']:.6f}"
            )
            errors = result["error_decomposition"]["false_negatives"]
            print(
                "    FN: not_retrieved={gt_not_retrieved} "
                "below_threshold={gt_retrieved_below_threshold} "
                "lost_uniqueness={gt_lost_by_target_uniqueness}".format(**errors)
            )
    print(f"\nPeak RSS: {telemetry.peak_rss if telemetry.peak_rss else 'NA'} bytes")
    print(f"Runtime: {time.monotonic() - telemetry.started:.1f}s")
    print(f"Heuristic recommendation: {metrics['heuristic_recommendation']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model-a", type=Path, default=Path("models/final_entity_matcher.joblib"))
    parser.add_argument("--model-b", type=Path, default=Path("models/baseline_reproduced.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B003"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--name-prune", type=int, default=300)
    parser.add_argument("--addr-prune", type=int, default=150)
    parser.add_argument("--near-tie-margin", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--skip-strict-id-check", action="store_true")
    parser.add_argument("--manifest-only", action="store_true", help="Write and validate the initial manifest without loading models/data")
    args = parser.parse_args(argv)
    if args.top_k != 50:
        parser.error("B003 is frozen at --top-k 50")
    if args.val_limit <= 0 or args.chunk_size <= 0:
        parser.error("--val-limit and --chunk-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    for path in (args.source1, args.source2, args.source3, args.ground_truth, args.val_ids, args.model_a, args.model_b):
        if not path.exists():
            raise FileNotFoundError(path)
    selected_ids, total_ids = load_selected_ids(args.val_ids, args.val_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = base_manifest(args, repo_root, selected_ids, total_ids)
    if args.manifest_only:
        manifest["status"] = "MANIFEST_ONLY"
        manifest["config_fingerprint"] = config_fingerprint(manifest)
        atomic_json(manifest_path, manifest)
        print(f"Manifest written: {manifest_path}")
        return 0

    production = import_production(repo_root)
    model_a = production["EntityMatcherModel"].load(str(args.model_a))
    model_b = production["EntityMatcherModel"].load(str(args.model_b))
    manifest["models"] = {
        "A": model_description(model_a, args.model_a),
        "B": model_description(model_b, args.model_b),
    }
    for model_key, description in manifest["models"].items():
        if description["feature_count"] != len(FEATURE_NAMES):
            raise RuntimeError(f"Model {model_key} expects {description['feature_count']} features, not {len(FEATURE_NAMES)}")
    manifest["config_fingerprint"] = config_fingerprint(manifest)
    if args.resume and manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise RuntimeError("Refusing to resume: B003 configuration fingerprint changed")
    elif manifest_path.exists() and any((args.output_dir / "raw_candidates").glob("*/chunk_*.DONE")):
        raise RuntimeError("B003 checkpoints exist. Use --resume or choose a new --output-dir")
    manifest["status"] = "RUNNING"
    atomic_json(manifest_path, manifest)

    telemetry = Telemetry(args.output_dir / "telemetry.jsonl")
    telemetry.record("start", extra={"gpu_inventory": manifest["gpu_inventory"]})
    validation_records, country_to_ids = load_validation_records(args.source1, selected_ids)
    manifest["validation_country_counts"] = {country: len(ids) for country, ids in country_to_ids.items()}
    atomic_json(manifest_path, manifest)

    for country in sorted(country_to_ids, key=lambda value: len(country_to_ids[value])):
        generate_country_candidates(
            args, country, country_to_ids[country], validation_records,
            production, model_a, model_b, telemetry,
        )

    metrics = evaluate_b003(args, selected_ids, validation_records, production, telemetry)
    metrics["runtime"] = {
        "total_seconds": time.monotonic() - telemetry.started,
        "peak_rss_bytes": telemetry.peak_rss or None,
    }
    metrics_path = args.output_dir / "metrics.json"
    candidate_metrics_path = args.output_dir / "candidate_metrics.json"
    atomic_json(metrics_path, metrics)
    atomic_json(candidate_metrics_path, {
        "schema_version": metrics["schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "candidate_set_sha256": metrics["candidate_set_sha256"],
        "candidate_counts": metrics["candidate_counts"],
        "candidate_recall": metrics["candidate_recall"],
        "candidate_restricted_oracle": metrics["candidate_restricted_oracle"],
    })
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = utc_now()
    manifest["outputs"] = {
        "metrics": str(metrics_path.resolve()),
        "candidate_metrics": str(candidate_metrics_path.resolve()),
        "telemetry": str(telemetry.path.resolve()),
        "raw_candidates": str((args.output_dir / "raw_candidates").resolve()),
        "diagnostic_candidates": str((args.output_dir / "diagnostic_candidates").resolve()),
        "predictions": str((args.output_dir / "predictions").resolve()),
    }
    manifest["candidate_set_sha256"] = metrics["candidate_set_sha256"]
    manifest["runtime"] = metrics["runtime"]
    atomic_json(manifest_path, manifest)
    atomic_text(args.output_dir / "B003.DONE", utc_now() + "\n")
    print_summary(metrics, telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
