#!/usr/bin/env python3
"""B004: multi-view, high-recall sparse retrieval evaluation.

R0 is the frozen production typed-key blocker. R1-R3 are target-fitted sparse
TF-IDF channels. This experiment never trains or scores a matcher. Ground truth is
loaded only after every retrieval checkpoint is complete.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

EXPERIMENT_ID = "B004"
CHANNELS = ("R0_EXISTING", "R1_NAME_CHAR", "R2_ADDRESS_WORD", "R3_NAME_WORD")
ABlations = (
    ("R0", ("R0_EXISTING",)),
    ("R0+R1", ("R0_EXISTING", "R1_NAME_CHAR")),
    ("R0+R1+R2", ("R0_EXISTING", "R1_NAME_CHAR", "R2_ADDRESS_WORD")),
    ("R0+R1+R2+R3", CHANNELS),
)
R0_EXPECTED_RECALL_50 = 0.908381
R0_EXPECTED_ORACLE_F05 = 0.959286
PROVENANCE_FIELDS = (
    "found_by_existing", "found_by_name_char", "found_by_address_word",
    "found_by_name_word", "found_by_reverse", "existing_rank",
    "name_char_rank", "address_word_rank", "name_word_rank", "reverse_rank",
    "existing_shared_keys", "name_char_score", "address_word_score",
    "name_word_score", "reverse_score", "retriever_count", "rrf_score",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_b003() -> Any:
    path = repo_root() / "experiments" / "B003"
    sys.path.insert(0, str(path))
    import run_b003  # type: ignore
    return run_b003


B003 = import_b003()


def import_retrieval_modules() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    src = repo_root() / "code" / "business_entity_resolution" / "src"
    sys.path.insert(0, str(src))
    import normalization as norm  # type: ignore
    from blocking import get_blocking_keys  # type: ignore
    from evaluation import evaluate_predictions  # type: ignore
    return norm, get_blocking_keys, evaluate_predictions


def atomic_done(path: Path, payload: Mapping[str, Any]) -> None:
    B003.atomic_text(path, json.dumps(dict(payload), sort_keys=True) + "\n")


def load_selected_records(
    source1: Path, selected_ids: Sequence[str]
) -> tuple[dict[str, tuple[str, str, str]], dict[str, list[str]]]:
    return B003.load_validation_records(source1, selected_ids)


def load_source_country(
    source_path: Path,
    country: str,
    telemetry: Any,
    progress_every: int,
) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    scanned = 0
    for entity_id, name, address, record_country in B003.iter_tsv(source_path):
        scanned += 1
        if record_country == country:
            records.append((entity_id, name, address))
        if scanned % progress_every == 0:
            telemetry.record(
                "tfidf_target_scan_progress", country=country,
                extra={"source": source_path.name, "rows_scanned": scanned, "country_rows": len(records)},
            )
    telemetry.record(
        "tfidf_target_scan_done", country=country,
        extra={"source": source_path.name, "rows_scanned": scanned, "country_rows": len(records)},
    )
    return records


def normalize_channel_text(channel: str, name: str, address: str, norm: Any) -> str:
    if channel in ("R1_NAME_CHAR", "R3_NAME_WORD"):
        _, core_name, _ = norm.normalize_name(name)
        return core_name
    if channel == "R2_ADDRESS_WORD":
        clean_address, _, _ = norm.normalize_address(address)
        return clean_address
    raise ValueError(f"Unsupported TF-IDF channel: {channel}")


def vectorizer_for(channel: str, args: argparse.Namespace) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    common = {
        "lowercase": False,
        "dtype": np.float32,
        "norm": "l2",
        "sublinear_tf": True,
        "min_df": args.tfidf_min_df,
        "strip_accents": None,
    }
    if channel == "R1_NAME_CHAR":
        return TfidfVectorizer(
            analyzer=args.name_char_analyzer,
            ngram_range=(args.name_char_ngram_min, args.name_char_ngram_max),
            max_features=args.name_char_max_features,
            **common,
        )
    if channel == "R2_ADDRESS_WORD":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
            max_features=args.address_word_max_features, **common,
        )
    if channel == "R3_NAME_WORD":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
            max_features=args.name_word_max_features, **common,
        )
    raise ValueError(channel)


def sparse_topn(
    query_matrix: Any,
    target_matrix: Any,
    top_k: int,
    min_score: float,
    n_threads: int,
) -> Any:
    """Return a CSR query×target matrix containing row-wise top cosine scores."""
    try:
        import sparse_dot_topn  # type: ignore
        if hasattr(sparse_dot_topn, "sp_matmul_topn"):
            result = sparse_dot_topn.sp_matmul_topn(
                query_matrix,
                target_matrix.T.tocsc(),
                top_n=top_k,
                threshold=min_score,
                sort=True,
                n_threads=n_threads,
            )
        else:
            result = sparse_dot_topn.awesome_cossim_topn(
                query_matrix,
                target_matrix.T.tocsc(),
                ntop=top_k,
                lower_bound=min_score,
                use_threads=n_threads > 1,
                n_jobs=n_threads,
            )
        return result.tocsr()
    except ImportError:
        if target_matrix.shape[0] > 100_000:
            raise RuntimeError(
                "sparse-dot-topn is required for full B004. Install experiments/B004/requirements.txt"
            )
        similarities = (query_matrix @ target_matrix.T).tocsr()
        from scipy.sparse import csr_matrix
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for row_index in range(similarities.shape[0]):
            start, end = similarities.indptr[row_index:row_index + 2]
            pairs = [
                (int(index), float(score))
                for index, score in zip(similarities.indices[start:end], similarities.data[start:end])
                if score >= min_score
            ]
            pairs.sort(key=lambda item: (-item[1], item[0]))
            for index, score in pairs[:top_k]:
                indices.append(index)
                data.append(score)
            indptr.append(len(indices))
        return csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=similarities.shape,
        )


def topn_rows(
    result: Any,
    query_ids: Sequence[str],
    target_ids: Sequence[str],
    source: str,
    country: str,
    channel: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_query, query_id in enumerate(query_ids):
        start, end = result.indptr[local_query:local_query + 2]
        pairs = [
            (int(target_index), float(score))
            for target_index, score in zip(result.indices[start:end], result.data[start:end])
        ]
        pairs.sort(key=lambda item: (-item[1], target_ids[item[0]]))
        for rank, (target_index, score) in enumerate(pairs, start=1):
            rows.append({
                "query_id": query_id,
                "candidate_id": target_ids[target_index],
                "candidate_source": source,
                "country": country,
                "channel": channel,
                "rank": rank,
                "score": score,
            })
    return rows


def channel_chunk_paths(
    output_dir: Path, channel: str, country: str, source: str, chunk_index: int
) -> tuple[Path, Path]:
    base = output_dir / "channels" / channel / country / source
    parquet = base / f"chunk_{chunk_index:05d}.parquet"
    return parquet, parquet.with_suffix(".DONE")


def all_chunks_done(
    output_dir: Path,
    channel: str,
    country: str,
    source: str,
    query_count: int,
    chunk_size: int,
) -> bool:
    chunks = (query_count + chunk_size - 1) // chunk_size
    return all(
        channel_chunk_paths(output_dir, channel, country, source, index)[1].exists()
        for index in range(chunks)
    )


def run_r0_country(
    args: argparse.Namespace,
    country: str,
    query_ids: Sequence[str],
    validation_records: Mapping[str, tuple[str, str, str]],
    norm: Any,
    get_blocking_keys: Callable[..., Any],
    telemetry: Any,
) -> None:
    source = "S2_S3"
    if all_chunks_done(args.output_dir, "R0_EXISTING", country, source, len(query_ids), args.query_chunk_size):
        telemetry.record("r0_already_complete", country=country, processed_s1=len(query_ids))
        return
    targets = B003.load_targets_for_country(
        args.source2, args.source3, country,
        progress=lambda path, scanned, matched: telemetry.record(
            "r0_target_scan_progress", country=country,
            extra={"source": path.name, "rows_scanned": scanned, "country_rows": matched},
        ),
        progress_every=args.progress_every_targets,
    )
    target_preprocessed, index, pruned = B003.build_production_index(
        targets, norm.normalize_name, norm.normalize_address, get_blocking_keys,
        args.r0_name_prune, args.r0_addr_prune,
        progress=lambda processed, keys: telemetry.record(
            "r0_index_progress", country=country,
            extra={"targets_processed": processed, "unpruned_keys": keys},
        ),
        progress_every=args.progress_every_targets,
    )
    del target_preprocessed
    telemetry.record(
        "r0_index_ready", country=country,
        extra={"target_count": len(targets), "active_keys": len(index), "pruned_keys": pruned},
    )
    del targets
    for chunk_index, chunk_ids in B003.chunked(query_ids, args.query_chunk_size):
        parquet, done = channel_chunk_paths(
            args.output_dir, "R0_EXISTING", country, source, chunk_index
        )
        if done.exists():
            continue
        rows = []
        for query_id in chunk_ids:
            name, address, record_country = validation_records[query_id]
            candidates = B003.retrieve_production_candidates(
                name, address, record_country, index, get_blocking_keys, args.r0_k,
            )
            for rank, (candidate_id, shared_keys) in enumerate(candidates, start=1):
                rows.append({
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "candidate_source": "S2" if candidate_id.startswith("S2-") else "S3",
                    "country": country,
                    "channel": "R0_EXISTING",
                    "rank": rank,
                    "score": float(shared_keys),
                })
        B003.write_parquet(parquet, rows)
        atomic_done(done, {"completed_at": B003.utc_now(), "rows": len(rows)})
        telemetry.record(
            "r0_chunk_done", country=country,
            processed_s1=min((chunk_index + 1) * args.query_chunk_size, len(query_ids)),
            candidate_pairs=len(rows), extra={"chunk": chunk_index},
        )


def run_tfidf_source_country(
    args: argparse.Namespace,
    country: str,
    source: str,
    source_path: Path,
    query_ids: Sequence[str],
    validation_records: Mapping[str, tuple[str, str, str]],
    norm: Any,
    telemetry: Any,
) -> None:
    pending_channels = [
        channel for channel in ("R1_NAME_CHAR", "R2_ADDRESS_WORD", "R3_NAME_WORD")
        if not all_chunks_done(
            args.output_dir, channel, country, source,
            len(query_ids), args.query_chunk_size,
        )
    ]
    if not pending_channels:
        telemetry.record("tfidf_source_already_complete", country=country, extra={"source": source})
        return
    target_records = load_source_country(
        source_path, country, telemetry, args.progress_every_targets
    )
    target_ids = [record[0] for record in target_records]
    for channel in pending_channels:
        telemetry.record(
            "tfidf_fit_start", country=country,
            extra={"source": source, "channel": channel, "targets": len(target_ids)},
        )
        target_texts = [
            normalize_channel_text(channel, name, address, norm)
            for _, name, address in target_records
        ]
        vectorizer = vectorizer_for(channel, args)
        target_matrix = vectorizer.fit_transform(target_texts)
        telemetry.record(
            "tfidf_fit_done", country=country,
            extra={
                "source": source, "channel": channel,
                "targets": target_matrix.shape[0], "features": target_matrix.shape[1],
                "nnz": target_matrix.nnz,
            },
        )
        del target_texts
        for chunk_index, chunk_ids in B003.chunked(query_ids, args.query_chunk_size):
            parquet, done = channel_chunk_paths(
                args.output_dir, channel, country, source, chunk_index
            )
            if done.exists():
                continue
            query_texts = [
                normalize_channel_text(
                    channel,
                    validation_records[query_id][0],
                    validation_records[query_id][1],
                    norm,
                )
                for query_id in chunk_ids
            ]
            query_matrix = vectorizer.transform(query_texts)
            result = sparse_topn(
                query_matrix, target_matrix, args.tfidf_k,
                args.tfidf_min_score, args.sparse_threads,
            )
            rows = topn_rows(result, chunk_ids, target_ids, source, country, channel)
            B003.write_parquet(parquet, rows)
            atomic_done(done, {"completed_at": B003.utc_now(), "rows": len(rows)})
            telemetry.record(
                "tfidf_chunk_done", country=country,
                processed_s1=min((chunk_index + 1) * args.query_chunk_size, len(query_ids)),
                candidate_pairs=len(rows),
                extra={"source": source, "channel": channel, "chunk": chunk_index},
            )
        del target_matrix, vectorizer


def load_channel_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet in sorted((output_dir / "channels").glob("*/*/*/chunk_*.parquet")):
        if not parquet.with_suffix(".DONE").exists():
            raise RuntimeError(f"Missing DONE marker: {parquet}")
        rows.extend(B003.read_parquet_rows(parquet))
    return rows


def empty_evidence(query_id: str, candidate_id: str, source: str, country: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "candidate_id": candidate_id,
        "candidate_source": source,
        "country": country,
        "found_by_existing": False,
        "found_by_name_char": False,
        "found_by_address_word": False,
        "found_by_name_word": False,
        "found_by_reverse": False,
        "existing_rank": None,
        "name_char_rank": None,
        "address_word_rank": None,
        "name_word_rank": None,
        "reverse_rank": None,
        "existing_shared_keys": None,
        "name_char_score": None,
        "address_word_score": None,
        "name_word_score": None,
        "reverse_score": None,
        "retriever_count": 0,
        "rrf_score": 0.0,
    }


def merge_provenance(rows: Iterable[Mapping[str, Any]], rrf_constant: float) -> dict[tuple[str, str], dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    field_map = {
        "R0_EXISTING": ("found_by_existing", "existing_rank", "existing_shared_keys"),
        "R1_NAME_CHAR": ("found_by_name_char", "name_char_rank", "name_char_score"),
        "R2_ADDRESS_WORD": ("found_by_address_word", "address_word_rank", "address_word_score"),
        "R3_NAME_WORD": ("found_by_name_word", "name_word_rank", "name_word_score"),
    }
    for row in rows:
        key = (str(row["query_id"]), str(row["candidate_id"]))
        evidence = merged.setdefault(
            key,
            empty_evidence(
                key[0], key[1], str(row["candidate_source"]), str(row["country"]),
            ),
        )
        channel = str(row["channel"])
        found_field, rank_field, score_field = field_map[channel]
        rank = int(row["rank"])
        score = float(row["score"])
        previous_rank = evidence[rank_field]
        if previous_rank is None or rank < previous_rank:
            evidence[rank_field] = rank
            evidence[score_field] = score
        evidence[found_field] = True
    for evidence in merged.values():
        evidence["retriever_count"] = sum(
            int(evidence[field]) for field in (
                "found_by_existing", "found_by_name_char",
                "found_by_address_word", "found_by_name_word", "found_by_reverse",
            )
        )
        evidence["rrf_score"] = sum(
            1.0 / (rrf_constant + float(evidence[rank_field]))
            for rank_field in (
                "existing_rank", "name_char_rank", "address_word_rank",
                "name_word_rank", "reverse_rank",
            )
            if evidence[rank_field] is not None
        )
    return merged


def evidence_has_channel(evidence: Mapping[str, Any], channel: str) -> bool:
    return bool(evidence[{
        "R0_EXISTING": "found_by_existing",
        "R1_NAME_CHAR": "found_by_name_char",
        "R2_ADDRESS_WORD": "found_by_address_word",
        "R3_NAME_WORD": "found_by_name_word",
    }[channel]])


def capped_rrf(
    candidates: Sequence[Mapping[str, Any]],
    cap: int,
    r0_guarantee: int,
) -> list[Mapping[str, Any]]:
    guaranteed = sorted(
        [
            row for row in candidates
            if row["existing_rank"] is not None and int(row["existing_rank"]) <= r0_guarantee
        ],
        key=lambda row: (int(row["existing_rank"]), str(row["candidate_id"])),
    )[:cap]
    selected = {(row["query_id"], row["candidate_id"]) for row in guaranteed}
    remaining = sorted(
        [row for row in candidates if (row["query_id"], row["candidate_id"]) not in selected],
        key=lambda row: (
            -float(row["rrf_score"]),
            -int(bool(row["found_by_existing"])),
            int(row["existing_rank"]) if row["existing_rank"] is not None else 10**9,
            str(row["candidate_id"]),
        ),
    )
    return guaranteed + remaining[:max(0, cap - len(guaranteed))]


def evaluate_candidate_set(
    name: str,
    channels: Sequence[str],
    evidence_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_ids: Sequence[str],
    truth: Mapping[str, set[str]],
    countries: Mapping[str, str],
    evaluate_predictions: Callable[..., Mapping[str, Any]],
    cap: int | None,
    r0_guarantee: int,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    ranked: dict[str, list[str]] = {}
    for query_id in selected_ids:
        eligible = [
            row for row in evidence_by_query.get(query_id, ())
            if any(evidence_has_channel(row, channel) for channel in channels)
        ]
        if cap is not None:
            eligible = list(capped_rrf(eligible, cap, r0_guarantee))
        else:
            eligible.sort(key=lambda row: (-float(row["rrf_score"]), str(row["candidate_id"])))
        ranked[query_id] = [str(row["candidate_id"]) for row in eligible]
    oracle = {query_id: truth[query_id] & set(ranked[query_id]) for query_id in selected_ids}
    oracle_metrics = evaluate_predictions(truth, oracle)
    recall = B003.candidate_recall_report(selected_ids, truth, ranked, countries)
    counts = B003.candidate_count_report(selected_ids, ranked)

    def full_link_recall(ids: Sequence[str], prefix: str | None = None) -> dict[str, Any]:
        denominator = retrieved = 0
        for query_id in ids:
            true_values = truth[query_id]
            if prefix is not None:
                true_values = {value for value in true_values if value.startswith(prefix)}
            denominator += len(true_values)
            retrieved += len(true_values & set(ranked[query_id]))
        return {
            "ground_truth_links": denominator,
            "retrieved_links": retrieved,
            "recall": retrieved / denominator if denominator else None,
        }

    oracle_by_country = {}
    full_by_country = {}
    for country in sorted(set(countries.values())):
        country_ids = [query_id for query_id in selected_ids if countries[query_id] == country]
        oracle_by_country[country] = evaluate_predictions(
            B003.subset_mapping(truth, country_ids),
            B003.subset_mapping(oracle, country_ids),
        )
        full_by_country[country] = full_link_recall(country_ids)
    full_by_source = {
        "S2": full_link_recall(list(selected_ids), "S2-"),
        "S3": full_link_recall(list(selected_ids), "S3-"),
    }
    overall_full = full_link_recall(list(selected_ids))
    result = {
        "name": name,
        "channels": list(channels),
        "cap": cap,
        "candidate_pairs": counts["total_pairs"],
        "average_candidates_per_s1": counts["average"],
        "candidate_counts": counts,
        "gt_link_recall": overall_full["recall"],
        "candidate_restricted_oracle": oracle_metrics,
        "oracle_by_country": oracle_by_country,
        "full_recall_by_country": full_by_country,
        "full_recall_by_source": full_by_source,
        "rank_cutoff_diagnostics": recall,
    }
    return result, ranked


def unique_channel_contribution(
    evidence: Iterable[Mapping[str, Any]], truth: Mapping[str, set[str]]
) -> dict[str, Any]:
    report = {}
    for channel in CHANNELS:
        channel_rows = [row for row in evidence if evidence_has_channel(row, channel)]
        gt_rows = [row for row in channel_rows if row["candidate_id"] in truth[row["query_id"]]]
        unique_gt = [row for row in gt_rows if int(row["retriever_count"]) == 1]
        report[channel] = {
            "candidate_pairs": len(channel_rows),
            "gt_links_found": len(gt_rows),
            "gt_links_uniquely_recovered": len(unique_gt),
            "gt_links_also_found_by_other_channel": len(gt_rows) - len(unique_gt),
            "false_candidate_pairs_added": len(channel_rows) - len(gt_rows),
            "retrieval_precision": len(gt_rows) / len(channel_rows) if channel_rows else None,
        }
    report["R4_REVERSE"] = {"status": "NOT_IMPLEMENTED_IN_B004", "reason": "Optional channel deferred so R1-R3 are not blocked"}
    return report


def build_missed_examples(
    args: argparse.Namespace,
    missing_pairs: Sequence[tuple[str, str]],
    validation_records: Mapping[str, tuple[str, str, str]],
) -> list[dict[str, Any]]:
    wanted = {target_id for _, target_id in missing_pairs[:args.max_missed_examples]}
    target_records: dict[str, tuple[str, str, str]] = {}
    for path in (args.source2, args.source3):
        for target_id, name, address, country in B003.iter_tsv(path):
            if target_id in wanted:
                target_records[target_id] = (name, address, country)
            if len(target_records) == len(wanted):
                break
    examples = []
    for query_id, target_id in missing_pairs[:args.max_missed_examples]:
        source_name, source_address, source_country = validation_records[query_id]
        target_name, target_address, target_country = target_records.get(target_id, (None, None, None))
        examples.append({
            "query_id": query_id, "candidate_id": target_id,
            "source": "S2" if target_id.startswith("S2-") else "S3",
            "country": source_country,
            "query_name": source_name, "query_address": source_address,
            "target_name": target_name, "target_address": target_address,
            "target_country": target_country,
        })
    return examples


def manifest(args: argparse.Namespace, selected_ids: Sequence[str], total_ids: int) -> dict[str, Any]:
    root = repo_root()
    code_paths = [
        Path(__file__).resolve(),
        root / "experiments" / "B003" / "run_b003.py",
        root / "code" / "business_entity_resolution" / "src" / "normalization.py",
        root / "code" / "business_entity_resolution" / "src" / "blocking.py",
        root / "code" / "business_entity_resolution" / "src" / "evaluation.py",
    ]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "INITIALIZED",
        "created_at": B003.utc_now(),
        "git_branch": B003.git_value(root, "branch", "--show-current"),
        "git_commit": B003.git_value(root, "rev-parse", "HEAD"),
        "git_status": B003.git_value(root, "status", "--porcelain=v1", "--untracked-files=all"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "semantic_code_sha256": {
            path.relative_to(root).as_posix(): B003.sha256_file(path) for path in code_paths
        },
        "validation": {
            "path": str(args.val_ids.resolve()),
            "file_sha256": B003.sha256_file(args.val_ids),
            "total_ids": total_ids,
            "selected_count": len(selected_ids),
            "selected_sha256": B003.sha256_ids(selected_ids),
        },
        "inputs": {
            "source1": str(args.source1.resolve()), "source2": str(args.source2.resolve()),
            "source3": str(args.source3.resolve()), "ground_truth": str(args.ground_truth.resolve()),
        },
        "retrieval": {
            "R0": {"k": args.r0_k, "name_prune": args.r0_name_prune, "address_prune": args.r0_addr_prune},
            "R1": {
                "analyzer": args.name_char_analyzer,
                "ngram_range": [args.name_char_ngram_min, args.name_char_ngram_max],
                "k_per_source": args.tfidf_k,
                "max_features": args.name_char_max_features,
            },
            "R2": {"analyzer": "word", "ngram_range": [1, 2], "k_per_source": args.tfidf_k, "max_features": args.address_word_max_features},
            "R3": {"analyzer": "word", "ngram_range": [1, 2], "k_per_source": args.tfidf_k, "max_features": args.name_word_max_features},
            "R4": {"enabled": False, "reason": "optional reverse channel deferred"},
            "fit_corpus": "target records only, separately for each country and source; no labels",
            "tfidf_min_df": args.tfidf_min_df,
            "tfidf_min_score": args.tfidf_min_score,
            "union": "canonical (query_id, candidate_id, candidate_source), uncapped",
            "cap_analysis": {"cap": args.union_cap, "rrf_constant": args.rrf_constant, "guaranteed_r0_top": args.r0_guarantee},
        },
        "candidate_rule_audit": {
            "hard_per_s1_limit_found": False,
            "evidence": "PS.txt describes candidate_pairs as the exact final model input and specifies no per-S1 cap",
        },
        "output_dir": str(args.output_dir.resolve()),
    }


def config_fingerprint(value: Mapping[str, Any]) -> str:
    keys = {
        "experiment_id": value["experiment_id"], "git_commit": value["git_commit"],
        "semantic_code_sha256": value["semantic_code_sha256"],
        "validation": value["validation"], "inputs": value["inputs"],
        "retrieval": value["retrieval"],
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode("utf-8")).hexdigest()


def evaluate(args: argparse.Namespace, selected_ids: Sequence[str], validation_records: Mapping[str, tuple[str, str, str]], evaluate_predictions: Callable[..., Any], telemetry: Any) -> dict[str, Any]:
    rows = load_channel_rows(args.output_dir)
    merged = merge_provenance(rows, args.rrf_constant)
    evidence_by_query: dict[str, list[dict[str, Any]]] = {query_id: [] for query_id in selected_ids}
    for evidence in merged.values():
        evidence_by_query[evidence["query_id"]].append(evidence)
    truth = B003.load_ground_truth(args.ground_truth, selected_ids)
    countries = {query_id: validation_records[query_id][2] for query_id in selected_ids}
    for evidence in merged.values():
        evidence["is_ground_truth"] = evidence["candidate_id"] in truth[evidence["query_id"]]
    query_order = {query_id: index for index, query_id in enumerate(selected_ids)}
    union_rows = sorted(
        merged.values(),
        key=lambda row: (query_order[row["query_id"]], -row["rrf_score"], row["candidate_id"]),
    )
    B003.write_parquet(args.output_dir / "union" / "uncapped_provenance.parquet", union_rows)

    ablations = []
    capped_ablations = []
    previous_gt = previous_pairs = 0
    best_ranked: dict[str, list[str]] = {}
    for name, channels in ABlations:
        result, ranked = evaluate_candidate_set(
            name, channels, evidence_by_query, selected_ids, truth, countries,
            evaluate_predictions, None, args.r0_guarantee,
        )
        gt_found = sum(len(truth[q] & set(ranked[q])) for q in selected_ids)
        result["incremental_gt_links_recovered"] = gt_found - previous_gt
        result["incremental_candidate_pairs"] = result["candidate_pairs"] - previous_pairs
        previous_gt, previous_pairs = gt_found, result["candidate_pairs"]
        ablations.append(result)
        best_ranked = ranked
        capped_result, _ = evaluate_candidate_set(
            name, channels, evidence_by_query, selected_ids, truth, countries,
            evaluate_predictions, args.union_cap, args.r0_guarantee,
        )
        capped_ablations.append(capped_result)

    r0 = ablations[0]
    reproduction = {
        "expected_recall_50": R0_EXPECTED_RECALL_50,
        "observed_recall": r0["gt_link_recall"],
        "expected_oracle_f05": R0_EXPECTED_ORACLE_F05,
        "observed_oracle_f05": r0["candidate_restricted_oracle"]["macro_f05"],
        "tolerance": args.r0_tolerance,
    }
    reproduction["status"] = "PASS" if (
        abs(reproduction["observed_recall"] - R0_EXPECTED_RECALL_50) <= args.r0_tolerance
        and abs(reproduction["observed_oracle_f05"] - R0_EXPECTED_ORACLE_F05) <= args.r0_tolerance
    ) else "FAIL"

    best = ablations[-1]
    missing_pairs = [
        (query_id, target_id)
        for query_id in selected_ids
        for target_id in sorted(truth[query_id] - set(best_ranked[query_id]))
    ]
    missed_examples = build_missed_examples(args, missing_pairs, validation_records)
    B003.atomic_json(args.output_dir / "missed_ground_truth.json", {
        "missing_pair_count": len(missing_pairs), "examples": missed_examples,
    })
    oracle_f05 = float(best["candidate_restricted_oracle"]["macro_f05"])
    if oracle_f05 >= 0.99:
        recommendation = "Freeze retrieval and proceed to B005 full-corpus hard-negative retraining."
    elif oracle_f05 >= 0.98:
        recommendation = "Inspect the recorded remaining misses for clusters before adding another retriever."
    else:
        recommendation = "Oracle remains below 0.98; inspect missed examples and design the next targeted retrieval channel."

    channel_report = unique_channel_contribution(merged.values(), truth)
    metrics = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "r0_reproduction": reproduction,
        "uncapped_ablation": ablations,
        "capped_50_rrf_ablation": capped_ablations,
        "channel_unique_contribution": channel_report,
        "best_configuration": best,
        "recommendation": recommendation,
        "missing_gt_pairs": len(missing_pairs),
    }
    B003.atomic_json(args.output_dir / "channel_ablation.json", {
        "uncapped": ablations, "capped_50_rrf": capped_ablations,
    })
    B003.atomic_json(args.output_dir / "candidate_oracle.json", {
        "best_configuration": best,
        "recommendation": recommendation,
        "missing_gt_pairs": len(missing_pairs),
    })
    B003.atomic_json(args.output_dir / "retrieval_metrics.json", metrics)
    telemetry.record("evaluation_complete", processed_s1=len(selected_ids), candidate_pairs=len(merged))
    return metrics


def print_summary(metrics: Mapping[str, Any]) -> None:
    print("\nB004 MULTI-VIEW RETRIEVAL", flush=True)
    print(f"{'Configuration':<24} {'Recall':>10} {'Oracle F0.5':>14} {'Pairs':>12}")
    for result in metrics["uncapped_ablation"]:
        print(
            f"{result['name']:<24} {result['gt_link_recall']:>10.6f} "
            f"{result['candidate_restricted_oracle']['macro_f05']:>14.6f} "
            f"{result['candidate_pairs']:>12,}"
        )
    print("\nUNIQUE GT RECOVERY")
    for channel, values in metrics["channel_unique_contribution"].items():
        if "gt_links_uniquely_recovered" in values:
            print(f"{channel:<20} {values['gt_links_uniquely_recovered']:>8,}")
    best = metrics["best_configuration"]
    print(f"\nBEST CONFIGURATION: {best['name']}")
    print(f"Recall: {best['gt_link_recall']:.6f}")
    print(f"Candidate-restricted oracle Macro F0.5: {best['candidate_restricted_oracle']['macro_f05']:.6f}")
    print(f"R0 reproduction: {metrics['r0_reproduction']['status']}")
    print(f"Recommendation: {metrics['recommendation']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, default=Path("experiments/val_s1_ids.txt"))
    parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/B004"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--query-chunk-size", type=int, default=250)
    parser.add_argument("--progress-every-targets", type=int, default=500_000)
    parser.add_argument("--r0-k", type=int, default=50)
    parser.add_argument("--r0-name-prune", type=int, default=300)
    parser.add_argument("--r0-addr-prune", type=int, default=150)
    parser.add_argument("--r0-tolerance", type=float, default=0.002)
    parser.add_argument("--tfidf-k", type=int, default=30)
    parser.add_argument("--tfidf-min-score", type=float, default=1e-6)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--name-char-analyzer", choices=("char", "char_wb"), default="char_wb")
    parser.add_argument("--name-char-ngram-min", type=int, default=3)
    parser.add_argument("--name-char-ngram-max", type=int, default=5)
    parser.add_argument("--name-char-max-features", type=int, default=1_000_000)
    parser.add_argument("--address-word-max-features", type=int, default=750_000)
    parser.add_argument("--name-word-max-features", type=int, default=500_000)
    parser.add_argument("--sparse-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--union-cap", type=int, default=50)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--r0-guarantee", type=int, default=10)
    parser.add_argument("--max-missed-examples", type=int, default=200)
    args = parser.parse_args(argv)
    if args.r0_k != 50:
        parser.error("B004 freezes R0 at K50")
    for name in ("val_limit", "query_chunk_size", "progress_every_targets", "tfidf_k", "union_cap"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    os.chdir(root)
    for path in (args.source1, args.source2, args.source3, args.ground_truth, args.val_ids):
        if not path.exists():
            raise FileNotFoundError(path)
    selected_ids, total_ids = B003.load_selected_ids(args.val_ids, args.val_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    run_manifest = manifest(args, selected_ids, total_ids)
    run_manifest["config_fingerprint"] = config_fingerprint(run_manifest)
    if args.resume and manifest_path.exists():
        previous = B003.read_json(manifest_path)
        if previous.get("config_fingerprint") != run_manifest["config_fingerprint"]:
            raise RuntimeError("Refusing resume because B004 configuration changed")
    elif manifest_path.exists() and any((args.output_dir / "channels").glob("*/*/*/*.DONE")):
        raise RuntimeError("B004 checkpoints exist; use --resume or a new output directory")
    run_manifest["status"] = "RUNNING"
    B003.atomic_json(manifest_path, run_manifest)

    telemetry = B003.Telemetry(args.output_dir / "telemetry.jsonl")
    telemetry.record("start")
    norm, get_blocking_keys, evaluate_predictions = import_retrieval_modules()
    validation_records, country_to_ids = load_selected_records(args.source1, selected_ids)
    run_manifest["validation_country_counts"] = {
        country: len(ids) for country, ids in country_to_ids.items()
    }
    B003.atomic_json(manifest_path, run_manifest)

    for country in sorted(country_to_ids, key=lambda value: len(country_to_ids[value])):
        query_ids = country_to_ids[country]
        run_r0_country(
            args, country, query_ids, validation_records, norm,
            get_blocking_keys, telemetry,
        )
        run_tfidf_source_country(
            args, country, "S2", args.source2, query_ids,
            validation_records, norm, telemetry,
        )
        run_tfidf_source_country(
            args, country, "S3", args.source3, query_ids,
            validation_records, norm, telemetry,
        )

    metrics = evaluate(
        args, selected_ids, validation_records, evaluate_predictions, telemetry
    )
    run_manifest["status"] = "COMPLETE" if metrics["r0_reproduction"]["status"] == "PASS" else "INVALID_R0_REPRODUCTION"
    run_manifest["completed_at"] = B003.utc_now()
    run_manifest["runtime_seconds"] = time.monotonic() - telemetry.started
    run_manifest["peak_rss_bytes"] = telemetry.peak_rss or None
    run_manifest["outputs"] = {
        "retrieval_metrics": str((args.output_dir / "retrieval_metrics.json").resolve()),
        "channel_ablation": str((args.output_dir / "channel_ablation.json").resolve()),
        "candidate_oracle": str((args.output_dir / "candidate_oracle.json").resolve()),
        "telemetry": str((args.output_dir / "telemetry.jsonl").resolve()),
        "union_provenance": str((args.output_dir / "union" / "uncapped_provenance.parquet").resolve()),
        "missed_ground_truth": str((args.output_dir / "missed_ground_truth.json").resolve()),
    }
    B003.atomic_json(manifest_path, run_manifest)
    if run_manifest["status"] == "COMPLETE":
        B003.atomic_text(args.output_dir / "B004.DONE", B003.utc_now() + "\n")
    print_summary(metrics)
    if run_manifest["status"] != "COMPLETE":
        print("B004 INVALID: R0 did not reproduce B003 within tolerance.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
