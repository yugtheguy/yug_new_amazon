#!/usr/bin/env python3
"""Emergency submission script: two-stage cascade."""

import argparse
import collections
import shutil
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/B003"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_b003 as B003
import apply_decision_rule as ADR
import run_final_inference as RFI

def patch_for_rescue(unresolved_ids, top_k):
    # Monkey-patch country_queries to only return unresolved queries
    original_country_queries = RFI.country_queries
    def patched_country_queries(source1, country, limit=None):
        ids, records = original_country_queries(source1, country, limit)
        filtered_ids = [qid for qid in ids if qid in unresolved_ids]
        print(f"Rescue monkey-patch: filtered {len(ids)} down to {len(filtered_ids)} queries for {country}")
        return filtered_ids, records
    RFI.country_queries = patched_country_queries

    # Monkey-patch retrieval_args to change tfidf_k to top_k
    original_retrieval_args = RFI.retrieval_args
    def patched_retrieval_args(args, output):
        rargs = original_retrieval_args(args, output)
        rargs.tfidf_k = top_k
        print(f"Rescue monkey-patch: set tfidf_k={top_k}")
        return rargs
    RFI.retrieval_args = patched_retrieval_args


def run_stage_a(args):
    print("=== STAGE A: R0 ONLY ===", flush=True)
    out_dir = args.output_dir
    test_dir = args.test_dir
    countries = ["India", "US", "France"]
    
    predictions = []
    for country in countries:
        country_out = out_dir / country
        country_out.mkdir(parents=True, exist_ok=True)
        
        # Reuse India R0
        if country == "India":
            final_test_r0 = ROOT / f"artifacts/final_test/{country}/_retrieval/channels/R0_EXISTING/{country}"
            emerg_r0 = country_out / f"_retrieval/channels/R0_EXISTING/{country}"
            if final_test_r0.exists() and not emerg_r0.exists():
                print(f"Copying R0 checkpoint for {country} from final_test to emergency_submission...", flush=True)
                shutil.copytree(str(final_test_r0), str(emerg_r0))
                # Also copy .DONE files from parent directories if needed
                for source in ["S2", "S3", "S2_S3"]:
                    src_parent = final_test_r0 / source
                    dst_parent = emerg_r0 / source
                    if src_parent.exists() and dst_parent.exists():
                        for f in src_parent.glob("*.DONE"):
                            if not (dst_parent / f.name).exists():
                                shutil.copy2(str(f), str(dst_parent))
        
        # Setup RFI args
        rfi_args = argparse.Namespace(
            source1=test_dir / "test_source1.tsv",
            source2=test_dir / "test_source2.tsv",
            source3=test_dir / "test_source3.tsv",
            model_dir=ROOT / "models/final_b007",
            output_dir=out_dir,
            country=[country],
            smoke_france=0,
            model_id="B007_03",
            resume=True,
            query_chunk_size=250,
            retrievers=["R0"],
            progress_every_targets=500000,
            sparse_threads=max(1, os.cpu_count() or 1)
        )
        
        print(f"\n--- Running STAGE A inference for {country} ---", flush=True)
        RFI.run_country(rfi_args, country)
        predictions.append(country_out / "predictions.parquet")
        
        # Determine unresolved
        metadata = B003.read_json(ROOT / "models/final_b007/metadata.json")
        frozen = metadata["models"]["B007_03"]
        threshold_s2 = frozen["threshold_s2"]
        threshold_s3 = frozen["threshold_s3"]
        
        import pyarrow.parquet as pq
        scores = collections.defaultdict(list)
        table = pq.read_table(country_out / "predictions.parquet", columns=["query_id", "candidate_source", "probability"])
        for q, src, prob in zip(table["query_id"].to_pylist(), table["candidate_source"].to_pylist(), table["probability"].to_pylist()):
            scores[str(q)].append((str(src), float(prob)))
            
        # Also need all IDs to find zero-candidate queries
        all_ids = [entity_id for entity_id, _, _, _c in B003.iter_tsv(rfi_args.source1) if _c == country]
        for q in all_ids:
            if q not in scores:
                scores[q] = []
                
        unresolved = []
        ambiguous = []
        for q, cands in scores.items():
            if not cands:
                unresolved.append(q)
                continue
            max_p_s2 = max([p for s, p in cands if s == "S2"] + [-1.0])
            max_p_s3 = max([p for s, p in cands if s == "S3"] + [-1.0])
            if max_p_s2 < threshold_s2 and max_p_s3 < threshold_s3:
                # Neither reached threshold
                if (threshold_s2 - max_p_s2 < 0.1 and max_p_s2 > 0) or (threshold_s3 - max_p_s3 < 0.1 and max_p_s3 > 0):
                    ambiguous.append(q)
                else:
                    unresolved.append(q)
                    
        (country_out / "unresolved_s1.txt").write_text("\n".join(unresolved))
        (country_out / "ambiguous_s1.txt").write_text("\n".join(ambiguous))
        (country_out / "r0_DONE").touch()
        print(f"{country}: {len(all_ids)} total S1, {len(unresolved)} unresolved, {len(ambiguous)} ambiguous", flush=True)
        
    print("\n=== MERGING STAGE A ===", flush=True)
    submission_dir = out_dir / "stage_a"
    submission_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "scripts/apply_decision_rule.py"),
        "--predictions", *map(str, predictions),
        "--test-dir", str(test_dir),
        "--output-dir", str(submission_dir)
    ]
    res = subprocess.run(cmd, text=True, capture_output=True)
    print(res.stdout, end="")
    print(res.stderr, file=sys.stderr, end="")
    if res.returncode == 0:
        print(f"\nSUBMISSION_A_READY: {submission_dir / 'matching_results.tsv'}", flush=True)
    else:
        print("\nERROR generating Stage A submission!")


def run_stage_b(args):
    print("=== STAGE B: R2 RESCUE ===", flush=True)
    out_dir = args.output_dir
    rescue_dir = out_dir / "rescue"
    test_dir = args.test_dir
    countries = ["India", "US", "France"]
    
    combined_predictions = []
    
    for country in countries:
        unresolved_path = out_dir / country / "unresolved_s1.txt"
        if not unresolved_path.exists():
            print(f"Skipping {country}, missing unresolved_s1.txt")
            continue
            
        unresolved_ids = set(unresolved_path.read_text().splitlines())
        if not unresolved_ids:
            print(f"No unresolved queries for {country}, using R0 predictions as-is")
            combined_predictions.append(out_dir / country / "predictions.parquet")
            continue
            
        country_rescue_dir = rescue_dir / country
        country_rescue_dir.mkdir(parents=True, exist_ok=True)
        
        patch_for_rescue(unresolved_ids, top_k=10)
        
        rfi_args = argparse.Namespace(
            source1=test_dir / "test_source1.tsv",
            source2=test_dir / "test_source2.tsv",
            source3=test_dir / "test_source3.tsv",
            model_dir=ROOT / "models/final_b007",
            output_dir=rescue_dir,
            country=[country],
            smoke_france=0,
            model_id="B007_03",
            resume=True,
            query_chunk_size=250,
            retrievers=["R0", "R2"],
            progress_every_targets=500000,
            sparse_threads=max(1, os.cpu_count() or 1)
        )
        
        try:
            print(f"\n--- Running STAGE B inference for {country} ---", flush=True)
            RFI.run_country(rfi_args, country)
        except Exception as e:
            print(f"Rescue for {country} failed or interrupted: {e}")
            print(f"Will fallback to partial/R0 predictions for {country}")
            
        # Combine R0 predictions with R2 rescue predictions
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        r0_table = pq.read_table(out_dir / country / "predictions.parquet")
        r0_df = r0_table.to_pandas()
        
        rescue_preds_path = country_rescue_dir / "predictions.parquet"
        if rescue_preds_path.exists():
            rescue_table = pq.read_table(rescue_preds_path)
            rescue_df = rescue_table.to_pandas()
            # Drop the rescued queries from R0
            rescued_queries = set(rescue_df['query_id'].unique())
            r0_df = r0_df[~r0_df['query_id'].isin(rescued_queries)]
            combined_df = pa.concat_tables([pa.Table.from_pandas(r0_df), rescue_table])
        else:
            combined_df = r0_table
            
        combined_path = country_rescue_dir / "combined_predictions.parquet"
        pq.write_table(combined_df, combined_path)
        combined_predictions.append(combined_path)
        
    print("\n=== MERGING STAGE B ===", flush=True)
    submission_dir = out_dir / "stage_b"
    submission_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    cmd = [
        sys.executable, str(ROOT / "scripts/apply_decision_rule.py"),
        "--predictions", *map(str, combined_predictions),
        "--test-dir", str(test_dir),
        "--output-dir", str(submission_dir)
    ]
    res = subprocess.run(cmd, text=True, capture_output=True)
    print(res.stdout, end="")
    print(res.stderr, file=sys.stderr, end="")
    if res.returncode == 0:
        print(f"\nSUBMISSION_B_READY: {submission_dir / 'matching_results.tsv'}", flush=True)
    else:
        print("\nERROR generating Stage B submission!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["r0-only", "r2-rescue"], required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/emergency_submission"))
    args = parser.parse_args()
    if args.mode == "r0-only":
        run_stage_a(args)
    elif args.mode == "r2-rescue":
        run_stage_b(args)
