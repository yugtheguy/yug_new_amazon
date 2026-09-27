#!/usr/bin/env python3
"""Emergency submission script: Fast R0 Shortlist Cascade."""

import argparse
import collections
import shutil
import sys
import os
import time
import json
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/B003", "experiments/B004", "experiments/B006", "experiments/B007"):
    sys.path.insert(0, str(ROOT / rel))
import run_b003 as B003
import run_b004 as B004
import run_b006 as B006
import run_b007 as B007R
import b007_features as B007F
import apply_decision_rule as ADR
import run_final_inference as RFI

SHORTLIST_K = 3

def run_stage_a_shortlist(args):
    print(f"=== STAGE A: R0 ONLY (SHORTLIST K={SHORTLIST_K}) ===", flush=True)
    out_dir = args.output_dir
    test_dir = args.test_dir
    countries = ["India", "US", "France"]
    
    # 1. Monkey-patch build_rows caller to inject shortlist
    original_run_country = RFI.run_country
    
    # We will hook into the main loop inside run_country.
    # Since run_country is a huge function, it's safer to monkey-patch `load_selected_merged_chunk`
    original_load = RFI.load_selected_merged_chunk
    def patched_load(retrieval_dir, country, chunk_index, retrievers):
        merged = original_load(retrieval_dir, country, chunk_index, retrievers)
        
        query_candidates = collections.defaultdict(list)
        for (qid, cid), ev in merged.items():
            query_candidates[qid].append((cid, ev))
            
        shortlisted_merged = {}
        for qid, cands in query_candidates.items():
            # Sort by existing_shared_keys descending
            cands.sort(key=lambda x: int(x[1].get("existing_shared_keys") or 0), reverse=True)
            for cid, ev in cands[:SHORTLIST_K]:
                shortlisted_merged[(qid, cid)] = ev
                
        # Only print for the first chunk to avoid spam
        if chunk_index == 0:
            print(f"\n[SHORTLIST REPORT] K={SHORTLIST_K}")
            print(f"R0 candidates before shortlist = {len(merged)}")
            print(f"R0 candidates after shortlist = {len(shortlisted_merged)}")
            reduction = len(merged) / max(1, len(shortlisted_merged))
            print(f"Reduction factor = {reduction:.2f}x\n", flush=True)
            
        return shortlisted_merged
        
    RFI.load_selected_merged_chunk = patched_load

    predictions = []
    for country in countries:
        country_out = out_dir / country
        country_out.mkdir(parents=True, exist_ok=True)
        
        if country == "India":
            final_test_r0 = ROOT / f"artifacts/final_test/{country}/_retrieval"
            emerg_r0 = country_out / "_retrieval"
            if final_test_r0.exists() and not emerg_r0.exists():
                print(f"Copying R0 checkpoint for {country} from final_test to emergency_submission...", flush=True)
                shutil.copytree(str(final_test_r0), str(emerg_r0))
        
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
            query_chunk_size=1000, # Increased chunk size as requested
            retrievers=["R0"],
            progress_every_targets=500000,
            sparse_threads=max(1, os.cpu_count() or 1)
        )
        
        print(f"\n--- Running STAGE A inference for {country} ---", flush=True)
        RFI.run_country(rfi_args, country)
        predictions.append(country_out / "predictions.parquet")
        
    print("\n=== MERGING STAGE A (SHORTLIST) ===", flush=True)
    submission_dir = out_dir / "stage_a_shortlist"
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
        print(f"\n✅ SUBMISSION_FAST_R0_SHORTLIST_B007_READY: {submission_dir / 'matching_results.tsv'}", flush=True)
    else:
        print("\n❌ ERROR generating Stage A submission!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/emergency_shortlist"))
    args = parser.parse_args()
    run_stage_a_shortlist(args)
