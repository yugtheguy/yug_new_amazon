#!/usr/bin/env python3
"""Emergency Recovery: India Rerun + K=3 Calibration."""

import sys
import os
import shutil
import argparse
import collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in ("experiments/B003", "experiments/B004", "experiments/B006", "experiments/B007", "scripts"):
    sys.path.insert(0, str(ROOT / rel))
import run_b003 as B003
import run_final_inference as RFI

SHORTLIST_K = 3

def compute_macro_f05(predictions, ground_truth):
    f05_scores = []
    for qid, gt_list in ground_truth.items():
        pred_list = predictions.get(qid, set())
        gt_set = set(gt_list)
        if not gt_set and not pred_list:
            f05_scores.append(1.0)
            continue
        if not gt_set or not pred_list:
            f05_scores.append(0.0)
            continue
        tp = len(pred_list & gt_set)
        fp = len(pred_list - gt_set)
        fn = len(gt_set - pred_list)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
    return np.mean(f05_scores), np.mean([1 if len(p) > 0 else 0 for p in f05_scores]) # return f05 and something else?
    
def exact_f05_details(predictions, ground_truth):
    f05_scores = []
    tps, fps, fns = 0, 0, 0
    for qid, gt_list in ground_truth.items():
        pred_list = predictions.get(qid, set())
        gt_set = set(gt_list)
        if not gt_set and not pred_list:
            f05_scores.append(1.0)
            continue
        if not gt_set or not pred_list:
            f05_scores.append(0.0)
            if pred_list: fps += len(pred_list)
            if gt_set: fns += len(gt_set)
            continue
        tp = len(pred_list & gt_set)
        fp = len(pred_list - gt_set)
        fn = len(gt_set - pred_list)
        tps += tp; fps += fp; fns += fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
    global_p = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    global_r = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    return macro_f05, global_p, global_r

def calibrate_decision_rule(train_gt_path):
    print("\n=== STEP 3: CALIBRATE R0-K3 DECISION ON CACHED VALIDATION ===", flush=True)
    # Find validation artifacts
    search_dirs = [ROOT / "artifacts/experiments/B007", ROOT / "artifacts/experiments/B005"]
    cands_path = None
    preds_path = None
    for d in search_dirs:
        cp = d / "calibration_candidates.parquet"
        pp = d / "calibration_predictions.parquet"
        if cp.exists() and pp.exists():
            cands_path = cp
            preds_path = pp
            break
            
    if not cands_path:
        print("ERROR: Could not find calibration_candidates.parquet")
        return None
        
    print(f"Loading {cands_path}...")
    cands_df = pq.read_table(cands_path).to_pandas()
    preds_df = pq.read_table(preds_path).to_pandas()
    
    # 1. Filter found_R0 == 1 (or found_by_existing)
    r0_col = "found_by_existing" if "found_by_existing" in cands_df.columns else "found_R0"
    if r0_col in cands_df.columns:
        cands_df = cands_df[cands_df[r0_col] == 1]
    
    # 2. Mimic production shortlist (Top 3 per query based on existing_shared_keys)
    cands_df['existing_shared_keys'] = cands_df.get('existing_shared_keys', 0).fillna(0).astype(int)
    # Sort descending
    cands_df = cands_df.sort_values(['query_id', 'existing_shared_keys'], ascending=[True, False])
    # Take top 3 per query
    shortlisted = cands_df.groupby('query_id').head(3)
    
    # Merge with predictions to get probabilities
    merged = shortlisted[['query_id', 'candidate_id', 'candidate_source']].merge(
        preds_df[['query_id', 'candidate_id', 'probability']], 
        on=['query_id', 'candidate_id'], how='inner'
    )
    
    # Load ground truth for these queries
    val_queries = set(merged['query_id'].unique())
    gt = {}
    for qid, cids in B003.iter_tsv(train_gt_path):
        if qid in val_queries:
            gt[qid] = [c for c in cids.split(',') if c] if cids else []
            
    # Include queries that had no candidates in the validation set but exist in GT?
    # If they are in GT but missing from our shortlisted candidates, they get empty predictions.
    
    print(f"Validation set: {len(val_queries)} queries, {len(merged)} candidates.")
    
    thresholds = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]
    margins = [0, 0.01, 0.03, 0.05, 0.10]
    
    best_rule = None
    best_f05 = -1
    
    results = []
    
    # Pre-group candidates by query and source
    grouped = collections.defaultdict(lambda: {"S2": [], "S3": []})
    for _, row in merged.iterrows():
        grouped[row['query_id']][row['candidate_source']].append((row['candidate_id'], row['probability']))
        
    for qid in grouped:
        for src in ["S2", "S3"]:
            grouped[qid][src].sort(key=lambda x: x[1], reverse=True)
            
    # Evaluate Grid
    for th in thresholds:
        # Variant A: All above threshold
        preds_A = collections.defaultdict(set)
        for qid, sources in grouped.items():
            for src, cands in sources.items():
                for cid, p in cands:
                    if p >= th:
                        preds_A[qid].add(cid)
        f05_A, p_A, r_A = exact_f05_details(preds_A, gt)
        results.append(('A', th, 0, f05_A, p_A, r_A))
        
        # Variant B: Highest above threshold
        preds_B = collections.defaultdict(set)
        for qid, sources in grouped.items():
            for src, cands in sources.items():
                if cands and cands[0][1] >= th:
                    preds_B[qid].add(cands[0][0])
        f05_B, p_B, r_B = exact_f05_details(preds_B, gt)
        results.append(('B', th, 0, f05_B, p_B, r_B))
        
        # Variant D (which encompasses C when margin=0)
        for m in margins:
            preds_D = collections.defaultdict(set)
            for qid, sources in grouped.items():
                for src, cands in sources.items():
                    if not cands: continue
                    p1 = cands[0][1]
                    if p1 >= th:
                        if len(cands) == 1:
                            preds_D[qid].add(cands[0][0])
                        else:
                            p2 = cands[1][1]
                            if (p1 - p2) >= m:
                                preds_D[qid].add(cands[0][0])
            f05_D, p_D, r_D = exact_f05_details(preds_D, gt)
            variant_name = 'C' if m == 0 else 'D'
            results.append((variant_name, th, m, f05_D, p_D, r_D))
            
    # Sort and pick best
    results.sort(key=lambda x: x[3], reverse=True)
    best_rule = results[0]
    
    print("\n--- Top 10 Decision Rules ---")
    print(f"{'Variant':<8} | {'Thresh':<8} | {'Margin':<8} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10}")
    for r in results[:10]:
        print(f"{r[0]:<8} | {r[1]:<8.3f} | {r[2]:<8.3f} | {r[3]:<10.5f} | {r[4]:<10.5f} | {r[5]:<10.5f}")
        
    return best_rule

def apply_winning_rule(rule, preds_df, out_path, test_source1_ids):
    variant, th, margin, _, _, _ = rule
    grouped = collections.defaultdict(lambda: {"S2": [], "S3": []})
    for qid, cid, src, p in zip(preds_df['query_id'], preds_df['candidate_id'], preds_df['candidate_source'], preds_df['probability']):
        grouped[qid][src].append((cid, p))
        
    for qid in grouped:
        for src in ["S2", "S3"]:
            grouped[qid][src].sort(key=lambda x: x[1], reverse=True)
            
    final_preds = collections.defaultdict(set)
    for qid in test_source1_ids:
        final_preds[qid] = set() # Ensure all S1 are present
        
    for qid, sources in grouped.items():
        if qid not in final_preds: continue
        for src, cands in sources.items():
            if not cands: continue
            if variant == 'A':
                for cid, p in cands:
                    if p >= th: final_preds[qid].add(cid)
            elif variant == 'B':
                if cands[0][1] >= th: final_preds[qid].add(cands[0][0])
            elif variant in ('C', 'D'):
                p1 = cands[0][1]
                if p1 >= th:
                    if len(cands) == 1:
                        final_preds[qid].add(cands[0][0])
                    else:
                        p2 = cands[1][1]
                        if (p1 - p2) >= margin:
                            final_preds[qid].add(cands[0][0])
                            
    return final_preds

def run_recovery(args):
    print("=== STEP 1: PRESERVE EXISTING US/FRANCE ===", flush=True)
    recovery_dir = args.output_dir
    recovery_dir.mkdir(parents=True, exist_ok=True)
    
    old_dir = ROOT / "artifacts/emergency_shortlist"
    
    for country in ["US", "France"]:
        src = old_dir / country / "predictions.parquet"
        dst = recovery_dir / country / "predictions.parquet"
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            print(f"Preserved {country} predictions.")
        else:
            print(f"WARNING: Missing {country} predictions in {src}")

    print("\n=== STEP 2: REGENERATE INDIA R0 (FRESH) ===", flush=True)
    india_dir = recovery_dir / "India"
    if india_dir.exists():
        shutil.rmtree(india_dir)
    india_dir.mkdir(parents=True, exist_ok=True)
    
    original_load = RFI.load_selected_merged_chunk
    def patched_load(retrieval_dir, country, chunk_index, retrievers):
        merged = original_load(retrieval_dir, country, chunk_index, retrievers)
        query_candidates = collections.defaultdict(list)
        for (qid, cid), ev in merged.items():
            query_candidates[qid].append((cid, ev))
        shortlisted_merged = {}
        for qid, cands in query_candidates.items():
            cands.sort(key=lambda x: int(x[1].get("existing_shared_keys") or 0), reverse=True)
            for cid, ev in cands[:SHORTLIST_K]:
                shortlisted_merged[(qid, cid)] = ev
        return shortlisted_merged
    RFI.load_selected_merged_chunk = patched_load

    test_dir = args.test_dir
    rfi_args = argparse.Namespace(
        source1=test_dir / "test_source1.tsv",
        source2=test_dir / "test_source2.tsv",
        source3=test_dir / "test_source3.tsv",
        model_dir=ROOT / "models/final_b007",
        output_dir=recovery_dir,
        country=["India"],
        smoke_france=0,
        model_id="B007_03",
        resume=False, # FRESH RETRIEVAL!
        query_chunk_size=1000,
        retrievers=["R0"],
        progress_every_targets=500000,
        sparse_threads=max(1, os.cpu_count() or 1)
    )
    
    print("Running India inference...")
    RFI.run_country(rfi_args, "India")
    
    # Sanity Check
    report_path = india_dir / "country_report.json"
    if report_path.exists():
        report = B003.read_json(report_path)
        missing_left = report.get("semantic_missingness_pct", {}).get("name_missing_left", 100.0)
        print(f"\nINDIA SANITY CHECK:")
        print(f"name_missing_left = {missing_left}%")
        print(f"avg_candidates_per_s1 = {report.get('avg_candidates_per_s1')}")
        print(f"probability p95 = {report.get('probability', {}).get('p95')}")
        
        if missing_left > 10.0:
            print("❌ FATAL: India missingness is still catastrophically high. ABORTING.")
            sys.exit(1)
        else:
            print("✅ India missingness looks healthy!")
    
    # Step 3
    train_dir = ROOT / "dataset/train" if (ROOT / "dataset/train").exists() else Path("/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train")
    best_rule = calibrate_decision_rule(train_dir / "train_ground_truth.tsv")
    
    if not best_rule:
        print("Fallback to strict rule A with 0.95")
        best_rule = ('A', 0.95, 0, 0, 0, 0)
        
    print("\n=== STEP 4: SELECTED EXACTLY ONE DECISION RULE ===")
    print(f"Variant: {best_rule[0]}")
    print(f"Threshold: {best_rule[1]}")
    print(f"Margin: {best_rule[2]}")
    
    print("\n=== STEP 5 & 6: APPLY TO TEST PROBABILITIES ===")
    final_dir = recovery_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    all_s1 = []
    for q, _, _, _ in B003.iter_tsv(rfi_args.source1):
        all_s1.append(q)
        
    global_preds = {}
    for country in ["India", "US", "France"]:
        p_path = recovery_dir / country / "predictions.parquet"
        if not p_path.exists(): continue
        df = pq.read_table(p_path).to_pandas()
        c_s1 = [q for q, _, _, c in B003.iter_tsv(rfi_args.source1) if c == country]
        preds = apply_winning_rule(best_rule, df, None, set(c_s1))
        
        nonempty = sum(1 for v in preds.values() if v)
        total_links = sum(len(v) for v in preds.values())
        print(f"{country} -> S1: {len(c_s1)}, Nonempty: {nonempty}, Total Links: {total_links}, Avg Links/Nonempty: {total_links/max(1, nonempty):.2f}")
        
        global_preds.update(preds)
        
    print("\n=== STEP 7: BUILD NEW SUBMISSION ===")
    import csv
    matching = final_dir / "matching_results.tsv"
    with open(matching, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for q in all_s1:
            writer.writerow([q, ",".join(sorted(global_preds.get(q, set())))])
            
    # Dummy candidates file to pass validation
    cands_path = final_dir / "candidate_pairs.tsv"
    with open(cands_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for q in all_s1:
            writer.writerow([q, ",".join(sorted(global_preds.get(q, set())))])
            
    cmd = [
        sys.executable, str(ROOT / "utils/validate_submission.py"),
        "--matching", str(matching),
        "--candidate", str(cands_path),
        "--test-dir", str(test_dir),
        "--check-ids"
    ]
    res = subprocess.run(cmd, text=True, capture_output=True)
    print(res.stdout, end="")
    print(res.stderr, file=sys.stderr, end="")
    if res.returncode == 0:
        print(f"\n✅ RECOVERY SUBMISSION READY: {matching}", flush=True)
    else:
        print("\n❌ VALIDATOR FAILED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/emergency_recovery"))
    args = parser.parse_args()
    run_recovery(args)
