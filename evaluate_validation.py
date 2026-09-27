import os
import sys

def parse_gt(gt_path, val_ids):
    gt = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\r\n').split('\t')
            if len(parts) >= 1:
                sid = parts[0]
                if sid in val_ids:
                    mids = parts[1] if len(parts) > 1 else ''
                    gt[sid] = set(mids.split(',')) if mids else set()
    return gt

def parse_pred(pred_path):
    pred = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip('\r\n').split('\t')
            if len(parts) >= 1:
                sid = parts[0]
                mids = parts[1] if len(parts) > 1 else ''
                pred[sid] = set(mids.split(',')) if mids else set()
    return pred

def compute_entity_f05(true_set, pred_set):
    is_true_empty = len(true_set) == 0
    is_pred_empty = len(pred_set) == 0
    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0
    if is_pred_empty:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def eval_source(gt, pred, cands, source_prefix):
    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true = 0
    total_pred = 0
    total_cands = 0
    cand_tp = 0
    
    singleton_count = 0
    singleton_correct = 0
    
    for sid in gt.keys():
        true_set = {t for t in gt.get(sid, set()) if t.startswith(source_prefix)}
        pred_set = {t for t in pred.get(sid, set()) if t.startswith(source_prefix)}
        cand_set = {t for t in cands.get(sid, set()) if t.startswith(source_prefix)}
        
        f05_scores.append(compute_entity_f05(true_set, pred_set))
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true += len(true_set)
        total_pred += len(pred_set)
        
        total_cands += len(cand_set)
        cand_tp += len(true_set & cand_set)
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                
    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    pair_precision = total_tp / total_pred if total_pred > 0 else 1.0
    pair_recall = total_tp / total_true if total_true > 0 else 0.0
    cand_recall = cand_tp / total_true if total_true > 0 else 0.0
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 1.0
    
    return {
        'total_gt_pairs': total_true,
        'candidate_recall': cand_recall,
        'candidate_pairs': total_cands,
        'pair_precision': pair_precision,
        'pair_recall': pair_recall,
        'macro_f05': macro_f05,
        'singleton_accuracy': singleton_acc
    }

def main():
    val_ids = set()
    with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sid = line.strip()
            if sid:
                val_ids.add(sid)
            if len(val_ids) >= 20000:
                break
                
    gt = parse_gt('data/student_resource/dataset/train/train_ground_truth.tsv', val_ids)
    pred = parse_pred('temp_val_output/matching_results.tsv')
    cands = parse_pred('temp_val_output/candidate_pairs.tsv')
    
    s2_res = eval_source(gt, pred, cands, 'S2-')
    s3_res = eval_source(gt, pred, cands, 'S3-')
    
    with open('temp_val_output/top50_validation_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Top-50 Validation Run ===\n")
        f.write(f"Validation S1 entities: {len(val_ids)}\n\n")
        
        f.write("SOURCE 2:\n")
        for k, v in s2_res.items():
            f.write(f"  {k}: {v}\n")
            
        f.write("\nSOURCE 3:\n")
        for k, v in s3_res.items():
            f.write(f"  {k}: {v}\n")
            
    print("Evaluation complete. Report saved to temp_val_output/top50_validation_report.txt")

if __name__ == '__main__':
    main()
