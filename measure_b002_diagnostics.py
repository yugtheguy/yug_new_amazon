import os, sys, time, collections, gc
import numpy as np
def update_peak_mem():
    pass


src_dir = os.path.abspath('code/business_entity_resolution/src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from blocking import get_blocking_keys

def stream_file(path, country_filter=None, max_rows=None):
    with open(path, 'rb') as f:
        f.readline()
        count = 0
        for line in f:
            try:
                line_str = line.decode('utf-8').rstrip('\r\n')
            except:
                continue
            p = line_str.split('\t')
            if not p or len(p) < 1: continue
            sid = p[0]
            name = p[1] if len(p) > 1 else ''
            addr = p[2] if len(p) > 2 else ''
            c = p[3] if len(p) > 3 else ''
            if country_filter is None or c == country_filter:
                yield sid, name, addr, c
                count += 1
                if max_rows and count >= max_rows:
                    break

val_ids = set()
with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sid = line.strip()
        if sid: val_ids.add(sid)
        if len(val_ids) >= 20000: break

gt_file = 'data/student_resource/dataset/train/train_ground_truth.tsv'
val_gt = {}
train_s1_ids = []
train_gt = {}
train_true_targets = set()
with open(gt_file, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        sid = p[0]
        mids = p[1].split(',') if len(p)>1 and p[1] else []
        if sid in val_ids:
            val_gt[sid] = set(mids)
        elif len(train_s1_ids) < 50000:
            train_s1_ids.append(sid)
            train_gt[sid] = set(mids)
            train_true_targets.update(mids)

print("PART 5: TARGET DENSITY")
s2_counts = collections.Counter()
s3_counts = collections.Counter()
for sid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source2.tsv'):
    s2_counts[c] += 1
for sid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source3.tsv'):
    s3_counts[c] += 1

val_s1_records = {}
s1_counts = collections.Counter()
for sid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source1.tsv'):
    if sid in val_ids:
        val_s1_records[sid] = (name, addr, c)
        s1_counts[c] += 1

total_s2 = sum(s2_counts.values())
total_s3 = sum(s3_counts.values())
print(f"S2 Total: {total_s2}")
for c, cnt in s2_counts.items():
    print(f"  {c}: {cnt} ({cnt/total_s2*100:.2f}%)")
print(f"S3 Total: {total_s3}")
for c, cnt in s3_counts.items():
    print(f"  {c}: {cnt} ({cnt/total_s3*100:.2f}%)")
print("Val S1 Counts:")
for c, cnt in s1_counts.items():
    print(f"  {c}: {cnt}")

update_peak_mem()

print("\nPART 1-3: BLOCKING MEASUREMENTS")

def eval_blocking(source_prefix, source_path):
    stats = {
        'current': {'cands': [], 'recall_all': 0, 'recall_20': 0, 'recall_50': 0},
        'unlimited': {'cands': [], 'recall_all': 0, 'recall_20': 0, 'recall_50': 0}
    }
    
    total_matches = sum(len([t for t in val_gt.get(sid, []) if t.startswith(source_prefix)]) for sid in val_s1_records)
    print(f"{source_prefix} Total Ground Truth Matches: {total_matches}")
    
    for country in s1_counts.keys():
        print(f"  Processing {country} for {source_prefix}...")
        
        tid_to_int = {}
        int_to_tid = []
        
        index_unlimited = collections.defaultdict(list)
        for tid, name, addr, c in stream_file(source_path, country):
            tid_idx = len(int_to_tid)
            tid_to_int[tid] = tid_idx
            int_to_tid.append(tid)
            keys = get_blocking_keys(name, addr, c)
            for k in keys:
                index_unlimited[k].append(tid_idx)
                
        # Convert to numpy arrays for fast concatenation
        for k in index_unlimited:
            index_unlimited[k] = np.array(index_unlimited[k], dtype=np.int32)
        
        index_current = {}
        for k, v in index_unlimited.items():
            limit = 300 if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact')) else 150
            if len(v) <= limit:
                index_current[k] = v
                
        num_targets = len(int_to_tid)
        
        # Evaluate for val S1 in this country
        for sid, (name, addr, c) in val_s1_records.items():
            if c != country: continue
            
            s_keys = get_blocking_keys(name, addr, c)
            true_mids = {t for t in val_gt.get(sid, []) if t.startswith(source_prefix)}
            
            def get_candidates(idx_dict):
                arrs = [idx_dict[k] for k in s_keys if k in idx_dict]
                if not arrs: return []
                all_hits = np.concatenate(arrs)
                counts = np.bincount(all_hits, minlength=num_targets)
                nz = np.nonzero(counts)[0]
                if len(nz) == 0: return []
                sorted_idx = nz[np.argsort(-counts[nz])]
                return [int_to_tid[i] for i in sorted_idx]
            
            # Current Pruning
            cand_ordered = get_candidates(index_current)
            stats['current']['cands'].append(len(cand_ordered))
            stats['current']['recall_all'] += len(set(cand_ordered) & true_mids)
            stats['current']['recall_50'] += len(set(cand_ordered[:50]) & true_mids)
            stats['current']['recall_20'] += len(set(cand_ordered[:20]) & true_mids)
            
            # No Pruning
            cand_ordered_unl = get_candidates(index_unlimited)
            stats['unlimited']['cands'].append(len(cand_ordered_unl))
            stats['unlimited']['recall_all'] += len(set(cand_ordered_unl) & true_mids)
            stats['unlimited']['recall_50'] += len(set(cand_ordered_unl[:50]) & true_mids)
            stats['unlimited']['recall_20'] += len(set(cand_ordered_unl[:20]) & true_mids)
            
        index_unlimited.clear()
        index_current.clear()
        tid_to_int.clear()
        int_to_tid.clear()
        gc.collect()
        update_peak_mem()
        
    for mode in ['current', 'unlimited']:
        c_arr = np.array(stats[mode]['cands'])
        print(f"--- {source_prefix} {mode} ---")
        if total_matches > 0:
            print(f"Recall All: {stats[mode]['recall_all'] / total_matches:.4f}")
            print(f"Recall @50: {stats[mode]['recall_50'] / total_matches:.4f}")
            print(f"Recall @20: {stats[mode]['recall_20'] / total_matches:.4f}")
        print(f"Total Cands: {c_arr.sum()}")
        print(f"Avg: {c_arr.mean():.1f}")
        print(f"Median: {np.median(c_arr)}")
        print(f"P95: {np.percentile(c_arr, 95)}")
        print(f"P99: {np.percentile(c_arr, 99)}")
        print(f"Max: {c_arr.max()}")
        print(f"Zero-cand S1: {np.sum(c_arr == 0)}")
        
    return stats, total_matches

s2_stats, s2_total_matches = eval_blocking('S2', 'data/student_resource/dataset/train/train_source2.tsv')
s3_stats, s3_total_matches = eval_blocking('S3', 'data/student_resource/dataset/train/train_source3.tsv')

print("\nPART 4: FORCED POSITIVE ANALYSIS (TRAIN B001)")
# Reproduce train.py logic for target subsets
train_target_set = set()
distractors_s2 = 0
for tid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source2.tsv'):
    if tid in train_true_targets or distractors_s2 < 50000:
        train_target_set.add(tid)
        if tid not in train_true_targets: distractors_s2 += 1
distractors_s3 = 0
for tid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source3.tsv'):
    if tid in train_true_targets or distractors_s3 < 50000:
        train_target_set.add(tid)
        if tid not in train_true_targets: distractors_s3 += 1

# Re-read train_s1 records
train_s1_records = {}
for sid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source1.tsv'):
    if sid in train_s1_ids:
        train_s1_records[sid] = (name, addr, c)

# Build index for the subset
index = collections.defaultdict(list)
for tid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source2.tsv'):
    if tid in train_target_set:
        for k in get_blocking_keys(name, addr, c): index[k].append(tid)
for tid, name, addr, c in stream_file('data/student_resource/dataset/train/train_source3.tsv'):
    if tid in train_target_set:
        for k in get_blocking_keys(name, addr, c): index[k].append(tid)

for k in list(index.keys()):
    limit = 300 if k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact') else 150
    if len(index[k]) > limit:
        del index[k]

total_positives = sum(len(train_gt[sid] & train_target_set) for sid in train_s1_ids)
naturally_retrieved = 0
forcibly_injected = 0
retrieved_but_not_top20 = 0

for sid in train_s1_ids:
    if sid not in train_s1_records: continue
    name, addr, c = train_s1_records[sid]
    s_keys = get_blocking_keys(name, addr, c)
    true_mids = train_gt[sid] & train_target_set
    
    counts = collections.Counter()
    for k in s_keys:
        if k in index:
            counts.update(index[k])
            
    cands_top20 = set(t for t, c_ in counts.most_common(20))
    cands_all = set(counts.keys())
    
    for mid in true_mids:
        if mid in cands_top20:
            naturally_retrieved += 1
        elif mid in cands_all:
            retrieved_but_not_top20 += 1
            forcibly_injected += 1
        else:
            forcibly_injected += 1

print(f"Total positives: {total_positives}")
print(f"Naturally retrieved (in top 20): {naturally_retrieved}")
print(f"Forcibly injected: {forcibly_injected}")
print(f"Of injected, retrieved but not top 20: {retrieved_but_not_top20}")
print(f"Injection percentage: {forcibly_injected/total_positives*100:.2f}%")

print("\nPART 6: MEMORY / RUNTIME")
update_peak_mem()
print("Peak memory tracking disabled")
