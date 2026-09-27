# Kaggle execution — B004

## 1. Clone the experiment branch

```bash
cd /kaggle/working
git clone --branch exp/B004-multiview-retrieval --single-branch \
  https://github.com/yugtheguy/yug_new_amazon.git amazon_ml
cd /kaggle/working/amazon_ml
git log -1 --oneline
```

## 2. Install dependencies

```bash
python -m pip install -r experiments/B004/requirements.txt
```

## 3. Locate the training directory

The known example is:

```text
/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train
```

Kaggle may mount it under a shorter slug. Locate it safely:

```python
from pathlib import Path
matches = list(Path('/kaggle/input').rglob('train_source1.tsv'))
print('\n'.join(map(str, matches)))
DATA_ROOT = matches[0].parent
```

## 4. Run B004

Set `DATA_ROOT` in the shell command to the directory printed above:

```bash
cd /kaggle/working/amazon_ml
DATA_ROOT=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train
OUT=/kaggle/working/amazon_ml/artifacts/experiments/B004

PYTHONHASHSEED=0 python experiments/B004/run_b004.py \
  --source1 "$DATA_ROOT/train_source1.tsv" \
  --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --val-limit 5000 \
  --output-dir "$OUT"
```

The runner prints live target-scan, TF-IDF fit, sparse retrieval, checkpoint, memory,
and evaluation progress.

## 5. Resume

Repeat the identical command with `--resume`. Changing retrieval configuration while
reusing checkpoints is rejected.

```bash
PYTHONHASHSEED=0 python experiments/B004/run_b004.py \
  --source1 "$DATA_ROOT/train_source1.tsv" \
  --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt --val-limit 5000 \
  --output-dir "$OUT" --resume
```

## 6. Inspect progress and results

```bash
tail -n 20 "$OUT/telemetry.jsonl"
find "$OUT/channels" -name '*.DONE' | wc -l
cat "$OUT/manifest.json"
cat "$OUT/retrieval_metrics.json"
```

Package the compact result files:

```bash
cd "$OUT"
zip -9 /kaggle/working/B004_core_results.zip \
  manifest.json retrieval_metrics.json channel_ablation.json \
  candidate_oracle.json missed_ground_truth.json telemetry.jsonl B004.DONE
```

Download `B004_core_results.zip`. Do not start B005 until these metrics are reviewed.
