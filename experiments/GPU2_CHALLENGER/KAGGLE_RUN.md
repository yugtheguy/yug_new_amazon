# Kaggle execution — GPU2 final challenger

## 1. Install and test

From the repository root in the new GPU2 notebook:

```bash
python -m pip install -r experiments/GPU2_CHALLENGER/requirements.txt
python -m unittest discover -s experiments/GPU2_CHALLENGER -p 'test_*.py' -v
```

## 2. Point to the existing cache

Set these paths to the mounted/completed B005 cache and training dataset. The
cache directory must contain `manifest.json` plus Parquets under
`features/{train,calibration,evaluation}/`. The metadata must be B005's model
metadata JSON (32 feature names and its frozen LightGBM parameters).

```bash
REPO=/kaggle/working/amazon_ml
B005_CACHE=/kaggle/input/REPLACE_WITH_CACHE/artifacts/experiments/B005
B005_METADATA=/kaggle/input/REPLACE_WITH_CACHE/models/experiments/B005/metadata.json
DATA_ROOT=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train
OUT="$REPO/artifacts/experiments/GPU2_CHALLENGER"
```

Quickly verify availability; do not regenerate anything if these checks fail:

```bash
test -f "$B005_CACHE/manifest.json"
test -f "$B005_METADATA"
find "$B005_CACHE/features" -name '*.parquet' | head
```

## 3. Run C1 and C2

This is the recommended first run. It stops before C3 and therefore maximizes the
chance of returning two completed measurements within the hard deadline.

```bash
cd "$REPO"
PYTHONHASHSEED=0 python -u experiments/GPU2_CHALLENGER/run_gpu2_challenger.py \
  --b005-artifacts "$B005_CACHE" \
  --b005-metadata "$B005_METADATA" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --output-dir "$OUT" \
  --device auto \
  --deadline-minutes 90
```

Use `--run-c3` only when the full 90-minute window is available. C3 will not start
unless C1 and C2 complete and at least 30 minutes remain:

```bash
  --deadline-minutes 90 --run-c3
```

## 4. Results

```bash
cat "$OUT/manifest.json"
cat "$OUT/c1_decision_metrics.json"
cat "$OUT/c2_source_models_metrics.json"
test ! -f "$OUT/c3_tuning_metrics.json" || cat "$OUT/c3_tuning_metrics.json"
cat "$OUT/comparison.json"
tail -n 40 "$OUT/telemetry.jsonl"
```

Do not promote any model automatically. When the runner prints `GPU2 CHALLENGER
STOP`, reassign GPU2 to final inference as planned.
