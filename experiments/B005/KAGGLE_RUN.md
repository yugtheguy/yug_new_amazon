# Kaggle execution — B005

## 1. Clone the B005 branch

```bash
cd /kaggle/working
git clone --branch exp/B005-hard-negative-retrain --single-branch \
  https://github.com/yugtheguy/yug_new_amazon.git amazon_ml
cd /kaggle/working/amazon_ml
git log -1 --oneline
```

## 2. Install dependencies

```bash
python -m pip install -r experiments/B005/requirements.txt
```

## 3. Run the complete pipeline

The command prints live progress for target scans, TF-IDF retrieval, checkpoints,
feature extraction, GPU/CPU model fitting, calibration, and evaluation. A heartbeat
also prints the current stage and elapsed time every 30 seconds, including while a
single long operation is still running. Change it with `--heartbeat-seconds`.

```bash
cd /kaggle/working/amazon_ml
DATA_ROOT=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train
OUT=/kaggle/working/amazon_ml/artifacts/experiments/B005
MODEL_OUT=/kaggle/working/amazon_ml/models/experiments/B005

PYTHONHASHSEED=0 python -u experiments/B005/run_b005.py \
  --source1 "$DATA_ROOT/train_source1.tsv" \
  --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --val-limit 5000 \
  --train-limit 50000 \
  --max-negatives-per-s1 50 \
  --output-dir "$OUT" \
  --model-dir "$MODEL_OUT" \
  --device auto
```

If the completed B004 artifact directory is still available and contains its
manifest plus `union/uncapped_provenance.parquet`, replace the final two command
lines with:

```bash
  --model-dir "$MODEL_OUT" \
  --b004-eval-dir /kaggle/working/amazon_ml/artifacts/experiments/B004 \
  --device auto
```

B005 validates the B004 selected-ID hash and frozen retrieval configuration before
using the cache. With this option it retrieves only the new train/calibration S1s.

`--device auto` attempts Kaggle's LightGBM GPU backend once and records the error
before automatically falling back to CPU. It never rebuilds LightGBM.

## 4. Resume safely

Repeat the identical command with `--resume`. The runner rejects a changed
configuration fingerprint. Completed country/channel retrieval and feature chunks
are reused.

```bash
PYTHONHASHSEED=0 python -u experiments/B005/run_b005.py \
  --source1 "$DATA_ROOT/train_source1.tsv" --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt --val-limit 5000 --train-limit 50000 \
  --max-negatives-per-s1 50 --output-dir "$OUT" --model-dir "$MODEL_OUT" \
  --device auto --resume
```

## 5. Inspect progress and outputs

```bash
tail -n 30 "$OUT/telemetry.jsonl"
find "$OUT/retrieval/channels" -name '*.DONE' | wc -l
find "$OUT/features" -name '*.DONE' | wc -l
cat "$OUT/manifest.json"
cat "$OUT/calibration_metrics.json"
cat "$OUT/evaluation_metrics.json"
cat "$OUT/errors.json"
cat "$MODEL_OUT/metadata.json"
```

Expected compact outputs:

```text
artifacts/experiments/B005/manifest.json
artifacts/experiments/B005/training_candidate_metrics.json
artifacts/experiments/B005/hard_negative_metrics.json
artifacts/experiments/B005/calibration_metrics.json
artifacts/experiments/B005/evaluation_metrics.json
artifacts/experiments/B005/errors.json
artifacts/experiments/B005/telemetry.jsonl
models/experiments/B005/lightgbm.joblib
models/experiments/B005/metadata.json
```

To try 100,000 pool rows later, use a new output directory and
`--train-limit 100000`; do not reuse the 50,000-row checkpoints.
