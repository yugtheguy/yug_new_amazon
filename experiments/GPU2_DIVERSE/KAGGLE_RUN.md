# Kaggle — GPU2 diversity challenge

## Install and test

```bash
python -m pip install -q -r experiments/GPU2_DIVERSE/requirements.txt
python -m unittest discover -s experiments/GPU2_DIVERSE -p 'test_*.py' -v
```

## Recommended D1-only run

```bash
REPO=/kaggle/working/amazon_ml1
B005_CACHE="$REPO/artifacts/experiments/B005"
B005_METADATA="$REPO/models/experiments/B005/metadata.json"
GROUND_TRUTH=/kaggle/input/REPLACE/train_ground_truth.tsv
OUT="$REPO/artifacts/experiments/GPU2_DIVERSE"

cd "$REPO"
PYTHONHASHSEED=0 python -u experiments/GPU2_DIVERSE/run_gpu2_diverse.py \
  --b005-artifacts "$B005_CACHE" \
  --b005-metadata "$B005_METADATA" \
  --ground-truth "$GROUND_TRUTH" \
  --val-ids experiments/val_s1_ids.txt \
  --output-dir "$OUT" \
  --deadline-minutes 45
```

Add `--run-xgboost` only if D1 is expected to leave enough of the 45-minute
window. Add `--try-second-catboost` only when a first CatBoost fit is known to be
very fast.

For blending, additionally supply both files and their probability column:

```bash
  --b007-calibration-predictions /path/b007_calibration_predictions.parquet \
  --b007-evaluation-predictions /path/b007_evaluation_predictions.parquet \
  --b007-probability-column b007_probability
```

Do not supply only one B007 file. The runner rejects incomplete or misaligned
prediction inputs.
