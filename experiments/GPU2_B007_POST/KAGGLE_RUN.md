# Kaggle run

```bash
python -m pip install -q -r experiments/GPU2_B007_POST/requirements.txt
python -m unittest discover -s experiments/GPU2_B007_POST -p 'test_*.py' -v

python -u experiments/GPU2_B007_POST/run_gpu2_b007_post.py \
  --b007-01-calibration /path/b007_01_calibration_predictions.parquet \
  --b007-01-evaluation /path/b007_01_evaluation_predictions.parquet \
  --b007-03-calibration /path/b007_03_calibration_predictions.parquet \
  --b007-03-evaluation /path/b007_03_evaluation_predictions.parquet \
  --ground-truth /path/train_ground_truth.tsv \
  --val-ids experiments/val_s1_ids.txt \
  --output-dir artifacts/experiments/GPU2_B007_POST \
  --deadline-minutes 40
```

Supply `--old-threshold-s2` and `--old-threshold-s3` when GPU1's production
thresholds are known. Otherwise their calibration-derived equivalents are used.
Add `--run-e3` only after E1/E2 are known to finish comfortably within the window.
