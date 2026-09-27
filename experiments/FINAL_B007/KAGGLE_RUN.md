# Kaggle production commands

Run from `/kaggle/working/amazon_ml`. Do not delete `artifacts/final_test`; it is
the resume/checkpoint store.

## 1. Freeze selected models and training-corpus statistics

```bash
TRAIN=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train

python -u scripts/freeze_final_b007.py \
  --b005-artifacts artifacts/experiments/B005 \
  --b005-metadata models/experiments/B005/metadata.json \
  --b007-artifacts artifacts/experiments/B007 \
  --source1 "$TRAIN/train_source1.tsv" \
  --source2 "$TRAIN/train_source2.tsv" \
  --source3 "$TRAIN/train_source3.tsv" \
  --ground-truth "$TRAIN/train_ground_truth.tsv" \
  --device gpu
```

## 2. France smoke test

```bash
TEST=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/test

python -u scripts/run_final_inference.py \
  --source1 "$TEST/test_source1.tsv" \
  --source2 "$TEST/test_source2.tsv" \
  --source3 "$TEST/test_source3.tsv" \
  --smoke-france 100 \
  --output-dir artifacts/final_test_smoke \
  --resume
```

Inspect `artifacts/final_test_smoke/France_smoke_100/france_smoke_report.json`.

## 3. Full country jobs

```bash
python -u scripts/run_final_inference.py --source1 "$TEST/test_source1.tsv" --source2 "$TEST/test_source2.tsv" --source3 "$TEST/test_source3.tsv" --country India --output-dir artifacts/final_test --resume
python -u scripts/run_final_inference.py --source1 "$TEST/test_source1.tsv" --source2 "$TEST/test_source2.tsv" --source3 "$TEST/test_source3.tsv" --country US --output-dir artifacts/final_test --resume
python -u scripts/run_final_inference.py --source1 "$TEST/test_source1.tsv" --source2 "$TEST/test_source2.tsv" --source3 "$TEST/test_source3.tsv" --country France --output-dir artifacts/final_test --resume
```

These commands are independent and may run on separate machines if they share or
later copy the complete country folders.

## 4. Merge, decide, write, and validate

```bash
python -u scripts/merge_final_predictions.py \
  --artifacts-dir artifacts/final_test \
  --country India --country US --country France \
  --test-dir "$TEST" \
  --train-source2 "$TRAIN/train_source2.tsv" \
  --train-source3 "$TRAIN/train_source3.tsv" \
  --output-dir artifacts/final_submission
```

Primary output: `artifacts/final_submission/matching_results.tsv`.
