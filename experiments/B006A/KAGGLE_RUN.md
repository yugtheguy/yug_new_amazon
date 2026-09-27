# Kaggle execution — B006A forensic audit

The existing B005 `features/` cache and model metadata are required. No retrieval
or fuzzy pair features are regenerated.

```bash
cd /kaggle/working/amazon_ml
python -m pip install -r experiments/B006A/requirements.txt

DATA_ROOT=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train

PYTHONHASHSEED=0 python -u experiments/B006A/run_b006a.py \
  --b005-artifacts /kaggle/working/amazon_ml/artifacts/experiments/B005 \
  --b005-metadata /kaggle/working/amazon_ml/models/experiments/B005/metadata.json \
  --b006-artifacts /kaggle/working/amazon_ml/artifacts/experiments/B006 \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --output-dir /kaggle/working/amazon_ml/artifacts/experiments/B006A \
  --device auto \
  --heartbeat-seconds 30
```

The first printed model is `B006A_00_REPRO`. If it misses `0.926948 ± 0.002`, the
runner writes diagnostics and exits before training any retrieval-feature model.

Key outputs:

```text
reproduction.json
pipeline_equivalence.json
merge_integrity.json
distribution_audit.json
encoding_audit.json
ablation_metrics.json
positive_recall_slices.json
root_cause.json
B006A_FORENSIC_REPORT.md
telemetry.jsonl
```
