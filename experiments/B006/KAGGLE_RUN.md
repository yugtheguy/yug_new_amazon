# Kaggle execution — B006

## Required cached B005 artifacts

B006 does not regenerate retrieval or the 32 fuzzy pair features. Keep or upload:

```text
artifacts/experiments/B005/manifest.json
artifacts/experiments/B005/features/train/*.parquet
artifacts/experiments/B005/features/calibration/*.parquet
artifacts/experiments/B005/features/evaluation/*.parquet
models/experiments/B005/metadata.json
```

If the Kaggle session was restarted, attach these as a Kaggle dataset and point the
two cache arguments at its mounted paths. Compact result JSON files alone are not
sufficient; the `features/` Parquets are required.

## Primary command

```bash
cd /kaggle/working/amazon_ml

git fetch origin \
  refs/heads/exp/B006-retrieval-provenance:refs/remotes/origin/exp/B006-retrieval-provenance
git checkout -B exp/B006-retrieval-provenance \
  refs/remotes/origin/exp/B006-retrieval-provenance

python -m pip install -r experiments/B006/requirements.txt

DATA_ROOT=/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train
B005_ARTIFACTS=/kaggle/working/amazon_ml/artifacts/experiments/B005
B005_METADATA=/kaggle/working/amazon_ml/models/experiments/B005/metadata.json
B006_OUT=/kaggle/working/amazon_ml/artifacts/experiments/B006
B006_MODEL=/kaggle/working/amazon_ml/models/experiments/B006

PYTHONHASHSEED=0 python -u experiments/B006/run_b006.py \
  --b005-artifacts "$B005_ARTIFACTS" \
  --b005-metadata "$B005_METADATA" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --output-dir "$B006_OUT" \
  --model-dir "$B006_MODEL" \
  --device auto \
  --heartbeat-seconds 30
```

The runner validates B005 completion, retrieval equivalence, exact split counts,
disjoint IDs, 2,401,548 training rows, 151,548 positives, 875,027 evaluation pairs,
the 32-feature schema, and the candidate oracle before fitting.

Inspect:

```bash
cat "$B006_OUT/evaluation_metrics.json"
cat "$B006_OUT/feature_importance.json"
cat "$B006_OUT/error_slices.json"
cat "$B006_MODEL/metadata.json"
```
