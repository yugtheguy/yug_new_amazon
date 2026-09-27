# B007 Kaggle Execution

This experiment calculates textual and structural address evidence directly from original candidate strings, caching the feature extraction and running LightGBM ablations.

### 1. Requirements

Ensure the dataset is attached to your Kaggle notebook:
`/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train`

You also need the completed B005 artifacts (specifically its manifest and evaluation caches). If you ran B005 locally or in Kaggle, make sure they are present at `/kaggle/working/amazon_ml/artifacts/experiments/B005` (or adjust the path).

### 2. Execution Command

Run the following command from the repository root:

```bash
# Locate data root dynamically if needed, but assuming standard Kaggle path:
DATA_ROOT="/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/train"

python experiments/B007/run_b007.py \
    --b005-artifacts artifacts/experiments/B005 \
    --b005-metadata artifacts/experiments/B005/model_metadata.json \
    --source1 $DATA_ROOT/train_source1.tsv \
    --source2 $DATA_ROOT/train_source2.tsv \
    --source3 $DATA_ROOT/train_source3.tsv \
    --ground-truth $DATA_ROOT/train_ground_truth.tsv \
    --val-ids experiments/val_s1_ids.txt \
    --output-dir artifacts/experiments/B007 \
    --device gpu
```

### 3. Estimated Runtime

- **Text Extraction:** Iterating through the three source TSVs to load texts for the candidate subset should take **~2-5 minutes**.
- **Feature Generation:** B007 computes IDF statistics and parses ~875k+ eval candidates and 2.4M train candidates. This Python loop may take **~5-15 minutes**.
- **Model Training:** LightGBM GPU training on 2.4M pairs for 4 ablations should take **~2-4 minutes** each.
- **Total estimated time:** **15–30 minutes** end-to-end.
