# Running B003 on Kaggle

## 1. Open a Kaggle notebook or terminal

Enable internet only if the notebook image is missing a required Python package.
GPU is optional; B003 uses CPU/RAM for blocking and only scores existing models.

## 2. Make the B003 working-tree snapshot available

B003 is intentionally uncommitted, so cloning the public repository alone will not
contain this experiment. Upload the current repository working tree as a private
Kaggle Dataset (preserving `.git`, but excluding local bytecode and generated
artifacts), attach it to the notebook, and copy it into writable storage:

```bash
cp -R /kaggle/input/akash-ka-repo-b003/AKASH_KA_REPO /kaggle/working/amazon_ml
cd /kaggle/working/amazon_ml
```

Adjust the attached-dataset path to its actual Kaggle slug. Before running, verify
that `experiments/B003/run_b003.py` exists and that production files are based on
audited commit `f8c234c111f6abc9951f2706b06ab7a0c5b9ec9c`.

## 3. Install dependencies

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python -m pip install -r experiments/B003/requirements.txt
```

## 4. Set dataset paths

Edit only these shell variables to match the attached Kaggle Dataset:

```bash
DATA_ROOT=/kaggle/input/amazon-ml-challenge-2026/train
OUT=/kaggle/working/amazon_ml/artifacts/experiments/B003
```

The directory must contain `train_source1.tsv`, `train_source2.tsv`,
`train_source3.tsv`, and `train_ground_truth.tsv`.

## 5. Run the frozen 5,000-query baseline

`PYTHONHASHSEED` must be set before Python starts because production candidate ties
depend on set/Counter insertion order.

```bash
PYTHONHASHSEED=0 python experiments/B003/run_b003.py \
  --source1 "$DATA_ROOT/train_source1.tsv" \
  --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --val-limit 5000 \
  --top-k 50 \
  --model-a models/final_entity_matcher.joblib \
  --model-b models/baseline_reproduced.joblib \
  --output-dir "$OUT"
```

Add `--hash-inputs` when the extra sequential reads of large inputs are acceptable.

## 6. Resume after interruption

Run the same command with `--resume` appended. Completed Parquet chunks with DONE
markers are reused. A changed model, validation subset, input path, K, pruning rule,
or Git commit changes the configuration fingerprint and blocks unsafe resume.

```bash
PYTHONHASHSEED=0 python experiments/B003/run_b003.py \
  --source1 "$DATA_ROOT/train_source1.tsv" \
  --source2 "$DATA_ROOT/train_source2.tsv" \
  --source3 "$DATA_ROOT/train_source3.tsv" \
  --ground-truth "$DATA_ROOT/train_ground_truth.tsv" \
  --val-ids experiments/val_s1_ids.txt \
  --val-limit 5000 --top-k 50 \
  --model-a models/final_entity_matcher.joblib \
  --model-b models/baseline_reproduced.joblib \
  --output-dir "$OUT" --resume
```

## 7. Inspect progress

The running cell prints live `[B003 +...s]` updates during target scans, index
construction, candidate chunks, strict validation, and final evaluation. You can
also inspect the durable telemetry stream from another cell:

```bash
tail -n 20 "$OUT/telemetry.jsonl"
find "$OUT/raw_candidates" -name '*.DONE' | wc -l
cat "$OUT/manifest.json"
```

## 8. Package important outputs

The raw and diagnostic Parquet checkpoints are useful for later analysis but are not
required for the first result review. Package the compact files first:

```bash
cd "$OUT"
zip -9 /kaggle/working/B003_core_results.zip \
  manifest.json metrics.json candidate_metrics.json telemetry.jsonl B003.DONE
```

Download `/kaggle/working/B003_core_results.zip` and return it for review. Do not
start B004 until B003 results have been checked.
