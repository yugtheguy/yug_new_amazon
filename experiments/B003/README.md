# B003 — Frozen Full-Corpus K50 Baseline

B003 measures the current entity-resolution system without changing its matching
algorithm. It uses the current production normalization, typed blocking keys,
posting-list pruning, `Counter.most_common(50)`, 32 features, both checked-in
LightGBM models, fixed thresholds, and greedy target uniqueness.

The runner differs from the historical validation only in methodology: every
validation query retrieves against the complete S2/S3 target population for its
country. Ground truth is not loaded until raw candidate generation and scoring are
complete.

## Outputs

All runtime outputs belong under `artifacts/experiments/B003/`, which is ignored by
Git. Candidate and score checkpoints use Zstandard-compressed Parquet. Each chunk
has a `.DONE` marker. `--resume` verifies the frozen configuration fingerprint
before reusing checkpoints.

Key files:

- `manifest.json`: code, data, model, validation and configuration identity
- `metrics.json`: complete model, threshold, country and error metrics
- `candidate_metrics.json`: retrieval recall and candidate-restricted oracle
- `telemetry.jsonl`: wall time and memory observations
- `raw_candidates/<country>/`: label-free candidate and score checkpoints
- `diagnostic_candidates/<country>/`: post-generation candidates annotated with GT
- `B003.DONE`: successful completion marker

## Semantic constraints

- `--top-k` is accepted for transparency but B003 rejects values other than 50.
- Thresholds are fixed; no threshold search is performed.
- Candidate rank is saved for diagnostics and never passed to either model.
- Target-ID existence checking is enabled by default.
- Set/Counter iteration is intentionally not sorted because production does not sort
  it. Set `PYTHONHASHSEED=0` before process start to freeze tie ordering.

See `KAGGLE_RUN.md` for the exact execution command.
