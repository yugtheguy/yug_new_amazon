# GPU2 final challenger

This is a strict cache-only runner for the 90-minute secondary GPU lane. It
consumes the completed B005 feature Parquets and B005 model metadata. It never
regenerates R0-R3, TF-IDF indexes, target indexes, or candidate unions.

The runner executes, in order:

1. **C1**: safe 40-feature LightGBM, calibration-only threshold/rule selection,
   then one frozen-5k evaluation.
2. **C2**: independent S2 and S3 LightGBM models, independent calibration, then
   one combined frozen-5k evaluation with target uniqueness.
3. **C3**: optional (`--run-c3`) six-configuration maximum LightGBM search.
   Calibration chooses one model and only that model is evaluated on frozen-5k.

`decision_rules.py` is independent of features/models so its selected C1 rule can
be applied directly to B007 or final-test probabilities.

Required outputs are written under `artifacts/experiments/GPU2_CHALLENGER/`.
Model files in that directory are handoff artifacts only; nothing is promoted.

Run the focused tests with:

```bash
python -m unittest discover -s experiments/GPU2_CHALLENGER -p 'test_*.py' -v
```

See `KAGGLE_RUN.md` for the execution command.
