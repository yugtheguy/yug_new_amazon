# GPU2 model-diversity challenge

This runner consumes only the completed B005/B006A cache and the frozen safe
40-feature schema. It never regenerates retrieval, candidates, indexes, or B007
features.

- D1 trains one CatBoost GPU model by default. `--try-second-catboost` permits
  exactly one additional fixed configuration only when at least 30 minutes remain.
- D2 is opt-in with `--run-xgboost`; it uses one fixed GPU configuration and starts
  only after CatBoost completes, clears the stop rule, and sufficient time remains.
- Optional B007 calibration/evaluation probability Parquets enable a tiny fixed
  blend grid. Canonical key equality is asserted before blending.
- Every model writes aligned calibration and evaluation raw-probability Parquets.

Use `KAGGLE_RUN.md` for exact commands.
