# B006 — Retrieval provenance, rank, and score features

B006 is a strict cached ablation over B005. It reads the completed B005 feature
Parquets directly, preserves all 32 base features and every selected training,
calibration, and evaluation row, then appends 37 retrieval-evidence features.

No retrieval, normalization, fuzzy pair feature, hard-negative selection, split, or
LightGBM hyperparameter is regenerated or changed. B005 model metadata supplies the
exact hyperparameters. Thresholds are recalibrated on the cached internal
calibration split using production target uniqueness.

Missing R0 rank is encoded as 51; missing R1/R2/R3 rank as 31. Missing scores and
gaps use 0 alongside explicit `found_R*` flags. Reciprocal rank is zero when missing
and otherwise `1 / (60 + rank)`, matching B004.
