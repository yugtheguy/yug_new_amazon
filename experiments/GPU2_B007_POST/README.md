# GPU2 B007 post-processing

Consumes four saved B007 probability Parquets and ground truth. E1 fixed-weight
blending and E2 local threshold refinement run by default. E3 competition rules
are opt-in with `--run-e3`. No retrieval, feature generation, or model fit occurs.

All joins use `(query_id, candidate_id, candidate_source)` with strict equality,
duplicate, label, source, and country assertions. See `KAGGLE_RUN.md`.
