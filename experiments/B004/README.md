# B004 — Multi-View High-Recall Retrieval

B004 measures whether three sparse retrieval views can close the candidate-oracle
gap measured by B003. It does not train or score a matcher.

## Channels

- `R0_EXISTING`: unchanged typed exact blocker, K50
- `R1_NAME_CHAR`: target-fitted `char_wb` TF-IDF over normalized core names,
  character 3–5 grams, top 30 independently for S2 and S3
- `R2_ADDRESS_WORD`: target-fitted word 1–2 gram TF-IDF over normalized addresses,
  top 30 independently for S2 and S3
- `R3_NAME_WORD`: target-fitted word 1–2 gram TF-IDF over normalized core names,
  top 30 independently for S2 and S3
- `R4_REVERSE`: deliberately deferred; it must not delay measurement of R1–R3

Vectorizers are fitted independently for each country and target source using only
unlabelled target text. Ground truth is loaded only after all retrieval checkpoints
are complete.

## Union and cap analysis

The canonical pair key is `(query_id, candidate_id)`; the source is retained as an
explicit field and is also encoded in the candidate ID namespace. All ranks, cosine
scores, and channel indicators survive the union.

The competition problem statement specifies no hard candidate cap. B004 therefore
uses the uncapped union for its primary oracle analysis. It also reports a capped-50
diagnostic using RRF with `C=60`, while guaranteeing the top 10 R0 candidates before
filling remaining positions by fused rank.

## Checkpointing

Each country/source/channel/query chunk is a compressed Parquet file with an atomic
`.DONE` marker. `--resume` validates a configuration fingerprint before reusing it.
Large outputs belong under `artifacts/experiments/B004/` and are ignored by Git.

## Primary decision metric

B004 success is candidate link recall and candidate-restricted entity-level Macro
F0.5—not stale LightGBM performance. R0 must first reproduce the B003 recall and
oracle within the configured tolerance.
