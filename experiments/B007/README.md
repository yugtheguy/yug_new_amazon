# B006A — B006 forensic audit

B006A never regenerates retrieval or pair features. It first reproduces B005 using
only the cached 32-feature columns. A miss beyond ±0.002 stops all feature
ablations. On reproduction success, it audits canonical triple keys, missing/rank
encoding, RRF, gap direction, feature distributions, and runs eight isolated
retrieval-feature family experiments with identical B005 rows and hyperparameters.

Original B006 gaps are non-positive (`candidate - best`). The isolated gap ablation
uses the preferred non-negative form (`best - candidate`); the full B006 ablation
retains the original encoding for faithful reproduction.
