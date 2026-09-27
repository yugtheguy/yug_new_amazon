# B005 — Full-corpus hard-negative retraining

B005 freezes B004 retrieval (`R0 + R1 + R2 + R3`) and measures only the value of
training the existing 32-feature LightGBM matcher on the real full-corpus candidate
distribution.

The first 50,000 non-validation ground-truth rows remain the experiment pool for
comparability with the repository's existing convention. A deterministic 90/10
country and match-count-stratified S1 split produces model-training and calibration
queries. The frozen 5,000 B003/B004 IDs are excluded from both.

All retrieved positives are retained. Missing positives are counted but never
force-added. Training negatives are selected deterministically with a configurable
budget, round-robin across multi-channel, R0, R1, R2, R3, near-address, near-name,
and top-RRF queues. Retrieval provenance is used only for mining and analysis; the
model matrix contains exactly the production 32 features.

The runner stops before training unless the frozen validation retrieval reproduces
recall `0.974627`, oracle Macro F0.5 `0.991071`, and exactly `875,027` pairs.
Thresholds are selected only on the internal calibration split, and every threshold
trial uses production target uniqueness.

The runner flushes event logs immediately and emits a 30-second liveness heartbeat
throughout long retrieval, preprocessing, feature extraction, and training stages.
