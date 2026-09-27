# Final production — B007_03

This lane freezes the measured B007_03 champion and performs B004-identical,
within-country test retrieval followed by B007_03 scoring. It contains no new
modeling experiment. Expensive country stages are checkpointed and resumable.

Training/inference semantics for B007 corpus statistics are explicit: B007 was
trained with document frequencies and exact name/address frequencies accumulated
over every row of `train_source1.tsv`, `train_source2.tsv`, and
`train_source3.tsv`. `freeze_final_b007.py` persists that object and test inference
loads it. It is never refit on France or other test records.

The official scored filename is `matching_results.tsv`; `candidate_pairs.tsv` is
also generated. `submission_b007_03.tsv` is an identical convenience copy.
