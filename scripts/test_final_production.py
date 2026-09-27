from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_decision_rule as decision
import run_final_inference as inference


class FinalProductionTests(unittest.TestCase):
    def test_retrieval_constants_are_frozen(self):
        args = type("Args", (), {"source2": Path("s2"), "source3": Path("s3"), "query_chunk_size": 250, "progress_every_targets": 500000, "sparse_threads": 2})()
        frozen = inference.retrieval_args(args, Path("out"))
        self.assertEqual(50, frozen.r0_k)
        self.assertEqual(30, frozen.tfidf_k)
        self.assertEqual("char_wb", frozen.name_char_analyzer)
        self.assertEqual((3, 5), (frozen.name_char_ngram_min, frozen.name_char_ngram_max))

    def test_official_empty_encoding_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matching_results.tsv"
            decision.write_lists(path, ("source1_entity_id", "matched_entity_ids"), ["S1-2", "S1-1"], {"S1-1": {"S2-b", "S2-a"}})
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(["source1_entity_id", "matched_entity_ids"], rows[0])
            self.assertEqual(["S1-2", ""], rows[1])
            self.assertEqual(["S1-1", "S2-a,S2-b"], rows[2])

    def test_france_parser_and_accent_smoke(self):
        parsed = inference.B007F.parse_numeric_address("12 bis rue de l'Église, 75001 Paris")
        self.assertEqual("12", parsed["premise_base"])
        self.assertEqual("bis", parsed["premise_suffix"])
        self.assertEqual("75001", parsed["postal"])
        self.assertGreater(inference.B007F.char_similarity("Étoile", "Etoile"), 0.99)

    def test_frozen_schema_has_no_rank_or_rrf_features(self):
        schema = list(inference.B007R.ABLATIONS["B007_03"])
        self.assertEqual(121, len(schema))
        self.assertFalse(any("rank" in name.lower() or "rrf" in name.lower() for name in schema))

    def test_end_to_end_pair_feature_row_is_finite(self):
        try:
            __import__("rapidfuzz")
        except ImportError:
            self.skipTest("rapidfuzz is installed by the production requirements")
        production = inference.B003.import_production(inference.ROOT)
        evidence = inference.B004.empty_evidence("S1-q", "S2-c", "S2", "France")
        evidence.update({"found_by_existing": True, "existing_rank": 1, "existing_shared_keys": 2.0, "retriever_count": 1, "rrf_score": 1.0 / 61.0})
        stats = inference.B007F.CorpusStats()
        stats.add("Café Étoile SARL", "12 bis rue Victor Hugo 75001 Paris")
        stats.add("Cafe Etoile", "12 rue Victor Hugo 75001 Paris")
        candidates, features, parser = inference.build_rows(
            ["S1-q"], {("S1-q", "S2-c"): evidence},
            {"S1-q": ("Café Étoile SARL", "12 bis rue Victor Hugo 75001 Paris", "France")},
            {"S2-c": ("Cafe Etoile", "12 rue Victor Hugo 75001 Paris", "France")},
            stats, production, list(inference.B007R.ABLATIONS["B007_03"]),
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual(1, len(features))
        self.assertEqual("S2", features[0]["candidate_source"])
        self.assertGreater(parser["postal"], 0)


if __name__ == "__main__":
    unittest.main()
