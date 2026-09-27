"""Small B004 retrieval tests; no competition data is loaded."""

from __future__ import annotations

import argparse
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

if "text_unidecode" not in sys.modules:
    sys.modules["text_unidecode"] = types.SimpleNamespace(unidecode=lambda value: value)

import run_b004  # noqa: E402

SRC = REPO_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))
import blocking  # noqa: E402
import normalization as norm  # noqa: E402
from evaluation import evaluate_predictions  # noqa: E402


def vector_args() -> argparse.Namespace:
    return argparse.Namespace(
        tfidf_min_df=1,
        name_char_analyzer="char_wb",
        name_char_ngram_min=3,
        name_char_ngram_max=5,
        name_char_max_features=None,
        address_word_max_features=None,
        name_word_max_features=None,
    )


def retrieve(channel: str, targets: list[tuple[str, str, str]], query: tuple[str, str]) -> list[str]:
    texts = [run_b004.normalize_channel_text(channel, name, address, norm) for _, name, address in targets]
    query_text = run_b004.normalize_channel_text(channel, query[0], query[1], norm)
    vectorizer = run_b004.vectorizer_for(channel, vector_args())
    target_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform([query_text])
    result = run_b004.sparse_topn(query_matrix, target_matrix, 3, 1e-6, 1)
    rows = run_b004.topn_rows(
        result, ["S1-Q"], [item[0] for item in targets], "S2", "US", channel
    )
    return [row["candidate_id"] for row in rows]


class B004Tests(unittest.TestCase):
    def test_name_char_retrieves_typo_variant(self) -> None:
        targets = [
            ("S2-1", "McDonalds Restaurant", "1 Main St"),
            ("S2-2", "Completely Different", "2 Oak St"),
        ]
        self.assertEqual(retrieve("R1_NAME_CHAR", targets, ("Mc Donalds", ""))[0], "S2-1")

    def test_name_word_handles_reordered_tokens(self) -> None:
        targets = [
            ("S2-1", "Ganesh Shree Traders", "1 Main St"),
            ("S2-2", "Ganesh Medical Center", "2 Oak St"),
        ]
        self.assertEqual(retrieve("R3_NAME_WORD", targets, ("Shree Ganesh Traders", ""))[0], "S2-1")

    def test_address_word_finds_same_address_variant(self) -> None:
        targets = [
            ("S2-1", "Different Name", "10 Main Street Suite 2"),
            ("S2-2", "Other", "99 Oak Road"),
        ]
        self.assertEqual(retrieve("R2_ADDRESS_WORD", targets, ("Unknown", "10 Main St Ste 2"))[0], "S2-1")

    def test_union_deduplicates_and_retains_provenance(self) -> None:
        rows = [
            {"query_id": "S1-1", "candidate_id": "S2-1", "candidate_source": "S2", "country": "US", "channel": "R0_EXISTING", "rank": 6, "score": 3.0},
            {"query_id": "S1-1", "candidate_id": "S2-1", "candidate_source": "S2", "country": "US", "channel": "R1_NAME_CHAR", "rank": 2, "score": 0.91},
            {"query_id": "S1-1", "candidate_id": "S2-1", "candidate_source": "S2", "country": "US", "channel": "R3_NAME_WORD", "rank": 1, "score": 0.95},
        ]
        merged = run_b004.merge_provenance(rows, 60.0)
        self.assertEqual(len(merged), 1)
        evidence = merged[("S1-1", "S2-1")]
        self.assertTrue(evidence["found_by_existing"])
        self.assertTrue(evidence["found_by_name_char"])
        self.assertTrue(evidence["found_by_name_word"])
        self.assertEqual(evidence["retriever_count"], 3)
        self.assertEqual(evidence["existing_rank"], 6)
        self.assertAlmostEqual(evidence["name_char_score"], 0.91)

    def test_rrf_cap_preserves_high_rank_r0(self) -> None:
        rows = []
        for rank in range(1, 61):
            row = run_b004.empty_evidence("S1-1", f"S2-{rank}", "S2", "US")
            if rank <= 12:
                row["found_by_existing"] = True
                row["existing_rank"] = rank
            else:
                row["found_by_name_char"] = True
                row["name_char_rank"] = rank - 12
            row["rrf_score"] = 1 / (60 + (row["existing_rank"] or row["name_char_rank"]))
            rows.append(row)
        selected = run_b004.capped_rrf(rows, 50, 10)
        selected_ids = {row["candidate_id"] for row in selected}
        self.assertEqual(len(selected), 50)
        for rank in range(1, 11):
            self.assertIn(f"S2-{rank}", selected_ids)

    def test_candidate_oracle_matches_b003_definition(self) -> None:
        truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
        candidates = {"S1-1": {"S2-1", "S2-X"}, "S1-2": {"S3-X"}}
        oracle = {query_id: truth[query_id] & candidates[query_id] for query_id in truth}
        result = evaluate_predictions(truth, oracle)
        self.assertAlmostEqual(result["macro_f05"], (5 / 6 + 1) / 2)

    def test_r0_reproduces_production_candidate_behavior(self) -> None:
        targets = {
            "S2-1": ("Alpha Foods", "10 Main St", "US"),
            "S2-2": ("Alpha Tools", "11 Main St", "US"),
            "S3-1": ("Beta Foods", "10 Oak Rd", "US"),
        }
        expected_index = blocking.build_inverted_index_for_country(targets)
        _, actual_index, _ = run_b004.B003.build_production_index(
            targets, norm.normalize_name, norm.normalize_address,
            blocking.get_blocking_keys, 300, 150,
        )
        self.assertEqual(expected_index, actual_index)
        expected = blocking.retrieve_candidates_for_s1(
            "Alpha Foods", "10 Main St", "US", expected_index, top_k=50
        )
        actual = run_b004.B003.retrieve_production_candidates(
            "Alpha Foods", "10 Main St", "US", actual_index,
            blocking.get_blocking_keys, 50,
        )
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
