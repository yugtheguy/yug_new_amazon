import unittest
import b007_features as b007

class TestB007Features(unittest.TestCase):
    def setUp(self):
        self.stats = b007.CorpusStats()
        self.stats.add("Company A", "12 rue Victor Hugo")
        self.stats.add("Company B", "12 bis rue Victor Hugo")
        self.stats.add("International Business Machines", "94105")
        self.stats.add("IBM", "94105-1234")
        self.stats.add("Société Générale SARL", "400001")
        self.stats.add("Societe Generale", "400021")
        self.stats.add("Test", "22 Main Rd Flat 7")
        self.stats.add("Test2", "22 Main Road Unit 9")
        self.stats.add("Test3", "72 Main Rd Flat 7")
        self.stats.add("Test4", "7 Main Rd Flat 22")
        
    def generate(self, q_name, q_addr, c_name, c_addr):
        texts = {"Q1": (q_name, q_addr), "C1": (c_name, c_addr)}
        f = b007.generate_features("Q1", "C1", texts, self.stats)
        return dict(zip(b007.B007_FEATURE_NAMES, f))
        
    def test_premise_and_unit_conflict(self):
        # 22 Main Rd Flat 7 vs 22 Main Road Unit 9
        f = self.generate("T1", "22 Main Rd Flat 7", "T2", "22 Main Road Unit 9")
        self.assertEqual(f["premise_exact"], 1.0)
        self.assertEqual(f["unit_conflict"], 1.0)
        
    def test_premise_conflict_unit_exact(self):
        # 22 Main Rd Flat 7 vs 72 Main Rd Flat 7
        f = self.generate("T1", "22 Main Rd Flat 7", "T2", "72 Main Rd Flat 7")
        self.assertEqual(f["premise_conflict"], 1.0)
        self.assertEqual(f["unit_exact"], 1.0)
        
    def test_cross_role_conflict(self):
        # 22 Main Rd Flat 7 vs 7 Main Rd Flat 22
        f = self.generate("T1", "22 Main Rd Flat 7", "T2", "7 Main Rd Flat 22")
        self.assertEqual(f["numeric_cross_role_conflict"], 1.0)

    def test_france_suffix(self):
        # 12 rue Victor Hugo vs 12 bis rue Victor Hugo
        f = self.generate("T1", "12 rue Victor Hugo", "T2", "12 bis rue Victor Hugo")
        self.assertEqual(f["premise_base_exact"], 1.0)
        self.assertEqual(f["premise_suffix_conflict"], 1.0)

    def test_accent_and_legal_form(self):
        # Société Générale SARL vs Societe Generale
        f = self.generate("Société Générale SARL", "A", "Societe Generale", "A")
        self.assertGreater(f["core_name_similarity"], 0.9)
        self.assertEqual(f["legal_form_left_present"], 1.0)
        
    def test_postal_base(self):
        # 94105 vs 94105-1234
        f = self.generate("T1", "94105", "T2", "94105-1234")
        self.assertEqual(f["postal_base_exact"], 1.0)

    def test_postal_prefix_conflict(self):
        # 400001 vs 400021
        f = self.generate("T1", "400001", "T2", "400021")
        self.assertEqual(f["postal_conflict"], 1.0)
        self.assertEqual(f["postal_prefix_match"], 0.0)
        
        # 40000 vs 400001
        f2 = self.generate("T1", "40000", "T2", "400001")
        # our regex is \b(\d{5,6})\b.
        # if one is 40000 (5) and one is 400001 (6)
        # prefix match should be 1.0, conflict 0.0
        self.assertEqual(f2["postal_prefix_match"], 1.0)
        self.assertEqual(f2["postal_conflict"], 0.0)

    def test_missing_address(self):
        f = self.generate("T1", "", "T2", "123 Main")
        self.assertEqual(f["address_missing_one_side"], 1.0)
        self.assertEqual(f["premise_conflict"], 0.0)
        self.assertEqual(f["postal_conflict"], 0.0)

if __name__ == '__main__':
    unittest.main()
