import unittest

from substrate_rules import RuleBook


class RuleBookTests(unittest.TestCase):
    def test_invariance_requires_variation_in_claimed_variables(self):
        book = RuleBook(min_group=2, min_coverage=0.8)
        book.add_invariance(
            "target", lambda row: row["target"], allowed=["group"], vary=["x"]
        )
        for _ in range(8):
            book.observe({"group": 0, "x": 0, "target": 0})
        for _ in range(2):
            book.observe({"group": 1, "x": 1, "target": 1})

        invariants, _ = book.scan()
        self.assertFalse(invariants[0].holds)

    def test_determinacy_requires_sufficient_group_coverage(self):
        book = RuleBook(min_group=2, min_coverage=0.8)
        book.add_determinacy("outcome", lambda row: row["outcome"], ["unique_key"])
        for key in range(10):
            book.observe({"unique_key": key, "outcome": key % 2})

        _, determinants = book.scan()
        self.assertFalse(determinants[0].exact)
        self.assertEqual(determinants[0].coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
