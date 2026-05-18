"""Tests for rule-based POS combinations."""

import unittest

from word_grid.tools.pos_combinations import algorithm_rule_based


class TestRuleBased(unittest.TestCase):
    def test_length_4(self):
        combos = algorithm_rule_based(4, 10)
        self.assertTrue(all(len(c[0]) == 4 for c in combos))
        self.assertGreater(len(combos), 0)


if __name__ == "__main__":
    unittest.main()
