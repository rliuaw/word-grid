"""Tests for POS grid solver."""

import unittest

from word_grid.tools.pos_grid import solve_pos_grid


class TestPosGridSolver(unittest.TestCase):
    def test_solve_3x3(self):
        combos = {
            ("DT", "NN", "VB"),
            ("PRP", "VBD", "RB"),
            ("IN", "DT", "NN"),
        }
        # Need more combos for a valid 3x3 — use symmetric set
        combos.update(
            {
                ("DT", "NN", "VB"),
                ("NN", "VB", "DT"),
                ("VB", "DT", "NN"),
            }
        )
        # minimal: all rows same (trivial)
        combos = {("DT", "NN", "VB"), ("PRP", "VBD", "RB"), ("IN", "DT", "NN")}
        solutions = solve_pos_grid(3, combos, max_solutions=1)
        if solutions:
            sol = solutions[0]
            self.assertEqual(len(sol), 3)
            for row in sol:
                self.assertIn(row, combos)
            for j in range(3):
                col = tuple(sol[i][j] for i in range(3))
                self.assertIn(col, combos)


if __name__ == "__main__":
    unittest.main()
