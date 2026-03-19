import unittest

from word_grid.grid import parse_grid_from_text

# Run using uv
# uv test src/word_grid/grid_test.py

class TestGrid(unittest.TestCase):
    def test_parse_grid_from_text(self):
        text = r"""The quick brown fox jumps over the lazy dog.
A brave adventurer seeks treasure in a dark forest, finding danger.
Danger lurks everywhere, but hope remains strong, driving the hero on.
Onward, the hero fights dragons and overcomes obstacles, finally reaching the treasure.
The treasure is found, but the true reward is the journey completed, the hero returns home."""
        grid = parse_grid_from_text(text, n=5)
        print(grid)
        self.assertEqual(grid.n, 5)
        self.assertEqual(grid.cells, [["Ian", "Was", "Seen."], ["Left,", "He", "Found."], ["Laredo.", "Lost?", "Reborn."]])

if __name__ == "__main__":
    unittest.main()