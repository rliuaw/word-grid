"""Tests for POS vocab filtering helpers."""

import unittest

from word_grid.ml.pos_vocab import is_punctuation_token, word_matches_pos


class TestPosVocabFilter(unittest.TestCase):
    def test_punctuation(self):
        self.assertTrue(is_punctuation_token("."))
        self.assertTrue(is_punctuation_token(","))
        self.assertFalse(is_punctuation_token("chair"))

    def test_word_matches_pos(self):
        self.assertTrue(word_matches_pos("chair", "NN"))
        self.assertFalse(word_matches_pos("chair", "VB"))
        self.assertFalse(word_matches_pos(".", "NN"))


if __name__ == "__main__":
    unittest.main()
