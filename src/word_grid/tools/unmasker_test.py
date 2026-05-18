"""Tests for BERT unmasker output parsing."""

import unittest

from word_grid.tools.unmasker import _parse_fill_mask_output


class TestParseFillMaskOutput(unittest.TestCase):
    def test_single_mask(self):
        raw = [
            {"token_str": " on", "score": 0.5, "sequence": "x"},
            {"token_str": " in", "score": 0.3, "sequence": "y"},
        ]
        masks = _parse_fill_mask_output(raw)
        self.assertEqual(len(masks), 1)
        self.assertEqual(len(masks[0]["results"]), 2)
        self.assertEqual(masks[0]["results"][0]["word"], "on")

    def test_multiple_masks(self):
        raw = [
            [{"token_str": " on", "score": 0.5, "sequence": "a"}],
            [{"token_str": " chair", "score": 0.4, "sequence": "b"}],
        ]
        masks = _parse_fill_mask_output(raw)
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[1]["results"][0]["word"], "chair")


if __name__ == "__main__":
    unittest.main()
