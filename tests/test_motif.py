import unittest
from unittest.mock import patch

from mohim.motif import MotifExtractor, movement_score, repetition_score


class _FakeStemSlice:
    def __init__(self, rms):
        self.rms = rms

    def square(self):
        return self

    def mean(self):
        return self

    def sqrt(self):
        return self.rms


class _FakeStem:
    def __init__(self, rms):
        self.rms = rms

    def __getitem__(self, _key):
        return _FakeStemSlice(self.rms)


class MotifScoreTests(unittest.TestCase):
    def test_legacy_fractional_pitch_movement(self):
        self.assertAlmostEqual(movement_score([60.0, 60.1]), 0.706)

    def test_legacy_flat_phrase_repetition(self):
        flat = [60, 60, 60, 60, 60, 60, 60, 60]
        self.assertEqual(repetition_score(flat), 1.0)

    def test_short_sequence_scores_zero(self):
        self.assertEqual(movement_score([60]), 0.0)
        self.assertEqual(repetition_score([60, 62]), 0.0)

    @patch("mohim.motif.find_repeating_motif")
    def test_extract_compares_first_passing_motif_from_each_stem(self, find_motif):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        find_motif.side_effect = lambda stem, *_args: (
            (100, 500, 0.60) if stem is guitar else (200, 600, 0.62)
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.extract("song.wav", stems, None, 100)

        self.assertEqual(find_motif.call_count, 2)
        self.assertEqual(result["stem_name"], "guitar")
        self.assertAlmostEqual(result["stem_scores"]["guitar"]["dominance"], 0.8)
        self.assertAlmostEqual(result["stem_scores"]["guitar"]["total"], 0.63)
        self.assertAlmostEqual(result["stem_scores"]["piano"]["total"], 0.557)

    @patch("mohim.motif.find_repeating_motif")
    def test_extract_records_stems_without_a_passing_motif(self, find_motif):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        find_motif.side_effect = lambda stem, *_args: (
            (100, 500, 0.60) if stem is guitar else None
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.extract("song.wav", stems, None, 100)

        self.assertFalse(result["stem_scores"]["piano"]["matched"])
        self.assertIsNone(result["stem_scores"]["piano"]["start_sec"])


if __name__ == "__main__":
    unittest.main()
