import unittest
from unittest.mock import patch

from mohim.motif import MotifExtractor, melodic_accompaniment, movement_score, repetition_score


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
        self.shape = (2, 1000)

    def __getitem__(self, _key):
        return _FakeStemSlice(self.rms)

    def clone(self):
        return _FakeStem(self.rms)

    def __add__(self, other):
        return _FakeStem(self.rms + other.rms)


class MotifScoreTests(unittest.TestCase):
    def test_melodic_accompaniment_excludes_drums_and_vocals(self):
        stems = {
            "guitar": _FakeStem(0.4),
            "piano": _FakeStem(0.3),
            "drums": _FakeStem(10.0),
            "vocals": _FakeStem(20.0),
        }

        result = melodic_accompaniment(stems, ("guitar", "piano", "bass", "other"))

        self.assertAlmostEqual(result.rms, 0.7)

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

    @patch("mohim.motif.score_repeating_motifs")
    def test_score_all_keeps_every_candidate_without_thresholding(self, score_motifs):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_motifs.side_effect = lambda stem, *_args: (
            [(100, 500, 0.20), (200, 600, 0.70)] if stem is guitar else [(300, 700, 0.40)]
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.score_all("song.wav", stems, 100)

        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([row["similarity"] for row in result["candidates"]], [0.2, 0.7, 0.4])
        self.assertAlmostEqual(result["melodic_accompaniment"].rms, 1.0)


if __name__ == "__main__":
    unittest.main()
