import unittest

from mohim.motif import movement_score, repetition_score


class MotifScoreTests(unittest.TestCase):
    def test_legacy_fractional_pitch_movement(self):
        self.assertAlmostEqual(movement_score([60.0, 60.1]), 0.706)

    def test_legacy_flat_phrase_repetition(self):
        flat = [60, 60, 60, 60, 60, 60, 60, 60]
        self.assertEqual(repetition_score(flat), 1.0)

    def test_short_sequence_scores_zero(self):
        self.assertEqual(movement_score([60]), 0.0)
        self.assertEqual(repetition_score([60, 62]), 0.0)


if __name__ == "__main__":
    unittest.main()
