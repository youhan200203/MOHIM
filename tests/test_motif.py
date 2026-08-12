import unittest

from mohim.motif import movement_score, repetition_score


class MotifScoreTests(unittest.TestCase):
    def test_repeated_phrase_scores_above_flat_phrase(self):
        repeated = [60, 62, 64, 65, 60, 62, 64, 65]
        flat = [60, 60, 60, 60, 60, 60, 60, 60]
        self.assertGreater(repetition_score(repeated), repetition_score(flat))
        self.assertGreater(movement_score(repeated), movement_score(flat))

    def test_short_sequence_scores_zero(self):
        self.assertEqual(movement_score([60]), 0.0)
        self.assertEqual(repetition_score([60, 62]), 0.0)


if __name__ == "__main__":
    unittest.main()
