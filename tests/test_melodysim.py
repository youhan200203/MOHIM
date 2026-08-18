import unittest

from mohim.melodysim import phase_aligned_four_bar_segments


class MelodySimSegmentTests(unittest.TestCase):
    def test_segments_share_candidate_four_bar_phase(self):
        rows = phase_aligned_four_bar_segments(
            range(17),
            candidate_start_sec=4.0,
            bars=4,
            audio_duration_sec=16.0,
        )

        self.assertEqual([row["start_sec"] for row in rows], [0.0, 4.0, 8.0, 12.0])
        self.assertEqual([row["end_sec"] for row in rows], [4.0, 8.0, 12.0, 16.0])
        self.assertEqual([row["is_source"] for row in rows], [False, True, False, False])

    def test_segments_use_candidate_phase_when_first_downbeat_is_offset(self):
        rows = phase_aligned_four_bar_segments(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            candidate_start_sec=2.0,
            bars=4,
        )

        self.assertEqual([row["start_sec"] for row in rows], [2.0, 6.0])
        self.assertTrue(rows[0]["is_source"])

    def test_candidate_must_match_detected_downbeat(self):
        with self.assertRaisesRegex(ValueError, "detected downbeat"):
            phase_aligned_four_bar_segments(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                candidate_start_sec=0.2,
                bars=4,
            )


if __name__ == "__main__":
    unittest.main()
