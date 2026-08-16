import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from mohim.motif import (
    MotifConfig,
    MotifExtractor,
    _max_shifted_cosine_similarity,
    find_repeating_motif,
    melodic_accompaniment,
    movement_score,
    repetition_score,
    score_repeating_motifs,
)


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


class _FakeArray:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeAudioStem:
    def __init__(self, values):
        self.values = values

    def mean(self, _axis):
        return _FakeArray(self.values)


class MotifScoreTests(unittest.TestCase):
    def test_onset_similarity_uses_best_shift_within_three_frames(self):
        first = np.zeros(256)
        second = np.zeros(256)
        first[100] = 1.0
        second[103] = 1.0

        def cosine_similarity(left, right):
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            score = float(np.dot(left.ravel(), right.ravel()) / denominator) if denominator else 0.0
            return np.array([[score]])

        similarity = _max_shifted_cosine_similarity(
            first,
            second,
            max_shift=3,
            cosine_similarity=cosine_similarity,
        )

        self.assertAlmostEqual(similarity, 1.0)

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

    @patch.object(MotifExtractor, "score_all")
    def test_extract_reuses_scores_and_compares_first_passing_motifs(self, score_all):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.59,
                    "chroma_similarity": 0.8, "similarity": 0.737,
                    "pitch_class_span": 0.7,
                },
                {
                    "stem_name": "guitar", "start_sec": 2.0, "end_sec": 6.0,
                    "active_ratio": 0.9, "onset_similarity": 0.61,
                    "chroma_similarity": 0.7, "similarity": 0.673,
                    "pitch_class_span": 0.31,
                },
                {
                    "stem_name": "piano", "start_sec": 3.0, "end_sec": 7.0,
                    "active_ratio": 0.8, "onset_similarity": 0.62,
                    "chroma_similarity": 0.9, "similarity": 0.819,
                    "pitch_class_span": 0.5,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.extract(
            "song.wav",
            stems,
            None,
            100,
            scored_result=score_all.return_value,
        )

        score_all.assert_not_called()
        self.assertEqual(result["stem_name"], "piano")
        self.assertAlmostEqual(result["stem_scores"]["guitar"]["start_sec"], 2.0)
        self.assertAlmostEqual(result["stem_scores"]["piano"]["similarity"], 0.819)
        self.assertNotIn("dominance", result["stem_scores"]["guitar"])
        self.assertNotIn("total", result["stem_scores"]["guitar"])

    @patch.object(MotifExtractor, "score_all")
    def test_extract_records_stems_without_a_passing_motif(self, score_all):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.7, "similarity": 0.67,
                    "pitch_class_span": 0.31,
                },
                {
                    "stem_name": "piano", "start_sec": 2.0, "end_sec": 6.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.8, "similarity": 0.74,
                    "pitch_class_span": 0.16666666666666666,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.extract("song.wav", stems, None, 100)

        self.assertFalse(result["stem_scores"]["piano"]["matched"])
        self.assertIsNone(result["stem_scores"]["piano"]["start_sec"])

    @patch("mohim.motif.pitch_class_span", side_effect=[0.25, 0.5, 0.75])
    @patch("mohim.motif.score_repeating_motifs")
    def test_score_all_keeps_every_filtered_candidate(self, score_motifs, span):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_motifs.side_effect = lambda stem, *_args: (
            [(100, 500, 0.10, 0.24, 0.20, 0.8), (200, 600, 0.60, 0.74, 0.70, 0.9)]
            if stem is guitar
            else [(300, 700, 0.30, 0.44, 0.40, 0.8)]
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.score_all("song.wav", stems, 100)

        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([row["candidate_index"] for row in result["candidates"]], [0, 1, 0])
        self.assertEqual([row["similarity"] for row in result["candidates"]], [0.2, 0.7, 0.4])
        self.assertEqual(
            [row["pitch_class_span"] for row in result["candidates"]],
            [0.25, 0.5, 0.75],
        )
        self.assertEqual(span.call_count, 3)
        self.assertNotIn("dominance", result["candidates"][0])
        self.assertNotIn("total", result["candidates"][0])
        self.assertAlmostEqual(result["melodic_accompaniment"].rms, 1.0)

    @patch("mohim.motif.pitch_class_span", side_effect=[0.8, 0.25])
    @patch("mohim.motif.score_repeating_motifs")
    def test_find_repeating_motif_uses_onset_and_pitch_thresholds(self, score_motifs, _span):
        score_motifs.return_value = [
            (100, 500, 0.59, 0.80, 0.737, 0.9),
            (200, 600, 0.60, 0.80, 0.740, 0.8),
            (300, 700, 0.60, 0.20, 0.320, 0.8),
        ]

        result = find_repeating_motif(
            _FakeStem(1.0), 100, [0, 1, 2], MotifConfig()
        )

        self.assertEqual(result, (200, 600, 0.740))

    def test_repeating_motifs_filter_by_absolute_active_ratio_and_report_components(self):
        librosa = types.ModuleType("librosa")
        onset_inputs = []

        def onset_strength(**kwargs):
            onset_inputs.append(kwargs)
            return np.ones(44)

        librosa.feature = types.SimpleNamespace(
            rms=lambda **_kwargs: np.array(
                [[0.02] * 7 + [0.001] * 3 + [0.02] * 8 + [0.001] * 2 + [0.02] * 20]
            ),
            chroma_cens=lambda **_kwargs: np.ones((12, 40)),
        )
        librosa.onset = types.SimpleNamespace(onset_strength=onset_strength)
        librosa.time_to_frames = lambda seconds, **_kwargs: int(seconds * 5)

        def amplitude_to_db(values, ref):
            self.assertEqual(ref, 1.0)
            return 20.0 * np.log10(np.maximum(values, 1e-8))

        librosa.amplitude_to_db = amplitude_to_db

        sklearn = types.ModuleType("sklearn")
        metrics = types.ModuleType("sklearn.metrics")
        pairwise = types.ModuleType("sklearn.metrics.pairwise")
        cosine_shapes = []

        def cosine_similarity(first, second):
            cosine_shapes.append((first.shape[1], second.shape[1]))
            return np.array([[0.6 if first.shape[1] <= 256 else 0.8]])

        pairwise.cosine_similarity = cosine_similarity
        stem_values = np.arange(800, dtype=np.float64)
        stem = _FakeAudioStem(stem_values)

        with patch.dict(
            sys.modules,
            {
                "librosa": librosa,
                "sklearn": sklearn,
                "sklearn.metrics": metrics,
                "sklearn.metrics.pairwise": pairwise,
            },
        ):
            result = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )
            lower_presence_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, min_presence=0.70),
            )

            pairwise.cosine_similarity = lambda first, _second: np.array(
                [[0.5 if first.shape[1] <= 256 else 0.8]]
            )
            difference_below_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )

            pairwise.cosine_similarity = lambda first, _second: np.array(
                [[0.4 if first.shape[1] <= 256 else 0.8]]
            )
            difference_at_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )

            pairwise.cosine_similarity = lambda first, _second: np.array(
                [[0.39 if first.shape[1] <= 256 else 0.8]]
            )
            difference_above_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )
            wider_difference_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, max_similarity_difference=0.42),
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(len(lower_presence_threshold), 4)
        self.assertEqual([row[0] for row in result], [200, 400, 600])
        self.assertEqual([row[5] for row in result], [0.8, 1.0, 1.0])
        for _, _, onset_similarity, chroma_similarity, similarity, _ in result:
            self.assertAlmostEqual(onset_similarity, 0.6)
            self.assertAlmostEqual(chroma_similarity, 0.8)
            self.assertAlmostEqual(similarity, 0.74)
        self.assertEqual(len(difference_below_threshold), 3)
        self.assertEqual(len(difference_at_threshold), 3)
        self.assertEqual(difference_above_threshold, [])
        self.assertEqual(len(wider_difference_threshold), 3)
        self.assertIn((256, 256), cosine_shapes)
        self.assertIn((768, 768), cosine_shapes)
        self.assertEqual(len(onset_inputs), 6)
        for onset_input in onset_inputs:
            self.assertEqual(onset_input["n_fft"], 2048)
            self.assertEqual(len(onset_input["y"]), 2048 + len(stem_values))
            np.testing.assert_array_equal(onset_input["y"][:2048], np.zeros(2048))
            np.testing.assert_array_equal(onset_input["y"][2048:], stem_values)


if __name__ == "__main__":
    unittest.main()
