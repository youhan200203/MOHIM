import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from mohim.motif import (
    MotifConfig,
    MotifExtractor,
    _max_shifted_cosine_similarity,
    _pairwise_shifted_cosine_similarity,
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

    def test_onset_similarity_ignores_constant_baseline(self):
        first = np.array([1.0, 4.0, 1.0, 3.0])
        second = first + 10.0

        def cosine_similarity(left, right):
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            score = float(np.dot(left.ravel(), right.ravel()) / denominator) if denominator else 0.0
            return np.array([[score]])

        similarity = _max_shifted_cosine_similarity(
            first,
            second,
            max_shift=0,
            cosine_similarity=cosine_similarity,
        )

        self.assertAlmostEqual(similarity, 1.0)

        raw_similarity = _max_shifted_cosine_similarity(
            first,
            second,
            max_shift=0,
            cosine_similarity=cosine_similarity,
            mean_center=False,
        )
        self.assertLess(raw_similarity, 1.0)

    def test_pairwise_shifted_cosine_matches_scalar_reference(self):
        rng = np.random.default_rng(42)
        first = rng.normal(size=(5, 256))
        second = rng.normal(size=(9, 256))
        first[0] = 1.0
        second[0] = 1.0

        def cosine_similarity(left, right):
            left_norm = np.linalg.norm(left, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right, axis=1, keepdims=True)
            normalized_left = np.divide(
                left,
                left_norm,
                out=np.zeros_like(left),
                where=left_norm != 0,
            )
            normalized_right = np.divide(
                right,
                right_norm,
                out=np.zeros_like(right),
                where=right_norm != 0,
            )
            return normalized_left @ normalized_right.T

        for mean_center in (False, True):
            actual = _pairwise_shifted_cosine_similarity(
                first,
                second,
                max_shift=3,
                cosine_similarity=cosine_similarity,
                mean_center=mean_center,
            )
            expected = np.asarray(
                [
                    [
                        _max_shifted_cosine_similarity(
                            left,
                            right,
                            max_shift=3,
                            cosine_similarity=cosine_similarity,
                            mean_center=mean_center,
                        )
                        for right in second
                    ]
                    for left in first
                ]
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

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

    @patch("mohim.motif.onset_variation", return_value=0.20)
    @patch.object(MotifExtractor, "score_all")
    def test_extract_reuses_scores_and_compares_first_passing_motifs(
        self, score_all, variation
    ):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.59,
                    "chroma_similarity": 0.8,
                    "mean_centered_onset_similarity": 0.8,
                    "mean_centered_chroma_similarity": 0.8,
                    "similarity": 0.8,
                    "pitch_class_span": 0.7,
                },
                {
                    "stem_name": "guitar", "start_sec": 2.0, "end_sec": 6.0,
                    "active_ratio": 0.9, "onset_similarity": 0.61,
                    "chroma_similarity": 0.99,
                    "mean_centered_onset_similarity": 0.5,
                    "mean_centered_chroma_similarity": 0.4,
                    "similarity": 0.47,
                    "pitch_class_span": 0.31,
                },
                {
                    "stem_name": "piano", "start_sec": 3.0, "end_sec": 7.0,
                    "active_ratio": 0.8, "onset_similarity": 0.62,
                    "chroma_similarity": 0.63,
                    "mean_centered_onset_similarity": 0.6,
                    "mean_centered_chroma_similarity": 0.7,
                    "similarity": 0.63,
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
        self.assertAlmostEqual(result["stem_scores"]["piano"]["similarity"], 0.63)
        self.assertAlmostEqual(result["onset_variation"], 0.20)
        variation.assert_called_once()
        self.assertNotIn("dominance", result["stem_scores"]["guitar"])
        self.assertNotIn("total", result["stem_scores"]["guitar"])

    @patch("mohim.motif.onset_variation", return_value=0.20)
    @patch.object(MotifExtractor, "score_all")
    def test_extract_records_stems_without_a_passing_motif(self, score_all, _variation):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.7,
                    "mean_centered_onset_similarity": 0.4,
                    "mean_centered_chroma_similarity": 0.4,
                    "similarity": 0.4,
                    "pitch_class_span": 0.31,
                },
                {
                    "stem_name": "piano", "start_sec": 2.0, "end_sec": 6.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.8,
                    "mean_centered_onset_similarity": 0.7,
                    "mean_centered_chroma_similarity": 0.7,
                    "similarity": 0.7,
                    "pitch_class_span": 0.16666666666666666,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.extract("song.wav", stems, None, 100)

        self.assertFalse(result["stem_scores"]["piano"]["matched"])
        self.assertIsNone(result["stem_scores"]["piano"]["start_sec"])

    @patch("mohim.motif.onset_variation", return_value=0.199)
    @patch.object(MotifExtractor, "score_all")
    def test_extract_rejects_final_candidate_below_onset_variation_threshold(
        self, score_all, variation
    ):
        stem = _FakeStem(0.8)
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.8,
                    "mean_centered_onset_similarity": 0.4,
                    "mean_centered_chroma_similarity": 0.4,
                    "similarity": 0.4,
                    "pitch_class_span": 0.5,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        with self.assertRaisesRegex(ValueError, "onset variation"):
            extractor.extract("song.wav", {"guitar": stem}, None, 100)

        variation.assert_called_once()

    @patch("mohim.motif.onset_variation")
    @patch.object(MotifExtractor, "score_all")
    def test_extract_rejects_final_candidate_below_mean_centered_threshold(
        self, score_all, variation
    ):
        stem = _FakeStem(0.8)
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.8,
                    "mean_centered_onset_similarity": 0.279,
                    "mean_centered_chroma_similarity": 0.8,
                    "similarity": 0.4353,
                    "pitch_class_span": 0.5,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        with self.assertRaisesRegex(ValueError, "mean-centered"):
            extractor.extract("song.wav", {"guitar": stem}, None, 100)

        variation.assert_not_called()

    @patch("mohim.motif.onset_variation")
    @patch.object(MotifExtractor, "score_all")
    def test_extract_uses_validated_final_audio_onset_variation(
        self, score_all, variation
    ):
        stem = _FakeStem(0.8)
        score_all.return_value = {
            "candidates": [
                {
                    "stem_name": "guitar", "start_sec": 1.0, "end_sec": 5.0,
                    "active_ratio": 0.9, "onset_similarity": 0.60,
                    "chroma_similarity": 0.8,
                    "mean_centered_onset_similarity": 0.4,
                    "mean_centered_chroma_similarity": 0.4,
                    "similarity": 0.4,
                    "pitch_class_span": 0.5,
                },
            ],
            "melodic_accompaniment": _FakeStem(1.0),
        }
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        with self.assertRaisesRegex(ValueError, "onset variation"):
            extractor.extract(
                "song.wav",
                {"guitar": stem},
                None,
                100,
                validated_onset_variation=0.199,
            )

        variation.assert_not_called()

    @patch("mohim.motif.pitch_class_span", return_value=0.5)
    @patch("mohim.motif.score_repeating_motifs")
    def test_score_all_keeps_every_filtered_candidate(self, score_motifs, span):
        guitar = _FakeStem(0.8)
        piano = _FakeStem(0.2)
        stems = {"guitar": guitar, "piano": piano}
        score_motifs.side_effect = lambda stem, *_args: (
            [
                (100, 500, 0.10, 0.24, 0.11, 0.21, 0.14, 0.8),
                (200, 600, 0.60, 0.74, 0.70, 0.60, 0.67, 0.9),
            ]
            if stem is guitar
            else [(300, 700, 0.30, 0.44, 0.40, 0.30, 0.37, 0.8)]
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.score_all("song.wav", stems, 100)

        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([row["candidate_index"] for row in result["candidates"]], [0, 1, 0])
        self.assertEqual([row["similarity"] for row in result["candidates"]], [0.14, 0.67, 0.37])
        self.assertEqual(
            [row["pitch_class_span"] for row in result["candidates"]],
            [None, 0.5, None],
        )
        self.assertEqual(span.call_count, 1)
        self.assertNotIn("dominance", result["candidates"][0])
        self.assertNotIn("total", result["candidates"][0])
        self.assertAlmostEqual(result["melodic_accompaniment"].rms, 1.0)

    @patch("mohim.motif._score_stem_candidates")
    def test_score_many_preserves_track_and_stem_order(self, score_stem):
        score_stem.side_effect = lambda name, *_args: [{"stem_name": name}]
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))
        tracks = [
            ("first.wav", {"guitar": _FakeStem(0.8), "bass": _FakeStem(0.2)}, 100),
            ("second.wav", {"piano": _FakeStem(0.6), "other": _FakeStem(0.4)}, 200),
        ]

        results = extractor.score_many(tracks, max_workers=12)

        self.assertEqual(
            [[row["stem_name"] for row in result["candidates"]] for result in results],
            [["guitar", "bass"], ["piano", "other"]],
        )
        self.assertEqual([result["sample_rate"] for result in results], [100, 200])
        self.assertEqual(score_stem.call_count, 4)

    @patch("mohim.motif.pitch_class_span", return_value=0.5)
    @patch("mohim.motif.score_repeating_motifs")
    def test_score_many_matches_sequential_scores(self, score_motifs, _span):
        score_motifs.side_effect = lambda stem, *_args: [
            (100, 500, 0.61, 0.70, 0.40, 0.50, stem.rms, 0.9)
        ]
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))
        tracks = [
            ("first.wav", {"guitar": _FakeStem(0.8), "bass": _FakeStem(0.2)}, 100),
            ("second.wav", {"piano": _FakeStem(0.6), "other": _FakeStem(0.4)}, 200),
        ]
        sequential = [
            extractor.score_all(audio_path, stems, sample_rate)
            for audio_path, stems, sample_rate in tracks
        ]

        parallel = extractor.score_many(tracks, max_workers=12)

        self.assertEqual(
            [result["candidates"] for result in parallel],
            [result["candidates"] for result in sequential],
        )

    @patch("mohim.motif.pitch_class_span", side_effect=[0.8, 0.25])
    @patch("mohim.motif.score_repeating_motifs")
    def test_find_repeating_motif_uses_onset_and_pitch_thresholds(self, score_motifs, _span):
        score_motifs.return_value = [
            (100, 500, 0.59, 0.80, 0.50, 0.60, 0.53, 0.9),
            (200, 600, 0.60, 0.80, 0.70, 0.60, 0.67, 0.8),
            (300, 700, 0.60, 0.20, 0.20, 0.30, 0.23, 0.8),
        ]

        result = find_repeating_motif(
            _FakeStem(1.0), 100, [0, 1, 2], MotifConfig()
        )

        self.assertEqual(result, (200, 600, 0.67))

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
        chroma_cosine_inputs = []

        def cosine_similarity(first, second):
            cosine_shapes.append((first.shape[1], second.shape[1]))
            if first.shape[1] == 768:
                chroma_cosine_inputs.append((first.copy(), second.copy()))
            value = 0.6 if first.shape[1] <= 256 else 0.8
            return np.full((first.shape[0], second.shape[0]), value)

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
            stricter_presence_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, min_presence=0.80),
            )

            pairwise.cosine_similarity = lambda first, second: np.full(
                (first.shape[0], second.shape[0]),
                0.5 if first.shape[1] <= 256 else 0.8,
            )
            difference_below_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, min_presence=0.80),
            )

            pairwise.cosine_similarity = lambda first, second: np.full(
                (first.shape[0], second.shape[0]),
                0.4 if first.shape[1] <= 256 else 0.8,
            )
            difference_at_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, min_presence=0.80),
            )

            pairwise.cosine_similarity = lambda first, second: np.full(
                (first.shape[0], second.shape[0]),
                0.39 if first.shape[1] <= 256 else 0.8,
            )
            difference_above_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1, min_presence=0.80),
            )
            wider_difference_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(
                    bars=1,
                    min_presence=0.80,
                    max_similarity_difference=0.42,
                ),
            )

        self.assertEqual(len(result), 4)
        self.assertEqual(len(stricter_presence_threshold), 3)
        self.assertEqual([row[0] for row in result], [0, 200, 400, 600])
        self.assertEqual([row[7] for row in result], [0.7, 0.8, 1.0, 1.0])
        for (
            _,
            _,
            onset_similarity,
            chroma_similarity,
            mean_centered_onset_similarity,
            mean_centered_chroma_similarity,
            similarity,
            _,
        ) in result:
            self.assertAlmostEqual(onset_similarity, 0.6)
            self.assertAlmostEqual(chroma_similarity, 0.8)
            self.assertAlmostEqual(mean_centered_onset_similarity, 0.6)
            self.assertAlmostEqual(mean_centered_chroma_similarity, 0.8)
            self.assertAlmostEqual(similarity, 0.66)
        self.assertEqual(len(difference_below_threshold), 3)
        self.assertEqual(len(difference_at_threshold), 3)
        self.assertEqual(difference_above_threshold, [])
        self.assertEqual(len(wider_difference_threshold), 3)
        self.assertIn((256, 256), cosine_shapes)
        self.assertIn((768, 768), cosine_shapes)
        self.assertTrue(chroma_cosine_inputs)
        chroma_means = [
            (
                first.reshape(-1, 12, 64).mean(axis=1),
                second.reshape(-1, 12, 64).mean(axis=1),
            )
            for first, second in chroma_cosine_inputs
        ]
        self.assertTrue(
            any(not np.allclose(first_mean, 0.0) for first_mean, _ in chroma_means)
        )
        self.assertTrue(
            any(
                np.allclose(first_mean, 0.0) and np.allclose(second_mean, 0.0)
                for first_mean, second_mean in chroma_means
            )
        )
        self.assertEqual(len(onset_inputs), 6)
        for onset_input in onset_inputs:
            self.assertEqual(onset_input["n_fft"], 2048)
            self.assertEqual(len(onset_input["y"]), 2048 + len(stem_values))
            np.testing.assert_array_equal(onset_input["y"][:2048], np.zeros(2048))
            np.testing.assert_array_equal(onset_input["y"][2048:], stem_values)


if __name__ == "__main__":
    unittest.main()
