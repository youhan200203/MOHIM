import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from mohim.motif import (
    MotifConfig,
    MotifExtractor,
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
            [(100, 500, 0.10, 0.24, 0.20, 0.8), (200, 600, 0.60, 0.74, 0.70, 0.9)]
            if stem is guitar
            else [(300, 700, 0.30, 0.44, 0.40, 0.7)]
        )
        extractor = MotifExtractor(lambda _path: ([], [0, 1, 2]))

        result = extractor.score_all("song.wav", stems, 100)

        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([row["similarity"] for row in result["candidates"]], [0.2, 0.7, 0.4])
        self.assertEqual(
            [row["onset_similarity"] for row in result["candidates"]], [0.1, 0.6, 0.3]
        )
        self.assertEqual([row["active_ratio"] for row in result["candidates"]], [0.8, 0.9, 0.7])
        self.assertAlmostEqual(result["melodic_accompaniment"].rms, 1.0)

    def test_repeating_motifs_filter_by_absolute_active_ratio_and_report_components(self):
        librosa = types.ModuleType("librosa")
        librosa.feature = types.SimpleNamespace(
            rms=lambda **_kwargs: np.array(
                [[0.02] * 6 + [0.001] * 4 + [0.02] * 7 + [0.001] * 3 + [0.02] * 20]
            ),
            chroma_cens=lambda **_kwargs: np.ones((12, 40)),
        )
        librosa.onset = types.SimpleNamespace(onset_strength=lambda **_kwargs: np.ones(40))
        librosa.time_to_frames = lambda seconds, **_kwargs: int(seconds * 5)

        def amplitude_to_db(values, ref):
            self.assertEqual(ref, 1.0)
            return 20.0 * np.log10(np.maximum(values, 1e-8))

        librosa.amplitude_to_db = amplitude_to_db

        sklearn = types.ModuleType("sklearn")
        metrics = types.ModuleType("sklearn.metrics")
        pairwise = types.ModuleType("sklearn.metrics.pairwise")
        pairwise.cosine_similarity = lambda first, _second: np.array(
            [[0.6 if first.shape[1] == 64 else 0.8]]
        )
        stem = _FakeAudioStem(np.zeros(800))

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

            pairwise.cosine_similarity = lambda first, _second: np.array(
                [[0.5 if first.shape[1] == 64 else 0.8]]
            )
            difference_below_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )

            pairwise.cosine_similarity = lambda first, _second: np.array(
                [[0.4 if first.shape[1] == 64 else 0.8]]
            )
            difference_at_threshold = score_repeating_motifs(
                stem,
                sample_rate=100,
                downbeats=[0.0, 2.0, 4.0, 6.0],
                config=MotifConfig(bars=1),
            )

        self.assertEqual(len(result), 3)
        self.assertEqual([row[0] for row in result], [200, 400, 600])
        self.assertEqual([row[5] for row in result], [0.7, 1.0, 1.0])
        for _, _, onset_similarity, chroma_similarity, similarity, _ in result:
            self.assertAlmostEqual(onset_similarity, 0.6)
            self.assertAlmostEqual(chroma_similarity, 0.8)
            self.assertAlmostEqual(similarity, 0.74)
        self.assertEqual(len(difference_below_threshold), 3)
        self.assertEqual(difference_at_threshold, [])


if __name__ == "__main__":
    unittest.main()
