import unittest

try:
    import torch
except ImportError:  # local lightweight test environments may omit training deps
    torch = None

from mohim.dataset import _apply_common_headroom, _sum_accompaniment


@unittest.skipIf(torch is None, "torch is not installed")
class DatasetAudioTargetTests(unittest.TestCase):
    def test_accompaniment_sums_every_non_vocal_stem(self):
        stems = {
            "vocals": torch.full((2, 4), 0.25),
            "drums": torch.full((2, 4), 0.10),
            "bass": torch.full((2, 4), 0.20),
            "other": torch.full((2, 4), -0.05),
        }
        accompaniment = _sum_accompaniment(stems)
        self.assertTrue(torch.allclose(accompaniment, torch.full((2, 4), 0.25)))

    def test_headroom_uses_one_gain_for_motif_and_both_targets(self):
        motif = torch.tensor([[0.5]])
        stems = {
            "vocals": torch.tensor([[0.7]]),
            "drums": torch.tensor([[0.5]]),
            "bass": torch.tensor([[0.3]]),
        }
        scaled_motif, scaled_accompaniment, scaled_stems, gain = _apply_common_headroom(
            motif, stems
        )
        self.assertAlmostEqual(gain, 0.999 / 1.5)
        self.assertTrue(torch.allclose(scaled_motif, motif * gain))
        self.assertTrue(torch.allclose(scaled_accompaniment, torch.tensor([[0.8]]) * gain))
        for name, audio in stems.items():
            self.assertTrue(torch.allclose(scaled_stems[name], audio * gain))
        reconstructed = scaled_accompaniment + scaled_stems["vocals"]
        self.assertLessEqual(float(reconstructed.abs().amax()), 0.999001)


if __name__ == "__main__":
    unittest.main()
