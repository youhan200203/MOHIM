import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # local lightweight test environments may omit audio deps
    torch = None

from mohim.separator import StemSeparator


@unittest.skipIf(torch is None, "torch is not installed")
class StemSeparatorBatchTests(unittest.TestCase):
    def test_separate_many_pads_one_batch_and_restores_each_length(self):
        calls = []
        mixtures = {
            "short.wav": torch.ones((2, 3)),
            "long.wav": torch.full((2, 5), 2.0),
        }
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.load = lambda path: (mixtures[path].clone(), 44_100)
        torchaudio.functional = types.SimpleNamespace(resample=lambda wav, *_args: wav)

        demucs = types.ModuleType("demucs")
        demucs_apply = types.ModuleType("demucs.apply")

        def apply_model(model, batch, *, device, shifts):
            calls.append((model, batch.shape, device, shifts))
            output = torch.zeros((batch.shape[0], 2, 2, batch.shape[-1]))
            output[0, 0] = 1.0
            output[0, 1] = 2.0
            output[1, 0] = 3.0
            output[1, 1] = 4.0
            return output

        demucs_apply.apply_model = apply_model
        model = types.SimpleNamespace(samplerate=44_100, sources=("vocals", "other"))
        separator = StemSeparator(device="cpu")
        separator._model = model

        with patch.dict(
            sys.modules,
            {"torchaudio": torchaudio, "demucs": demucs, "demucs.apply": demucs_apply},
        ):
            results = separator.separate_many(["short.wav", "long.wav"])

        self.assertEqual(calls, [(model, torch.Size([2, 2, 5]), "cpu", 0)])
        self.assertEqual(results[0][0]["vocals"].shape, (2, 3))
        self.assertEqual(results[1][0]["vocals"].shape, (2, 5))
        self.assertTrue(torch.all(results[0][0]["other"] == 2.0))
        self.assertTrue(torch.all(results[1][0]["other"] == 4.0))
        self.assertEqual(results[0][2].shape, (2, 3))
        self.assertEqual(results[1][2].shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
