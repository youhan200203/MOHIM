import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # local lightweight test environments may omit training deps
    torch = None

from mohim.dataset import (
    DatasetBuilder,
    _apply_common_headroom,
    _sum_accompaniment,
    _track_directory_name,
)
from mohim.local_dataset import LocalTrack


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


def _track() -> LocalTrack:
    return LocalTrack(
        track_id="shape-of-you-id",
        artist="Ed Sheeran",
        title="Shape of You",
        genres=("Pop",),
        language="English",
        lyrics="Lyrics",
        duration_seconds=234.0,
        audio_url=None,
        audio_path=None,
        entry={},
    )


class DatasetBuilderTests(unittest.TestCase):
    def test_track_directory_uses_artist_and_title(self):
        track = _track()

        self.assertEqual(_track_directory_name(track), "Ed Sheeran - Shape of You")

    def test_resume_reads_title_directory_and_preserves_track_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "Ed Sheeran - Shape of You"
            sample_dir.mkdir()
            for filename in ("motif.flac", "accompaniment.flac", "vocals.flac", "other.flac", "lyrics.txt"):
                (sample_dir / filename).touch()
            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "status": "accepted",
                        "track_id": "shape-of-you-id",
                        "motif_stem": "other",
                        "motif_seed_file": "motif.flac",
                        "accompaniment_target_file": "accompaniment.flac",
                        "vocal_target_file": "vocals.flac",
                        "stem_files": {"vocals": "vocals.flac", "other": "other.flac"},
                    }
                ),
                encoding="utf-8",
            )
            builder = DatasetBuilder(
                audio_dir=root,
                output_dir=root,
                separator=object(),
                motif_extractor=object(),
            )

            result = builder.process_track(_track(), {})

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.track_id, "shape-of-you-id")
        self.assertEqual(Path(result.output_dir).name, "Ed Sheeran - Shape of You")


if __name__ == "__main__":
    unittest.main()
