import json
import tempfile
import unittest
from pathlib import Path

from mohim.manifest import build_dual_stream_manifest


class ManifestTests(unittest.TestCase):
    def test_build_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "abc"
            sample.mkdir()
            for name in (
                "motif.flac", "accompaniment.flac", "vocals.flac", "bass.flac", "other.flac"
            ):
                (sample / name).touch()
            (sample / "lyrics.txt").write_text("hello world\n", encoding="utf-8")
            metadata = {
                "schema_version": 4,
                "status": "accepted",
                "track_id": "abc",
                "artist": "Artist",
                "title": "Title",
                "motif_stem": "guitar",
                "motif_onset_variation": 0.3,
                "motif_seed_file": "motif.flac",
                "accompaniment_target_file": "accompaniment.flac",
                "vocal_target_file": "vocals.flac",
                "stem_files": {
                    "vocals": "vocals.flac",
                    "bass": "bass.flac",
                    "other": "other.flac",
                },
                "motif_start_sec": 1.0,
                "motif_end_sec": 9.0,
            }
            (sample / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest = build_dual_stream_manifest(root, root / "manifest.json")
            self.assertEqual(manifest["metadata"]["num_samples"], 1)
            self.assertEqual(manifest["samples"][0]["lyrics"], "hello world")
            self.assertTrue(Path(manifest["samples"][0]["motif_seed_audio"]).is_file())
            self.assertTrue(Path(manifest["samples"][0]["accompaniment_target_audio"]).is_file())
            self.assertNotIn("motif_target_audio", manifest["samples"][0])


if __name__ == "__main__":
    unittest.main()
