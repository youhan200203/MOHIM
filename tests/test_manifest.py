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
            for name in ("motif.flac", "guitar.flac", "vocals.flac"):
                (sample / name).touch()
            (sample / "lyrics.txt").write_text("hello world\n", encoding="utf-8")
            metadata = {
                "status": "accepted",
                "track_id": "abc",
                "artist": "Artist",
                "title": "Title",
                "motif_stem": "guitar",
                "motif_seed_file": "motif.flac",
                "motif_target_file": "guitar.flac",
                "vocal_target_file": "vocals.flac",
                "motif_start_sec": 1.0,
                "motif_end_sec": 9.0,
            }
            (sample / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest = build_dual_stream_manifest(root, root / "manifest.json")
            self.assertEqual(manifest["metadata"]["num_samples"], 1)
            self.assertEqual(manifest["samples"][0]["lyrics"], "hello world")
            self.assertTrue(Path(manifest["samples"][0]["motif_seed_audio"]).is_file())


if __name__ == "__main__":
    unittest.main()
