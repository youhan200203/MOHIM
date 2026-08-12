import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mohim.local_dataset import (
    _validate_youtube_candidate,
    ingest_genius_seed,
    load_local_tracks,
)


class LocalDatasetTests(unittest.TestCase):
    def test_youtube_candidate_rejects_alternate_versions(self):
        accepted, reason = _validate_youtube_candidate(
            "Example Artist",
            "Example Song",
            {
                "title": "Example Artist - Example Song (Live Cover)",
                "uploader": "Example Artist",
                "duration": 200,
            },
        )
        self.assertFalse(accepted)
        self.assertIn("excluded term", reason)

    def test_seed_ingest_tries_next_downloadable_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed.json"
            seed.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "id": "example-song",
                                "artist": "Example Artist",
                                "title": "Example Song",
                                "lyrics": "Lyrics",
                                "genres": ["Pop"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            search_results = [
                {
                    "id": "unavailable",
                    "title": "Example Artist - Example Song (Official Audio)",
                    "uploader": "Example Artist - Topic",
                    "duration": 180,
                },
                {
                    "id": "available",
                    "title": "Example Artist - Example Song (Official Audio)",
                    "uploader": "Example Artist - Topic",
                    "duration": 180,
                },
            ]
            audio_path = root / "dataset" / "audio" / "available.m4a"
            audio_path.parent.mkdir(parents=True)
            audio_path.touch()
            downloaded = {
                **search_results[1],
                "local_audio_path": str(audio_path),
            }
            with patch(
                "mohim.local_dataset._search_youtube", return_value=search_results
            ), patch(
                "mohim.local_dataset._download_audio",
                side_effect=[RuntimeError("unavailable"), downloaded],
            ):
                payload = ingest_genius_seed(seed, root / "dataset")

            self.assertEqual(len(payload["tracks"]), 1)
            self.assertEqual(payload["tracks"][0]["youtube_id"], "available")
            self.assertEqual(payload["tracks"][0]["lyrics"], "Lyrics")
            self.assertTrue((root / "dataset" / "tracks.json").is_file())

    def test_seed_ingest_resumes_downloaded_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed.json"
            seed.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "id": "example-song",
                                "artist": "Example Artist",
                                "title": "Example Song",
                                "lyrics": "Lyrics",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset = root / "dataset"
            audio_path = dataset / "audio" / "available.m4a"
            audio_path.parent.mkdir(parents=True)
            audio_path.touch()
            (dataset / "tracks.json").write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "track_id": "example-song",
                                "seed_id": "example-song",
                                "audio_path": str(audio_path),
                                "lyrics": "Lyrics",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch("mohim.local_dataset._download_audio") as download:
                payload = ingest_genius_seed(seed, dataset)
            download.assert_not_called()
            self.assertEqual(len(payload["tracks"]), 1)

    def test_load_local_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "tracks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "track_id": "song-1",
                                "artist": "Artist",
                                "title": "Title",
                                "genres": ["Pop"],
                                "language": "English",
                                "lyrics": "Lyrics",
                                "audio_path": "/tmp/song.m4a",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tracks = load_local_tracks(manifest)
            self.assertEqual(tracks[0].track_id, "song-1")
            self.assertEqual(tracks[0].lyrics, "Lyrics")


if __name__ == "__main__":
    unittest.main()
