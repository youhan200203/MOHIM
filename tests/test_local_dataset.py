import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mohim_dali.local_dataset import (
    _artist_title,
    _validate_youtube_candidate,
    ingest_genius_tag_tracks,
    ingest_youtube_tracks,
    load_local_tracks,
)


class LocalDatasetTests(unittest.TestCase):
    def test_artist_title_uses_common_youtube_title(self):
        artist, title = _artist_title({"title": "Example Artist - Example Song (Official Video)"})
        self.assertEqual(artist, "Example Artist")
        self.assertEqual(title, "Example Song")

    def test_ingest_leaves_failed_lyrics_editable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "downloaded.m4a"
            audio_path.touch()
            info = {
                "id": "video123",
                "title": "Artist - Title (Official Audio)",
                "duration": 180,
                "local_audio_path": str(audio_path),
            }
            with patch("mohim_dali.local_dataset._download_audio", return_value=info), patch(
                "mohim_dali.local_dataset._search_genius_lyrics", return_value=("", None)
            ):
                payload = ingest_youtube_tracks(["https://youtu.be/video123"], root)

            self.assertEqual(payload["tracks"][0]["lyrics"], "")
            self.assertEqual(payload["tracks"][0]["lyrics_status"], "manual_required")
            saved = json.loads((root / "tracks.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["metadata"]["num_missing_lyrics"], 1)

    def test_genius_error_does_not_discard_downloaded_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "downloaded.m4a"
            audio_path.touch()
            info = {
                "id": "video456",
                "title": "Artist - Title",
                "local_audio_path": str(audio_path),
            }
            with patch("mohim_dali.local_dataset._download_audio", return_value=info), patch(
                "mohim_dali.local_dataset._search_genius_lyrics", side_effect=RuntimeError("lookup failed")
            ):
                payload = ingest_youtube_tracks(
                    ["https://youtu.be/video456"], root, genius_access_token="token"
                )

            self.assertEqual(len(payload["tracks"]), 1)
            self.assertEqual(payload["tracks"][0]["lyrics_status"], "manual_required")
            self.assertEqual(payload["tracks"][0]["lyrics_error"], "RuntimeError: lookup failed")

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

    def test_youtube_candidate_accepts_matching_official_audio(self):
        accepted, reason = _validate_youtube_candidate(
            "Example Artist",
            "Example Song",
            {
                "title": "Example Artist - Example Song (Official Audio)",
                "uploader": "Example Artist - Topic",
                "duration": 200,
            },
        )

        self.assertTrue(accepted, reason)

    def test_ingest_genius_tag_pairs_lyrics_and_youtube_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "downloaded.m4a"
            audio_path.touch()
            candidate = {
                "artist": "Example Artist",
                "title": "Example Song",
                "genius_url": "https://genius.com/example-song-lyrics",
            }
            search_info = {
                "id": "video789",
                "title": "Example Artist - Example Song (Official Audio)",
                "uploader": "Example Artist - Topic",
                "duration": 180,
                "webpage_url": "https://www.youtube.com/watch?v=video789",
            }
            downloaded_info = dict(search_info, local_audio_path=str(audio_path))
            with patch("mohim_dali.local_dataset._genius_client", return_value=object()), patch(
                "mohim_dali.local_dataset._iter_genius_tag_candidates",
                return_value=iter([candidate]),
            ), patch(
                "mohim_dali.local_dataset._genius_lyrics_from_url",
                return_value="[Verse]\nPaired lyrics",
            ), patch(
                "mohim_dali.local_dataset._search_youtube", return_value=search_info
            ), patch(
                "mohim_dali.local_dataset._download_audio", return_value=downloaded_info
            ):
                payload = ingest_genius_tag_tracks(root, max_tracks=1)

            track = payload["tracks"][0]
            self.assertEqual(track["lyrics"], "[Verse]\nPaired lyrics")
            self.assertEqual(track["lyrics_source_url"], candidate["genius_url"])
            self.assertEqual(track["source_url"], search_info["webpage_url"])
            self.assertEqual(track["audio_path"], str(audio_path))
            self.assertEqual(track["genres"], ["Pop"])

    def test_ingest_genius_tag_resumes_completed_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "audio" / "video789.m4a"
            audio_path.parent.mkdir()
            audio_path.touch()
            genius_url = "https://genius.com/example-song-lyrics"
            (root / "tracks.json").write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "id": "video789",
                                "source_url": "https://www.youtube.com/watch?v=video789",
                                "audio_path": str(audio_path),
                                "artist": "Example Artist",
                                "title": "Example Song",
                                "genres": ["Pop"],
                                "language": "English",
                                "lyrics": "Existing lyrics",
                                "lyrics_source_url": genius_url,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate = {
                "artist": "Example Artist",
                "title": "Example Song",
                "genius_url": genius_url,
            }
            with patch("mohim_dali.local_dataset._genius_client", return_value=object()), patch(
                "mohim_dali.local_dataset._iter_genius_tag_candidates",
                return_value=iter([candidate]),
            ), patch("mohim_dali.local_dataset._download_audio") as download:
                payload = ingest_genius_tag_tracks(root, max_tracks=1)

            download.assert_not_called()
            self.assertEqual(len(payload["tracks"]), 1)

    def test_load_local_tracks_reads_manually_pasted_lyrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "song.m4a"
            audio_path.touch()
            manifest_path = root / "tracks.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "id": "song-1",
                                "source_url": "https://youtu.be/song-1",
                                "audio_path": str(audio_path),
                                "artist": "Artist",
                                "title": "Title",
                                "genres": ["Pop"],
                                "language": "English",
                                "lyrics": "Manually pasted lyrics",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            tracks = load_local_tracks(manifest_path)

            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].dali_id, "song-1")
            self.assertEqual(tracks[0].lyrics, "Manually pasted lyrics")
            self.assertEqual(tracks[0].audio_path, str(audio_path))


if __name__ == "__main__":
    unittest.main()
