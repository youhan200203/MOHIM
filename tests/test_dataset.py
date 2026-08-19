import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:  # local lightweight test environments may omit training deps
    torch = None

from mohim.dataset import (
    DatasetBuilder,
    LocalTrack,
    _apply_common_headroom,
    _sum_accompaniment,
    _track_directory_name,
    load_local_tracks,
)


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
                        "motif_onset_variation": 0.3,
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

    def test_process_track_reuses_precomputed_separation_and_motif_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "song.m4a"
            audio_path.touch()
            separator = Mock()
            motif_extractor = Mock()
            motif_extractor.extract.return_value = {
                "audio": object(),
                "stem_name": "other",
                "stem_scores": {},
                "similarity": 0.7,
                "onset_variation": 0.3,
                "start_sec": 1.0,
                "end_sec": 5.0,
            }
            builder = DatasetBuilder(
                audio_dir=root,
                output_dir=root / "output",
                separator=separator,
                motif_extractor=motif_extractor,
            )
            stems = {"vocals": object(), "other": object()}
            mixture = object()
            motif_scores = {"candidates": [], "melodic_accompaniment": object()}

            with patch(
                "mohim.dataset._apply_common_headroom",
                return_value=(object(), object(), stems, 1.0),
            ), patch("mohim.dataset.save_audio"):
                result = builder.process_track(
                    _track(),
                    {"shape-of-you-id": audio_path},
                    separation_result=(stems, 44_100, mixture),
                    motif_scores=motif_scores,
                    validated_onset_variation=0.2,
                )

        self.assertEqual(result.status, "accepted")
        separator.separate.assert_not_called()
        motif_extractor.extract.assert_called_once_with(
            audio_path,
            stems,
            mixture,
            44_100,
            scored_result=motif_scores,
            validated_onset_variation=0.2,
        )

    def test_resume_does_not_reuse_onset_variation_below_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "Ed Sheeran - Shape of You"
            sample_dir.mkdir()
            for filename in (
                "motif.flac",
                "accompaniment.flac",
                "vocals.flac",
                "other.flac",
                "lyrics.txt",
            ):
                (sample_dir / filename).touch()
            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "status": "accepted",
                        "track_id": "shape-of-you-id",
                        "motif_stem": "other",
                        "motif_onset_variation": 0.199,
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

            result = builder._accepted_result(_track(), sample_dir)

        self.assertIsNone(result)

    def test_record_rejection_invalidates_cached_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "Ed Sheeran - Shape of You"
            sample_dir.mkdir()
            metadata_path = sample_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps({"schema_version": 4, "status": "accepted"}),
                encoding="utf-8",
            )
            builder = DatasetBuilder(
                audio_dir=root,
                output_dir=root,
                separator=object(),
                motif_extractor=object(),
            )

            result = builder.record_rejection(
                _track(), "final_candidate_onset_variation_below_threshold"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "rejected")
        self.assertEqual(metadata["status"], "rejected")
        self.assertEqual(
            metadata["rejection_reason"],
            "final_candidate_onset_variation_below_threshold",
        )

    def test_reprocess_rejection_invalidates_previous_accepted_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "song.m4a"
            audio_path.touch()
            sample_dir = root / "output" / "Ed Sheeran - Shape of You"
            sample_dir.mkdir(parents=True)
            metadata_path = sample_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "status": "accepted",
                        "track_id": "shape-of-you-id",
                    }
                ),
                encoding="utf-8",
            )
            motif_extractor = Mock()
            motif_extractor.extract.side_effect = ValueError(
                "Selected motif onset variation 0.199 is below 0.200."
            )
            builder = DatasetBuilder(
                audio_dir=root,
                output_dir=root / "output",
                separator=object(),
                motif_extractor=motif_extractor,
                resume=False,
            )

            result = builder.process_track(
                _track(),
                {"shape-of-you-id": audio_path},
                separation_result=({"vocals": object(), "other": object()}, 44_100, object()),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "rejected")
        self.assertEqual(metadata["status"], "rejected")
        self.assertIn("onset variation", metadata["rejection_reason"])

    def test_load_local_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "tracks.json"
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
