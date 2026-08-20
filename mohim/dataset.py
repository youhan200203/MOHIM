"""Resumable MOHIM audio dataset builder."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .motif import MotifExtractor
from .separator import StemSeparator, save_audio


@dataclass(frozen=True)
class LocalTrack:
    track_id: str
    artist: str
    title: str
    genres: tuple[str, ...]
    language: str
    lyrics: str
    duration_seconds: float | None
    audio_url: str | None
    audio_path: str | None
    entry: Any


def load_local_tracks(
    manifest_path: str | Path,
    *,
    require_lyrics: bool = True,
    max_duration_seconds: float | None = None,
) -> list[LocalTrack]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tracks"), list):
        raise ValueError(f"JSON must contain a 'tracks' list: {path}")

    tracks: list[LocalTrack] = []
    for item in payload["tracks"]:
        duration = item.get("duration_seconds")
        if (
            max_duration_seconds is not None
            and duration is not None
            and float(duration) > max_duration_seconds
        ):
            continue
        lyrics = str(item.get("lyrics") or "").strip()
        if require_lyrics and not lyrics:
            continue
        genres = item.get("genres") or ()
        if isinstance(genres, str):
            genres = [genres]
        tracks.append(
            LocalTrack(
                track_id=str(item.get("track_id") or item.get("seed_id") or item.get("id")),
                artist=str(item.get("artist") or "").strip(),
                title=str(item.get("title") or "").strip(),
                genres=tuple(str(value).strip() for value in genres if str(value).strip()),
                language=str(item.get("language") or "").strip(),
                lyrics=lyrics,
                duration_seconds=float(duration) if duration is not None else None,
                audio_url=str(item.get("source_url") or "") or None,
                audio_path=str(item.get("audio_path") or "") or None,
                entry=item,
            )
        )
    return tracks


def index_audio_files(audio_dir: str | Path) -> dict[str, Path]:
    root = Path(audio_dir).expanduser().resolve()
    if not root.is_dir():
        return {}
    extensions = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}
    return {
        path.stem: path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    }


def resolve_audio_path(track: LocalTrack, audio_index: Mapping[str, Path]) -> Path | None:
    if track.audio_path:
        direct = Path(track.audio_path).expanduser()
        if direct.is_file():
            return direct.resolve()
    youtube_id = str(track.entry.get("youtube_id") or "") if isinstance(track.entry, Mapping) else ""
    return audio_index.get(youtube_id) or audio_index.get(track.track_id)


def _track_directory_name(track: LocalTrack) -> str:
    label = f"{track.artist} - {track.title}" if track.artist else track.title
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", label)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:120] or track.track_id


def _sum_accompaniment(stems: dict[str, Any]) -> Any:
    """Sum every separated source except vocals without per-stem normalization."""
    import torch

    non_vocal = [audio for name, audio in stems.items() if name != "vocals"]
    if not non_vocal:
        raise ValueError("Separator did not return any non-vocal stems.")
    reference_shape = non_vocal[0].shape
    if any(audio.shape != reference_shape for audio in non_vocal):
        raise ValueError("Separated stems are not time-aligned.")
    return torch.stack(non_vocal, dim=0).sum(dim=0)


def _apply_common_headroom(
    motif: Any,
    stems: dict[str, Any],
    *,
    peak_limit: float = 0.999,
) -> tuple[Any, Any, dict[str, Any], float]:
    """Prevent PCM clipping with one gain shared by motif and every stem."""
    if "vocals" not in stems:
        raise ValueError("Separator did not return a vocals stem.")
    accompaniment = _sum_accompaniment(stems)
    peaks = [motif.abs().amax(), (accompaniment + stems["vocals"]).abs().amax()]
    peaks.extend(audio.abs().amax() for audio in stems.values())
    peak = max(float(value) for value in peaks)
    gain = min(1.0, peak_limit / peak) if peak > 0.0 else 1.0
    scaled_stems = {name: audio * gain for name, audio in stems.items()}
    return motif * gain, _sum_accompaniment(scaled_stems), scaled_stems, gain


@dataclass(frozen=True)
class BuildResult:
    track_id: str
    status: str
    reason: str | None
    output_dir: str | None
    motif_stem: str | None


class DatasetBuilder:
    def __init__(
        self,
        *,
        audio_dir: str | Path,
        output_dir: str | Path,
        separator: StemSeparator,
        motif_extractor: MotifExtractor,
        audio_format: str = "flac",
        resume: bool = True,
    ) -> None:
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.separator = separator
        self.motif_extractor = motif_extractor
        self.audio_format = audio_format.lower()
        if self.audio_format not in {"flac", "wav"}:
            raise ValueError("audio_format must be either 'flac' or 'wav'.")
        self.resume = resume

    def _rejected_result(
        self,
        track: LocalTrack,
        sample_dir: Path,
        reason: str,
    ) -> BuildResult:
        metadata_path = sample_dir / "metadata.json"
        if sample_dir.is_dir():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            metadata.update(
                {
                    "schema_version": 4,
                    "status": "rejected",
                    "track_id": track.track_id,
                    "artist": track.artist,
                    "title": track.title,
                    "rejection_reason": reason,
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return BuildResult(track.track_id, "rejected", reason, None, None)

    def _accepted_result(
        self,
        track: LocalTrack,
        sample_dir: Path,
        *,
        validated_onset_variation: float | None = None,
    ) -> BuildResult | None:
        metadata_path = sample_dir / "metadata.json"
        if not self.resume or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        stem_files = metadata.get("stem_files")
        required_names = (
            metadata.get("motif_seed_file"),
            metadata.get("accompaniment_target_file"),
            metadata.get("vocal_target_file"),
        )
        debug_stem_names = tuple(stem_files.values()) if isinstance(stem_files, dict) else ()
        required = [sample_dir / name for name in (*required_names, *debug_stem_names) if name]
        required.append(sample_dir / "lyrics.txt")
        stored_onset_variation = metadata.get("motif_onset_variation")
        onset_variation_threshold = float(
            getattr(
                getattr(self.motif_extractor, "config", None),
                "onset_variation_threshold",
                0.20,
            )
        )
        if (
            metadata.get("schema_version") == 4
            and metadata.get("status") == "accepted"
            and all(required_names)
            and isinstance(stem_files, dict)
            and stem_files.get("vocals") == metadata.get("vocal_target_file")
            and isinstance(stored_onset_variation, (int, float))
            and float(stored_onset_variation) >= onset_variation_threshold
            and (
                validated_onset_variation is None
                or float(stored_onset_variation) == float(validated_onset_variation)
            )
            and "drums" not in stem_files
            and all(path.is_file() for path in required)
        ):
            return BuildResult(track.track_id, "skipped", "already_processed", str(sample_dir), metadata.get("motif_stem"))
        return None

    def record_rejection(self, track: LocalTrack, reason: str) -> BuildResult:
        """Record a rejection decided by the canonical final-motif validation."""

        sample_dir = self.output_dir / _track_directory_name(track)
        return self._rejected_result(track, sample_dir, reason)

    def process_track(
        self,
        track: LocalTrack,
        audio_index: dict[str, Path],
        *,
        separation_result: tuple[dict[str, Any], int, Any] | None = None,
        motif_scores: dict[str, Any] | None = None,
        validated_onset_variation: float | None = None,
        selected_candidate: dict[str, Any] | None = None,
    ) -> BuildResult:
        sample_dir = self.output_dir / _track_directory_name(track)
        cached = self._accepted_result(
            track,
            sample_dir,
            validated_onset_variation=validated_onset_variation,
        )
        if cached:
            return cached

        audio_path = resolve_audio_path(track, audio_index)
        if audio_path is None:
            return self._rejected_result(track, sample_dir, "audio_not_found")
        if not track.lyrics.strip():
            return self._rejected_result(track, sample_dir, "lyrics_missing")

        try:
            if separation_result is None:
                stems, sample_rate, mixture = self.separator.separate(audio_path)
            else:
                stems, sample_rate, mixture = separation_result
            if "vocals" not in stems:
                raise ValueError("Separator did not return a vocals stem.")
            if motif_scores is None:
                motif = self.motif_extractor.extract(
                    audio_path,
                    stems,
                    mixture,
                    sample_rate,
                    validated_onset_variation=validated_onset_variation,
                    selected_candidate=selected_candidate,
                )
            else:
                motif = self.motif_extractor.extract(
                    audio_path,
                    stems,
                    mixture,
                    sample_rate,
                    scored_result=motif_scores,
                    validated_onset_variation=validated_onset_variation,
                    selected_candidate=selected_candidate,
                )
            motif_audio, accompaniment, scaled_stems, target_gain = _apply_common_headroom(
                motif["audio"], stems
            )
        except Exception as exc:  # keep a large batch running and report the exact cause
            return self._rejected_result(
                track,
                sample_dir,
                f"{type(exc).__name__}: {exc}",
            )

        sample_dir.mkdir(parents=True, exist_ok=True)
        extension = self.audio_format
        motif_seed_name = f"motif.{extension}"
        accompaniment_target_name = f"accompaniment.{extension}"
        vocal_target_name = f"vocals.{extension}"

        save_audio(sample_dir / motif_seed_name, motif_audio, sample_rate, audio_format=self.audio_format)
        save_audio(
            sample_dir / accompaniment_target_name,
            accompaniment,
            sample_rate,
            audio_format=self.audio_format,
        )
        save_audio(
            sample_dir / vocal_target_name,
            scaled_stems["vocals"],
            sample_rate,
            audio_format=self.audio_format,
        )
        stem_files = {"vocals": vocal_target_name}
        for stem_name, stem_audio in scaled_stems.items():
            if stem_name in {"drums", "vocals"}:
                continue
            if not stem_name.replace("_", "").isalnum():
                raise ValueError(f"Unsafe separator stem name: {stem_name!r}")
            stem_file = f"{stem_name}.{extension}"
            save_audio(
                sample_dir / stem_file,
                stem_audio,
                sample_rate,
                audio_format=self.audio_format,
            )
            stem_files[stem_name] = stem_file
        (sample_dir / "lyrics.txt").write_text(track.lyrics.strip() + "\n", encoding="utf-8")

        metadata: dict[str, Any] = {
            "schema_version": 4,
            "status": "accepted",
            "track_id": track.track_id,
            "artist": track.artist,
            "title": track.title,
            "genres": list(track.genres),
            "language": track.language,
            "source_audio": str(audio_path),
            "sample_rate": sample_rate,
            "target_gain": target_gain,
            "motif_stem": motif["stem_name"],
            "motif_scores": motif["stem_scores"],
            "motif_similarity": motif["similarity"],
            "motif_onset_variation": motif["onset_variation"],
            "motif_start_sec": motif["start_sec"],
            "motif_end_sec": motif["end_sec"],
            "motif_seed_file": motif_seed_name,
            "accompaniment_target_file": accompaniment_target_name,
            "vocal_target_file": vocal_target_name,
            "stem_files": stem_files,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return BuildResult(track.track_id, "accepted", None, str(sample_dir), motif["stem_name"])

    def build(self, tracks: Iterable[LocalTrack], *, max_songs: int | None = None) -> list[BuildResult]:
        from tqdm.auto import tqdm

        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_index = index_audio_files(self.audio_dir)
        selected = list(tracks)
        if max_songs is not None:
            selected = selected[:max_songs]

        results = [self.process_track(track, audio_index) for track in tqdm(selected, desc="Preparing MOHIM samples")]
        report_path = self.output_dir / "build_report.jsonl"
        report_path.write_text(
            "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results),
            encoding="utf-8",
        )
        return results
