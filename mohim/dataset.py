"""Resumable MOHIM audio dataset builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .local_dataset import LocalTrack, index_audio_files, resolve_audio_path
from .motif import MotifExtractor
from .separator import StemSeparator, save_audio


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

    def _accepted_result(self, track: LocalTrack, sample_dir: Path) -> BuildResult | None:
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
        if (
            metadata.get("schema_version") == 3
            and metadata.get("status") == "accepted"
            and all(required_names)
            and isinstance(stem_files, dict)
            and stem_files.get("vocals") == metadata.get("vocal_target_file")
            and "drums" not in stem_files
            and all(path.is_file() for path in required)
        ):
            return BuildResult(track.track_id, "skipped", "already_processed", str(sample_dir), metadata.get("motif_stem"))
        return None

    def process_track(self, track: LocalTrack, audio_index: dict[str, Path]) -> BuildResult:
        sample_dir = self.output_dir / track.track_id
        cached = self._accepted_result(track, sample_dir)
        if cached:
            return cached

        audio_path = resolve_audio_path(track, audio_index)
        if audio_path is None:
            return BuildResult(track.track_id, "rejected", "audio_not_found", None, None)
        if not track.lyrics.strip():
            return BuildResult(track.track_id, "rejected", "lyrics_missing", None, None)

        try:
            stems, sample_rate, mixture = self.separator.separate(audio_path)
            if "vocals" not in stems:
                raise ValueError("Separator did not return a vocals stem.")
            motif = self.motif_extractor.extract(audio_path, stems, mixture, sample_rate)
            motif_audio, accompaniment, scaled_stems, target_gain = _apply_common_headroom(
                motif["audio"], stems
            )
        except Exception as exc:  # keep a large batch running and report the exact cause
            return BuildResult(track.track_id, "rejected", f"{type(exc).__name__}: {exc}", None, None)

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
            "schema_version": 3,
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
