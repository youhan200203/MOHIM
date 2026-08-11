"""Resumable DALI-to-MOHIM audio dataset builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .dali import DaliTrack, index_audio_files, resolve_audio_path
from .motif import MotifExtractor
from .separator import StemSeparator, save_audio


@dataclass(frozen=True)
class BuildResult:
    dali_id: str
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

    def _accepted_result(self, track: DaliTrack, sample_dir: Path) -> BuildResult | None:
        metadata_path = sample_dir / "metadata.json"
        if not self.resume or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        required = [
            sample_dir / metadata.get("motif_seed_file", ""),
            sample_dir / metadata.get("motif_target_file", ""),
            sample_dir / metadata.get("vocal_target_file", ""),
            sample_dir / "lyrics.txt",
        ]
        if metadata.get("status") == "accepted" and all(path.is_file() for path in required):
            return BuildResult(track.dali_id, "skipped", "already_processed", str(sample_dir), metadata.get("motif_stem"))
        return None

    def process_track(self, track: DaliTrack, audio_index: dict[str, Path]) -> BuildResult:
        sample_dir = self.output_dir / track.dali_id
        cached = self._accepted_result(track, sample_dir)
        if cached:
            return cached

        audio_path = resolve_audio_path(track, audio_index)
        if audio_path is None:
            return BuildResult(track.dali_id, "rejected", "audio_not_found", None, None)
        if not track.lyrics.strip():
            return BuildResult(track.dali_id, "rejected", "lyrics_missing", None, None)

        try:
            stems, sample_rate, mixture = self.separator.separate(audio_path)
            if "vocals" not in stems:
                raise ValueError("Separator did not return a vocals stem.")
            motif = self.motif_extractor.extract(audio_path, stems, mixture, sample_rate)
        except Exception as exc:  # keep a large batch running and report the exact cause
            return BuildResult(track.dali_id, "rejected", f"{type(exc).__name__}: {exc}", None, None)

        sample_dir.mkdir(parents=True, exist_ok=True)
        extension = self.audio_format
        motif_seed_name = f"motif.{extension}"
        motif_target_name = f"{motif['stem_name']}.{extension}"
        vocal_target_name = f"vocals.{extension}"

        save_audio(sample_dir / motif_seed_name, motif["audio"], sample_rate, audio_format=self.audio_format)
        save_audio(sample_dir / motif_target_name, stems[motif["stem_name"]], sample_rate, audio_format=self.audio_format)
        save_audio(sample_dir / vocal_target_name, stems["vocals"], sample_rate, audio_format=self.audio_format)
        (sample_dir / "lyrics.txt").write_text(track.lyrics.strip() + "\n", encoding="utf-8")

        metadata: dict[str, Any] = {
            "status": "accepted",
            "dali_id": track.dali_id,
            "artist": track.artist,
            "title": track.title,
            "genres": list(track.genres),
            "language": track.language,
            "source_audio": str(audio_path),
            "sample_rate": sample_rate,
            "motif_stem": motif["stem_name"],
            "motif_scores": motif["stem_scores"],
            "motif_similarity": motif["similarity"],
            "motif_start_sec": motif["start_sec"],
            "motif_end_sec": motif["end_sec"],
            "motif_seed_file": motif_seed_name,
            "motif_target_file": motif_target_name,
            "vocal_target_file": vocal_target_name,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return BuildResult(track.dali_id, "accepted", None, str(sample_dir), motif["stem_name"])

    def build(self, tracks: Iterable[DaliTrack], *, max_songs: int | None = None) -> list[BuildResult]:
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
