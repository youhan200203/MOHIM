"""Find a repeated four-bar motif in each candidate stem and select the best."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_F0_DETECTOR: Any = None


@dataclass(frozen=True)
class MotifConfig:
    candidate_stems: tuple[str, ...] = ("guitar", "piano", "bass", "other")
    bars: int = 4
    search_seconds: float = 30.0
    similarity_threshold: float = 0.56
    silence_db: float = -40.0
    min_presence: float = 0.70
    min_stem_score: float = 0.25


def movement_score(midi_notes: Iterable[float], octave_fold: bool = True) -> float:
    values = np.asarray(list(midi_notes), dtype=np.float64)
    if len(values) < 2:
        return 0.0
    intervals = np.diff(values)
    if octave_fold:
        intervals = ((intervals + 6) % 12) - 6
    absolute = np.abs(intervals)
    moved_ratio = np.mean(absolute > 0)
    average_interval = np.clip(np.mean(absolute) / 5.0, 0.0, 1.0)
    pitch_classes = values % 12
    unique_ratio = len(set(pitch_classes.tolist())) / len(pitch_classes)
    return float(0.4 * moved_ratio + 0.3 * average_interval + 0.3 * unique_ratio)


def repetition_score(
    midi_notes: Iterable[float], min_ngram: int = 3, max_ngram: int = 6, octave_fold: bool = True
) -> float:
    values = np.asarray(list(midi_notes), dtype=np.float64)
    if len(values) < min_ngram + 1:
        return 0.0
    intervals = np.diff(values)
    if octave_fold:
        intervals = ((intervals + 6) % 12) - 6

    covered = np.zeros(len(intervals), dtype=bool)
    total_weight = 0.0
    matched_weight = 0.0
    for size in range(min_ngram, min(max_ngram, len(intervals)) + 1):
        positions: dict[tuple[float, ...], list[int]] = {}
        for start in range(len(intervals) - size + 1):
            phrase = tuple(intervals[start : start + size])
            positions.setdefault(phrase, []).append(start)
        for starts in positions.values():
            total_weight += size
            if len(starts) > 1:
                matched_weight += size
                for start in starts:
                    covered[start : start + size] = True
    if total_weight == 0:
        return 0.0
    return float(0.5 * matched_weight / total_weight + 0.5 * covered.mean())


def _extract_midi_notes(mono: np.ndarray, sample_rate: int) -> list[float]:
    from swift_f0 import SwiftF0, segment_notes

    global _F0_DETECTOR
    if _F0_DETECTOR is None:
        _F0_DETECTOR = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.9)
    result = _F0_DETECTOR.detect_from_array(mono.astype(np.float32), sample_rate)
    notes = segment_notes(
        result,
        split_semitone_threshold=0.8,
        min_note_duration=0.05,
        unvoiced_grace_period=0.05,
    )
    return [float(note.pitch_midi) for note in notes]


def score_motif_candidate(stem_wav: Any, full_wav: Any, sample_rate: int, hop_length: int = 512) -> dict[str, float]:
    import librosa

    stem = stem_wav.mean(0).detach().cpu().numpy()
    full = full_wav.mean(0).detach().cpu().numpy()
    peak = np.max(np.abs(full))
    stem = stem / peak
    full = full / peak
    stem_rms = librosa.feature.rms(y=stem, hop_length=hop_length)[0]
    full_rms = librosa.feature.rms(y=full, hop_length=hop_length)[0]
    active = full_rms > 1e-4
    contribution = stem_rms[active] / (full_rms[active] + 1e-8)
    presence = float(0.4 * np.mean(contribution) + 0.6 * np.percentile(contribution, 95))
    midi_notes = _extract_midi_notes(stem.astype(np.float32), sample_rate) if presence >= 0.30 else []
    movement = movement_score(midi_notes)
    repetition = repetition_score(midi_notes)
    total = 0.5 * presence + 0.25 * movement + 0.25 * repetition
    return {
        "presence": presence,
        "movement": movement,
        "repetition": repetition,
        "total": float(total),
    }


def _resize_time(feature: np.ndarray, target_frames: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, feature.shape[1])
    new_x = np.linspace(0.0, 1.0, target_frames)
    return np.vstack([np.interp(new_x, old_x, row) for row in feature])


def find_repeating_motif(
    stem_wav: Any,
    sample_rate: int,
    downbeats: Iterable[float],
    config: MotifConfig,
) -> tuple[int, int, float] | None:
    """Return the first candidate that passes the configured threshold."""
    for start, end, _, _, similarity, _ in score_repeating_motifs(
        stem_wav, sample_rate, downbeats, config
    ):
        if similarity >= config.similarity_threshold:
            return start, end, similarity
    return None


def score_repeating_motifs(
    stem_wav: Any,
    sample_rate: int,
    downbeats: Iterable[float],
    config: MotifConfig,
) -> list[tuple[int, int, float, float, float, float]]:
    """Score every four-bar start downbeat inside the search window."""
    import librosa
    from sklearn.metrics.pairwise import cosine_similarity

    mono = stem_wav.mean(0).detach().cpu().numpy()
    downbeats_array = np.asarray(list(downbeats), dtype=np.float64).reshape(-1)
    if len(downbeats_array) <= config.bars * 2:
        return []
    bar_length = float(np.median(np.diff(downbeats_array)))
    hop_length = 512
    motif_length = bar_length * config.bars
    onset = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop_length)
    chroma = librosa.feature.chroma_cens(y=mono, sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)

    segments: list[tuple[float, float, np.ndarray, np.ndarray, float]] = []
    for start_sec in downbeats_array:
        end_sec = start_sec + motif_length
        if end_sec * sample_rate > len(mono):
            continue
        start_frame = int(librosa.time_to_frames(start_sec, sr=sample_rate, hop_length=hop_length))
        end_frame = int(librosa.time_to_frames(end_sec, sr=sample_rate, hop_length=hop_length))
        onset_segment = _resize_time(onset[None, start_frame:end_frame], 64).ravel()
        chroma_segment = _resize_time(chroma[:, start_frame:end_frame], 64).ravel()
        active = rms_db[start_frame:end_frame] > config.silence_db
        active_ratio = float(np.sum(active) / len(active)) if len(active) else 0.0
        if active_ratio < config.min_presence:
            continue
        segments.append((start_sec, end_sec, onset_segment, chroma_segment, active_ratio))

    candidates: list[tuple[int, int, float, float, float, float]] = []
    for start_sec, end_sec, first_onset, first_chroma, active_ratio in segments:
        if start_sec >= config.search_seconds:
            break
        onset_scores: list[float] = []
        chroma_scores: list[float] = []
        for other_start, _, other_onset, other_chroma, _ in segments:
            if abs(other_start - start_sec) < motif_length * 0.9:
                continue
            onset_similarity = cosine_similarity(first_onset[None], other_onset[None])[0, 0]
            chroma_similarity = cosine_similarity(first_chroma[None], other_chroma[None])[0, 0]
            onset_scores.append(float(onset_similarity))
            chroma_scores.append(float(chroma_similarity))
        if not onset_scores:
            continue
        onset_similarity = float(np.mean(onset_scores))
        chroma_similarity = float(np.mean(chroma_scores))
        if abs(onset_similarity - chroma_similarity) >= 0.40:
            continue
        similarity = 0.3 * onset_similarity + 0.7 * chroma_similarity
        candidates.append(
            (
                int(start_sec * sample_rate),
                int(end_sec * sample_rate),
                onset_similarity,
                chroma_similarity,
                float(similarity),
                active_ratio,
            )
        )
    return candidates


def melodic_accompaniment(stems: dict[str, Any], candidate_stems: Iterable[str]) -> Any:
    """Sum available melodic stems without drums, vocals, or per-stem normalization."""
    selected = [stems[name] for name in candidate_stems if name in stems]
    if not selected:
        raise ValueError("No configured motif candidate stems were produced by the separator.")
    reference_shape = selected[0].shape
    if any(stem.shape != reference_shape for stem in selected):
        raise ValueError("Motif candidate stems are not time-aligned.")
    return sum(selected[1:], selected[0].clone())


class MotifExtractor:
    def __init__(self, beat_tracker: Any, config: MotifConfig | None = None) -> None:
        self.beat_tracker = beat_tracker
        self.config = config or MotifConfig()

    def extract(self, audio_path: str | Path, stems: dict[str, Any], full_wav: Any, sample_rate: int) -> dict[str, Any]:
        """Return the strongest first-passing motif across all candidate stems."""
        del full_wav  # Kept in the public API for DatasetBuilder compatibility.
        candidates = [name for name in self.config.candidate_stems if name in stems]
        if not candidates:
            raise ValueError("No configured motif candidate stems were produced by the separator.")
        _, downbeats = self.beat_tracker(str(audio_path))
        downbeats = tuple(np.asarray(downbeats, dtype=np.float64).reshape(-1).tolist())
        matches = {
            name: find_repeating_motif(stems[name], sample_rate, downbeats, self.config)
            for name in candidates
        }

        scores: dict[str, dict[str, float | bool | None]] = {}
        for name, match in matches.items():
            if match is None:
                scores[name] = {
                    "matched": False,
                    "start_sec": None,
                    "end_sec": None,
                    "similarity": 0.0,
                    "dominance": 0.0,
                    "total": 0.0,
                }
                continue
            start, end, similarity = match
            candidate_rms = float(stems[name][:, start:end].square().mean().sqrt())
            total_rms = sum(
                float(stems[other][:, start:end].square().mean().sqrt())
                for other in candidates
            )
            dominance = candidate_rms / max(total_rms, 1e-8)
            total = 0.85 * similarity + 0.15 * dominance
            scores[name] = {
                "matched": True,
                "start_sec": start / sample_rate,
                "end_sec": end / sample_rate,
                "similarity": float(similarity),
                "dominance": float(dominance),
                "total": float(total),
            }

        matched = [name for name in candidates if matches[name] is not None]
        if not matched:
            raise ValueError("No repeated four-bar motif passed the similarity threshold.")
        name = max(matched, key=lambda candidate: float(scores[candidate]["total"]))
        match = matches[name]
        assert match is not None
        start, end, similarity = match
        return {
            "stem_name": name,
            "stem_scores": scores,
            "start_frame": start,
            "end_frame": end,
            "start_sec": start / sample_rate,
            "end_sec": end / sample_rate,
            "similarity": similarity,
            "audio": melodic_accompaniment(stems, candidates)[:, start:end],
        }

    def score_all(
        self,
        audio_path: str | Path,
        stems: dict[str, Any],
        sample_rate: int,
    ) -> dict[str, Any]:
        """Return every pre-threshold candidate for manual calibration."""
        candidates = [name for name in self.config.candidate_stems if name in stems]
        if not candidates:
            raise ValueError("No configured motif candidate stems were produced by the separator.")
        _, downbeats = self.beat_tracker(str(audio_path))
        downbeats = tuple(np.asarray(downbeats, dtype=np.float64).reshape(-1).tolist())
        rows: list[dict[str, float | int | str]] = []
        for name in candidates:
            matches = score_repeating_motifs(stems[name], sample_rate, downbeats, self.config)
            for candidate_index, (
                start,
                end,
                onset_similarity,
                chroma_similarity,
                similarity,
                active_ratio,
            ) in enumerate(matches):
                candidate_rms = float(stems[name][:, start:end].square().mean().sqrt())
                total_rms = sum(
                    float(stems[other][:, start:end].square().mean().sqrt())
                    for other in candidates
                )
                dominance = candidate_rms / max(total_rms, 1e-8)
                rows.append(
                    {
                        "stem_name": name,
                        "candidate_index": candidate_index,
                        "start_sec": start / sample_rate,
                        "end_sec": end / sample_rate,
                        "active_ratio": active_ratio,
                        "onset_similarity": onset_similarity,
                        "chroma_similarity": chroma_similarity,
                        "similarity": float(similarity),
                        "dominance": float(dominance),
                        "total": float(0.85 * similarity + 0.15 * dominance),
                    }
                )
        return {
            "candidates": rows,
            "melodic_accompaniment": melodic_accompaniment(stems, candidates),
            "sample_rate": sample_rate,
        }


def create_beat_tracker(checkpoint_path: str | Path, device: str = "cuda") -> Any:
    from beat_this.inference import File2Beats

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Beat This checkpoint not found: {checkpoint}")
    return File2Beats(checkpoint_path=str(checkpoint), device=device, dbn=False)
