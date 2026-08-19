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
    silence_db: float = -40.0
    min_presence: float = 0.70
    max_similarity_difference: float = 0.40
    onset_threshold: float = 0.60
    pitch_class_span_threshold: float = 0.25
    onset_variation_threshold: float = 0.20
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
    return {
        "presence": presence,
        "movement": movement,
        "repetition": repetition,
    }


def _resize_time(feature: np.ndarray, target_frames: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, feature.shape[1])
    new_x = np.linspace(0.0, 1.0, target_frames)
    return np.vstack([np.interp(new_x, old_x, row) for row in feature])


def _max_shifted_cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
    max_shift: int,
    cosine_similarity: Any,
) -> float:
    scores = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            first_overlap = first[-shift:]
            second_overlap = second[:shift]
        elif shift > 0:
            first_overlap = first[:-shift]
            second_overlap = second[shift:]
        else:
            first_overlap = first
            second_overlap = second
        scores.append(float(cosine_similarity(first_overlap[None], second_overlap[None])[0, 0]))
    return max(scores)


def pitch_class_span(audio: Any, sample_rate: int, hop_length: int = 512) -> float:
    """Return the fraction of pitch classes active in at least 10% of valid frames."""
    import librosa

    if hasattr(audio, "detach"):
        mono = audio.mean(0).detach().cpu().numpy()
    else:
        values = np.asarray(audio)
        mono = values.mean(0) if values.ndim > 1 else values
    chroma = librosa.feature.chroma_cens(y=mono, sr=sample_rate, hop_length=hop_length)
    chroma_sum = chroma.sum(axis=0, keepdims=True)
    valid_chroma = chroma_sum.ravel() > 1e-8
    if not np.any(valid_chroma):
        return 0.0
    chroma_norm = np.divide(
        chroma,
        chroma_sum,
        out=np.zeros_like(chroma),
        where=chroma_sum > 1e-8,
    )
    valid_norm = chroma_norm[:, valid_chroma]
    active_pc = valid_norm >= (0.60 * valid_norm.max(axis=0, keepdims=True))
    pitch_class_count = int(np.sum(np.mean(active_pc, axis=1) >= 0.10))
    return pitch_class_count / 12.0


def _robust_cv(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    return float(np.clip(mad / (abs(median) + 1e-8), 0.0, 1.0))


def onset_variation(audio: Any, sample_rate: int, hop_length: int = 512) -> float:
    """Return robust variation in onset strengths and inter-onset intervals."""
    import librosa

    if hasattr(audio, "detach"):
        mono = audio.mean(0).detach().cpu().numpy()
    else:
        values = np.asarray(audio)
        mono = values.mean(0) if values.ndim > 1 else values
    onset = librosa.onset.onset_strength(
        y=mono,
        sr=sample_rate,
        hop_length=hop_length,
    )
    if not len(onset) or float(np.max(onset)) <= 0.0:
        return 0.0
    peaks = librosa.util.peak_pick(
        onset,
        pre_max=2,
        post_max=2,
        pre_avg=2,
        post_avg=2,
        delta=0.05 * float(np.max(onset)),
        wait=1,
    )
    return 0.5 * (_robust_cv(onset[peaks]) + _robust_cv(np.diff(peaks)))


def find_repeating_motif(
    stem_wav: Any,
    sample_rate: int,
    downbeats: Iterable[float],
    config: MotifConfig,
    candidate_wav: Any | None = None,
) -> tuple[int, int, float] | None:
    """Return the first candidate that passes onset and pitch-span thresholds."""
    feature_audio = stem_wav if candidate_wav is None else candidate_wav
    for start, end, onset_similarity, _, similarity, _ in score_repeating_motifs(
        stem_wav, sample_rate, downbeats, config
    ):
        span = pitch_class_span(feature_audio[:, start:end], sample_rate)
        if (
            onset_similarity >= config.onset_threshold
            and span >= config.pitch_class_span_threshold
        ):
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
    onset_n_fft = 2048
    onset_preroll_frames = onset_n_fft // hop_length
    motif_length = bar_length * config.bars
    padded_mono = np.pad(mono, (onset_preroll_frames * hop_length, 0))
    padded_onset = librosa.onset.onset_strength(
        y=padded_mono,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=onset_n_fft,
    )
    chroma = librosa.feature.chroma_cens(y=mono, sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
    onset = padded_onset[onset_preroll_frames : onset_preroll_frames + len(rms)]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)

    segments: list[tuple[float, float, np.ndarray, np.ndarray, float]] = []
    for start_sec in downbeats_array:
        end_sec = start_sec + motif_length
        if end_sec * sample_rate > len(mono):
            continue
        start_frame = int(librosa.time_to_frames(start_sec, sr=sample_rate, hop_length=hop_length))
        end_frame = int(librosa.time_to_frames(end_sec, sr=sample_rate, hop_length=hop_length))
        onset_segment = _resize_time(onset[None, start_frame:end_frame], 256).ravel()
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
            onset_similarity = _max_shifted_cosine_similarity(
                first_onset,
                other_onset,
                max_shift=3,
                cosine_similarity=cosine_similarity,
            )
            chroma_similarity = cosine_similarity(first_chroma[None], other_chroma[None])[0, 0]
            onset_scores.append(float(onset_similarity))
            chroma_scores.append(float(chroma_similarity))
        if not onset_scores:
            continue
        onset_similarity = float(np.mean(onset_scores))
        chroma_similarity = float(np.mean(chroma_scores))
        if abs(onset_similarity - chroma_similarity) > config.max_similarity_difference:
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

    def extract(
        self,
        audio_path: str | Path,
        stems: dict[str, Any],
        full_wav: Any,
        sample_rate: int,
        *,
        scored_result: dict[str, Any] | None = None,
        validated_onset_variation: float | None = None,
    ) -> dict[str, Any]:
        """Return the strongest first onset-and-pitch-passing motif across stems."""
        del full_wav  # Kept in the public API for DatasetBuilder compatibility.
        result = (
            scored_result
            if scored_result is not None
            else self.score_all(audio_path, stems, sample_rate)
        )
        candidates = [name for name in self.config.candidate_stems if name in stems]
        eligible = [
            row
            for row in result["candidates"]
            if row["onset_similarity"] >= self.config.onset_threshold
            and row["pitch_class_span"] >= self.config.pitch_class_span_threshold
        ]
        first_by_stem: dict[str, dict[str, float | int | str]] = {}
        for row in sorted(eligible, key=lambda item: float(item["start_sec"])):
            first_by_stem.setdefault(str(row["stem_name"]), row)

        scores: dict[str, dict[str, float | bool | None]] = {}
        for name in candidates:
            match = first_by_stem.get(name)
            if match is None:
                scores[name] = {
                    "matched": False,
                    "start_sec": None,
                    "end_sec": None,
                    "active_ratio": None,
                    "onset_similarity": None,
                    "chroma_similarity": None,
                    "pitch_class_span": None,
                    "similarity": 0.0,
                }
                continue
            scores[name] = {
                "matched": True,
                "start_sec": float(match["start_sec"]),
                "end_sec": float(match["end_sec"]),
                "active_ratio": float(match["active_ratio"]),
                "onset_similarity": float(match["onset_similarity"]),
                "chroma_similarity": float(match["chroma_similarity"]),
                "pitch_class_span": float(match["pitch_class_span"]),
                "similarity": float(match["similarity"]),
            }

        if not first_by_stem:
            raise ValueError(
                "No repeated four-bar motif passed the onset and pitch-class-span thresholds."
            )
        match = max(first_by_stem.values(), key=lambda row: float(row["similarity"]))
        name = str(match["stem_name"])
        start = round(float(match["start_sec"]) * sample_rate)
        end = round(float(match["end_sec"]) * sample_rate)
        motif_audio = result["melodic_accompaniment"][:, start:end]
        variation = (
            onset_variation(motif_audio, sample_rate)
            if validated_onset_variation is None
            else float(validated_onset_variation)
        )
        if variation < self.config.onset_variation_threshold:
            raise ValueError(
                "Selected motif onset variation "
                f"{variation:.6f} is below {self.config.onset_variation_threshold:.6f}."
            )
        return {
            "stem_name": name,
            "stem_scores": scores,
            "start_frame": start,
            "end_frame": end,
            "start_sec": float(match["start_sec"]),
            "end_sec": float(match["end_sec"]),
            "similarity": float(match["similarity"]),
            "onset_variation": variation,
            "audio": motif_audio,
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
        melodic = melodic_accompaniment(stems, candidates)
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
                        "pitch_class_span": pitch_class_span(
                            melodic[:, start:end], sample_rate
                        ),
                    }
                )
        return {
            "candidates": rows,
            "melodic_accompaniment": melodic,
            "sample_rate": sample_rate,
        }


def create_beat_tracker(checkpoint_path: str | Path, device: str = "cuda") -> Any:
    from beat_this.inference import File2Beats

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Beat This checkpoint not found: {checkpoint}")
    return File2Beats(checkpoint_path=str(checkpoint), device=device, dbn=False)
