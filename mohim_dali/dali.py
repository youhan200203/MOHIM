"""Load official DALI annotations and expose plain-lyrics track records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DaliTrack:
    """Small, serializable view over one DALI annotation entry."""

    dali_id: str
    artist: str
    title: str
    genres: tuple[str, ...]
    language: str
    lyrics: str
    audio_url: str | None
    audio_path: str | None
    entry: Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _annotation_root(entry: Any) -> Mapping[str, Any]:
    annotations = _field(entry, "annotations", {}) or {}
    if annotations.get("type") == "vertical" and hasattr(entry, "vertical2horizontal"):
        entry.vertical2horizontal()
        annotations = _field(entry, "annotations", {}) or {}
    return annotations.get("annot", annotations)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return _flatten_text(value.get("text", ""))
    if isinstance(value, Iterable):
        parts = [_flatten_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return ""


def extract_plain_lyrics(entry: Any) -> str:
    """Return human-readable lyrics while discarding all DALI timestamps."""

    annotations = _annotation_root(entry)
    for level, separator in (("lines", "\n"), ("paragraphs", "\n\n"), ("words", " ")):
        items = annotations.get(level) or []
        texts = [_flatten_text(_field(item, "text", "")) for item in items]
        texts = [text for text in texts if text]
        if texts:
            return separator.join(texts).strip()
    return ""


def _to_track(dali_id: str, entry: Any) -> DaliTrack:
    info = _field(entry, "info", {}) or {}
    metadata = info.get("metadata", {}) or {}
    audio = info.get("audio", {}) or {}
    genres = metadata.get("genres", ()) or ()
    if isinstance(genres, str):
        genres = (genres,)

    return DaliTrack(
        dali_id=str(info.get("id") or dali_id),
        artist=str(info.get("artist") or "").strip(),
        title=str(info.get("title") or "").strip(),
        genres=tuple(str(genre).strip() for genre in genres if str(genre).strip()),
        language=str(metadata.get("language") or "").strip(),
        lyrics=extract_plain_lyrics(entry),
        audio_url=str(audio.get("url")) if audio.get("url") else None,
        audio_path=str(audio.get("path")) if audio.get("path") not in (None, "None", "") else None,
        entry=entry,
    )


def load_dali(data_path: str | Path, *, keep: Iterable[str] | None = None) -> list[DaliTrack]:
    """Load a directory containing the access-controlled official DALI data."""

    try:
        import DALI as dali_code
    except ImportError as exc:  # pragma: no cover - dependency is optional in unit tests
        raise RuntimeError("Install the 'dali-dataset' package before loading DALI.") from exc

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DALI data path does not exist: {path}")

    keep_ids = list(keep or [])
    dataset = dali_code.get_the_DALI_dataset(str(path), skip=[], keep=keep_ids)
    return [_to_track(str(dali_id), entry) for dali_id, entry in dataset.items()]


def filter_tracks(
    tracks: Iterable[DaliTrack],
    *,
    language: str | None = "english",
    genre: str | None = "pop",
    require_lyrics: bool = True,
) -> list[DaliTrack]:
    """Filter DALI records using tolerant language and genre matching."""

    language_aliases = {
        "english": {"english", "eng", "en"},
        "eng": {"english", "eng", "en"},
        "en": {"english", "eng", "en"},
    }
    wanted_language = language.lower().strip() if language else None
    accepted_languages = language_aliases.get(wanted_language, {wanted_language}) if wanted_language else set()
    wanted_genre = genre.lower().strip() if genre else None

    selected: list[DaliTrack] = []
    for track in tracks:
        track_language = track.language.lower().strip()
        if wanted_language and track_language not in accepted_languages:
            continue
        if wanted_genre and not any(wanted_genre in item.lower() for item in track.genres):
            continue
        if require_lyrics and not track.lyrics.strip():
            continue
        selected.append(track)
    return selected


def index_audio_files(audio_dir: str | Path) -> dict[str, Path]:
    """Index local audio by filename stem so DALI IDs can be resolved quickly."""

    root = Path(audio_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {root}")
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    return {
        path.stem: path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }


def resolve_audio_path(track: DaliTrack, audio_index: Mapping[str, Path]) -> Path | None:
    """Resolve a DALI track to an already acquired local audio file."""

    if track.audio_path:
        candidate = Path(track.audio_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    exact = audio_index.get(track.dali_id)
    if exact:
        return exact

    normalized = f"{track.artist}-{track.title}".lower().replace(" ", "_")
    for stem, path in audio_index.items():
        if stem.lower().replace(" ", "_") == normalized:
            return path
    return None
