"""Download YouTube audio for a reviewed Genius seed manifest."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LocalTrack:
    track_id: str
    artist: str
    title: str
    genres: tuple[str, ...]
    language: str
    lyrics: str
    audio_url: str | None
    audio_path: str | None
    entry: Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tracks"), list):
        raise ValueError(f"JSON must contain a 'tracks' list: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clean_title(value: str) -> str:
    title = re.sub(r"\s*[\[(](official|lyrics?|audio|video|mv).*?[\])]\s*", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", title).strip()


def _first_entries(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = info.get("entries")
    if entries is None:
        return [dict(info)]
    return [dict(entry) for entry in entries if isinstance(entry, Mapping)]


def _search_youtube(query: str, max_results: int) -> list[dict[str, Any]]:
    import yt_dlp

    options = {
        "extract_flat": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"ytsearch{max_results}:{query}", download=False)
    if not isinstance(info, Mapping):
        return []
    return _first_entries(info)


_REJECTED_VIDEO_TERMS = (
    "cover",
    "instrumental",
    "karaoke",
    "live",
    "nightcore",
    "remix",
    "slowed",
    "sped up",
)
_INSIGNIFICANT_WORDS = {"and", "audio", "feat", "featuring", "ft", "official", "the", "video"}


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _clean_title(value).casefold())


def _significant_words(value: str) -> set[str]:
    return {
        word
        for word in _normalized_words(value)
        if len(word) >= 3 and word not in _INSIGNIFICANT_WORDS
    }


def _validate_youtube_candidate(
    artist: str,
    title: str,
    info: Mapping[str, Any],
    *,
    min_duration: float = 90.0,
    max_duration: float = 480.0,
) -> tuple[bool, str]:
    youtube_title = str(info.get("title") or "").strip()
    normalized_title = " ".join(_normalized_words(youtube_title))
    rejected = next(
        (term for term in _REJECTED_VIDEO_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized_title)),
        None,
    )
    if rejected:
        return False, f"title contains excluded term: {rejected}"

    duration = info.get("duration")
    if duration is not None and not min_duration <= float(duration) <= max_duration:
        return False, f"duration is outside {min_duration:g}-{max_duration:g} seconds"

    expected_words = _significant_words(title) or set(_normalized_words(title))
    candidate_words = _significant_words(youtube_title) or set(_normalized_words(youtube_title))
    overlap = len(expected_words & candidate_words) / max(1, len(expected_words))
    similarity = SequenceMatcher(
        None,
        " ".join(_normalized_words(title)),
        " ".join(_normalized_words(youtube_title)),
    ).ratio()
    artist_words = _significant_words(artist)
    uploader_words = _significant_words(str(info.get("uploader") or info.get("channel") or ""))
    if overlap < 0.6 and similarity < 0.5:
        return False, "title does not sufficiently match the seed"
    if artist_words and not artist_words & (candidate_words | uploader_words):
        return False, "result does not contain the seed artist"
    return True, ""


def _youtube_url(info: Mapping[str, Any]) -> str:
    url = str(info.get("webpage_url") or info.get("original_url") or info.get("url") or "").strip()
    if url.startswith("http"):
        return url
    video_id = str(info.get("id") or url).strip()
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else ""


def _download_audio(url: str, destination: Path) -> dict[str, Any]:
    import yt_dlp

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mohim-youtube-") as temporary:
        temp_dir = Path(temporary)
        options = {
            "format": "bestaudio/best",
            "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                    "preferredquality": "256",
                }
            ],
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
        if not isinstance(info, Mapping):
            raise RuntimeError(f"yt-dlp returned no metadata for {url}")
        entries = _first_entries(info)
        result = entries[0] if entries else dict(info)
        video_id = str(result.get("id") or "").strip()
        candidates = sorted(temp_dir.glob(f"{video_id}.*"))
        if not video_id or not candidates:
            raise FileNotFoundError(f"Downloaded audio was not produced for {url}")
        target = destination / f"{video_id}{candidates[0].suffix.lower()}"
        shutil.move(str(candidates[0]), target)
        result["local_audio_path"] = str(target.resolve())
        return result


def ingest_genius_seed(
    seed_path: str | Path,
    dataset_dir: str | Path,
    *,
    max_tracks: int | None = None,
    search_results: int = 8,
) -> dict[str, Any]:
    """Pair seed lyrics with downloadable YouTube audio and save after every track."""

    if max_tracks is not None and max_tracks <= 0:
        raise ValueError("max_tracks must be positive or None")
    if search_results <= 0:
        raise ValueError("search_results must be positive")

    seed = _read_json(Path(seed_path).expanduser().resolve())
    candidates = list(seed["tracks"])
    if max_tracks is not None:
        candidates = candidates[:max_tracks]

    root = Path(dataset_dir).expanduser().resolve()
    audio_dir = root / "audio"
    manifest_path = root / "tracks.json"
    previous = _read_json(manifest_path) if manifest_path.is_file() else {"tracks": []}
    existing = {str(item.get("seed_id") or item.get("track_id") or item.get("id")): dict(item) for item in previous["tracks"]}
    errors: list[dict[str, str]] = []
    interrupted = False

    def save() -> dict[str, Any]:
        tracks = sorted(existing.values(), key=lambda item: str(item.get("seed_id", "")))
        payload = {
            "metadata": {
                "name": "mohim_youtube_dataset",
                "seed_path": str(Path(seed_path).expanduser().resolve()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "num_tracks": len(tracks),
                "num_errors": len(errors),
                "interrupted": interrupted,
            },
            "tracks": tracks,
            "errors": errors,
        }
        _write_json(manifest_path, payload)
        return payload

    save()
    print(f"tracks JSON initialized: {manifest_path}", flush=True)
    try:
        for index, seed_track in enumerate(candidates, 1):
            seed_id = str(seed_track.get("id") or f"seed-{index}")
            prior = existing.get(seed_id)
            if prior and Path(str(prior.get("audio_path") or "")).is_file():
                print(f"[{index}/{len(candidates)}] {seed_track.get('artist')} - {seed_track.get('title')}: already downloaded", flush=True)
                continue

            artist = str(seed_track.get("artist") or "").strip()
            title = str(seed_track.get("title") or "").strip()
            lyrics = str(seed_track.get("lyrics") or "").strip()
            query = f"{artist} {title} official audio"
            failures: list[str] = []
            try:
                if not artist or not title or not lyrics:
                    raise ValueError("seed track must contain artist, title, and lyrics")
                downloaded: dict[str, Any] | None = None
                selected_url = ""
                for result in _search_youtube(query, search_results):
                    accepted, reason = _validate_youtube_candidate(artist, title, result)
                    if not accepted:
                        failures.append(reason)
                        continue
                    selected_url = _youtube_url(result)
                    if not selected_url:
                        failures.append("candidate has no URL")
                        continue
                    try:
                        downloaded = _download_audio(selected_url, audio_dir)
                        break
                    except Exception as exc:
                        failures.append(f"{type(exc).__name__}: {exc}")
                if downloaded is None:
                    raise RuntimeError("no downloadable matching result; " + " | ".join(failures[-5:]))

                existing[seed_id] = {
                    "track_id": seed_id,
                    "seed_id": seed_id,
                    "youtube_id": str(downloaded.get("id") or ""),
                    "source_url": selected_url,
                    "audio_path": downloaded["local_audio_path"],
                    "artist": artist,
                    "title": title,
                    "genres": list(seed_track.get("genres") or ["Pop"]),
                    "language": str(seed_track.get("language") or "English"),
                    "lyrics": lyrics,
                    "lyrics_source": seed_track.get("lyrics_source") or "genius",
                    "lyrics_source_url": seed_track.get("lyrics_source_url"),
                    "duration_seconds": downloaded.get("duration"),
                    "youtube_title": downloaded.get("title"),
                    "youtube_search_query": query,
                }
                status = "downloaded"
            except Exception as exc:
                errors.append(
                    {
                        "seed_id": seed_id,
                        "artist": artist,
                        "title": title,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                status = f"failed: {type(exc).__name__}: {exc}"
            save()
            print(f"[{index}/{len(candidates)}] {artist} - {title}: {status}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; completed downloads remain saved", flush=True)
    return save()


def load_local_tracks(manifest_path: str | Path, *, require_lyrics: bool = True) -> list[LocalTrack]:
    payload = _read_json(Path(manifest_path).expanduser().resolve())
    tracks: list[LocalTrack] = []
    for item in payload["tracks"]:
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
    return {path.stem: path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in extensions}


def resolve_audio_path(track: LocalTrack, audio_index: Mapping[str, Path]) -> Path | None:
    if track.audio_path:
        direct = Path(track.audio_path).expanduser()
        if direct.is_file():
            return direct.resolve()
    youtube_id = str(track.entry.get("youtube_id") or "") if isinstance(track.entry, Mapping) else ""
    return audio_index.get(youtube_id) or audio_index.get(track.track_id)
