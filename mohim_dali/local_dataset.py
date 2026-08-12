"""Build and load editable datasets from Genius metadata and YouTube audio."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dali import DaliTrack


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"metadata": {"name": "mohim_local_youtube"}, "tracks": [], "errors": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tracks"), list):
        raise ValueError(f"Local dataset manifest must contain a 'tracks' list: {path}")
    return dict(payload)


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clean_title(value: str) -> str:
    title = re.sub(r"\s*[\[(](official|lyrics?|audio|video|mv).*?[\])]\s*", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", title).strip()


def _artist_title(info: Mapping[str, Any]) -> tuple[str, str]:
    artist = str(info.get("artist") or info.get("uploader") or "").strip()
    title = _clean_title(str(info.get("track") or info.get("title") or "").strip())
    if " - " in title and not info.get("track"):
        possible_artist, possible_title = title.split(" - ", 1)
        if possible_artist.strip() and possible_title.strip():
            artist = possible_artist.strip()
            title = possible_title.strip()
    return artist, title


def _search_genius_lyrics(title: str, artist: str, access_token: str | None) -> tuple[str, str | None]:
    if not access_token:
        return "", None
    import lyricsgenius

    genius = lyricsgenius.Genius(
        access_token,
        verbose=False,
        remove_section_headers=False,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
    )
    song = genius.search_song(title, artist or None)
    if song is None:
        return "", None
    lyrics = str(song.lyrics or "").strip()
    lyrics = re.sub(r"^.*?Lyrics\s*", "", lyrics, count=1, flags=re.I | re.S)
    lyrics = re.sub(r"\s*\d*Embed\s*$", "", lyrics, flags=re.I).strip()
    return lyrics, getattr(song, "url", None)


def _genius_client(access_token: str | None):
    import lyricsgenius

    return lyricsgenius.Genius(
        access_token,
        verbose=False,
        remove_section_headers=False,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
    )


def _iter_genius_tag_candidates(client: Any, tag: str, max_pages: int) -> Iterable[dict[str, str]]:
    page = 1
    seen: set[str] = set()
    while page and page <= max_pages:
        response = client.tag(tag, page=page) or {}
        for hit in response.get("hits") or []:
            genius_url = str(hit.get("url") or "").strip()
            title = str(hit.get("title") or "").strip()
            artists = hit.get("artists") or []
            if isinstance(artists, str):
                artist = artists.strip()
            else:
                artist = ", ".join(str(value).strip() for value in artists if str(value).strip())
            if not artist:
                title_with_artists = str(hit.get("title_with_artists") or "")
                if " by " in title_with_artists:
                    _, artist = title_with_artists.rsplit(" by ", 1)
            if genius_url and title and artist and genius_url not in seen:
                seen.add(genius_url)
                yield {"artist": artist, "title": title, "genius_url": genius_url}
        next_page = response.get("next_page")
        page = int(next_page) if next_page else 0


def _genius_lyrics_from_url(client: Any, genius_url: str) -> str:
    lyrics = str(client.lyrics(song_url=genius_url) or "").strip()
    lyrics = re.sub(r"^.*?Lyrics\s*", "", lyrics, count=1, flags=re.I | re.S)
    return re.sub(r"\s*\d*Embed\s*$", "", lyrics, flags=re.I).strip()


def _first_entry(info: Mapping[str, Any]) -> dict[str, Any]:
    entries = info.get("entries")
    if entries is not None:
        for entry in entries:
            if isinstance(entry, Mapping):
                return dict(entry)
        raise RuntimeError("YouTube search returned no results")
    return dict(info)


def _search_youtube(query: str) -> dict[str, Any]:
    import yt_dlp

    options = {
        "extract_flat": False,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"ytsearch1:{query}", download=False)
    if not isinstance(info, Mapping):
        raise RuntimeError(f"YouTube search returned no metadata for {query}")
    return _first_entry(info)


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

_INSIGNIFICANT_MATCH_WORDS = {"and", "audio", "feat", "featuring", "ft", "official", "the", "video"}


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _clean_title(value).casefold())


def _significant_words(value: str) -> set[str]:
    return {
        word
        for word in _normalized_words(value)
        if len(word) >= 3 and word not in _INSIGNIFICANT_MATCH_WORDS
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
        return False, f"YouTube title contains excluded term: {rejected}"

    duration = info.get("duration")
    if duration is not None and not min_duration <= float(duration) <= max_duration:
        return False, f"YouTube duration is outside {min_duration:g}-{max_duration:g} seconds"

    expected_words = _significant_words(title) or set(_normalized_words(title))
    candidate_words = _significant_words(youtube_title) or set(_normalized_words(youtube_title))
    title_overlap = len(expected_words & candidate_words) / max(1, len(expected_words))
    title_similarity = SequenceMatcher(
        None,
        " ".join(_normalized_words(title)),
        " ".join(_normalized_words(youtube_title)),
    ).ratio()
    artist_words = _significant_words(artist)
    uploader_words = _significant_words(str(info.get("uploader") or ""))
    artist_overlap = bool(artist_words & (candidate_words | uploader_words))
    if title_overlap < 0.6 and title_similarity < 0.5:
        return False, "YouTube title does not sufficiently match the Genius title"
    if artist_words and not artist_overlap:
        return False, "YouTube result does not contain the Genius artist"
    return True, ""


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
        info = _first_entry(info)
        video_id = str(info.get("id") or "").strip()
        if not video_id:
            raise RuntimeError(f"YouTube metadata has no video id: {url}")
        candidates = sorted(temp_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"Downloaded audio was not produced for {url}")
        target = destination / f"{video_id}{candidates[0].suffix.lower()}"
        shutil.move(str(candidates[0]), target)
        result = dict(info)
        result["local_audio_path"] = str(target.resolve())
        return result


def ingest_genius_tag_tracks(
    dataset_dir: str | Path,
    *,
    genius_access_token: str | None = None,
    tag: str = "pop",
    max_tracks: int = 100,
    max_pages: int = 50,
    default_language: str = "English",
) -> dict[str, Any]:
    """Pair Genius tag lyrics with a validated first-result YouTube download."""

    if max_tracks <= 0:
        raise ValueError("max_tracks must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    root = Path(dataset_dir).expanduser().resolve()
    audio_dir = root / "audio"
    manifest_path = root / "tracks.json"
    payload = _read_manifest(manifest_path)
    existing = {
        str(item.get("lyrics_source_url") or item.get("source_url")): dict(item)
        for item in payload["tracks"]
    }
    completed = sum(
        Path(str(item.get("audio_path", ""))).is_file() and bool(str(item.get("lyrics") or "").strip())
        for item in existing.values()
    )
    errors: list[dict[str, str]] = []
    client = _genius_client(genius_access_token)

    for candidate in _iter_genius_tag_candidates(client, tag, max_pages):
        if completed >= max_tracks:
            break
        genius_url = candidate["genius_url"]
        previous = existing.get(genius_url)
        if previous and Path(str(previous.get("audio_path", ""))).is_file():
            continue
        artist = candidate["artist"]
        title = candidate["title"]
        try:
            lyrics = str((previous or {}).get("lyrics") or "").strip()
            if not lyrics:
                lyrics = _genius_lyrics_from_url(client, genius_url)
            if not lyrics:
                raise RuntimeError("Genius returned no lyrics")

            query = f"{artist} {title} official audio"
            search_info = _search_youtube(query)
            accepted, reason = _validate_youtube_candidate(artist, title, search_info)
            if not accepted:
                raise RuntimeError(reason)
            youtube_url = str(search_info.get("webpage_url") or search_info.get("original_url") or "").strip()
            if not youtube_url:
                video_id = str(search_info.get("id") or "").strip()
                youtube_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
            if not youtube_url:
                raise RuntimeError("YouTube search result has no URL")

            info = _download_audio(youtube_url, audio_dir)
            video_id = str(info["id"])
            existing[genius_url] = {
                "id": video_id,
                "source_url": youtube_url,
                "audio_path": info["local_audio_path"],
                "artist": artist,
                "title": title,
                "genres": [tag.title()],
                "language": default_language,
                "lyrics": lyrics,
                "lyrics_status": "found",
                "lyrics_source": "genius",
                "lyrics_source_url": genius_url,
                "lyrics_error": None,
                "duration_seconds": info.get("duration"),
                "youtube_title": info.get("title") or search_info.get("title"),
                "youtube_search_query": query,
                "candidate_source": f"genius_tag:{tag}",
            }
            completed += 1
        except Exception as exc:
            errors.append(
                {
                    "artist": artist,
                    "title": title,
                    "genius_url": genius_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    tracks = sorted(existing.values(), key=lambda item: str(item.get("id", "")))
    payload = {
        "metadata": {
            "name": "mohim_genius_youtube",
            "tag": tag,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_tracks": len(tracks),
            "num_missing_lyrics": sum(not str(item.get("lyrics") or "").strip() for item in tracks),
        },
        "tracks": tracks,
        "errors": errors,
    }
    _write_manifest(manifest_path, payload)
    return payload


def ingest_youtube_tracks(
    urls: Iterable[str],
    dataset_dir: str | Path,
    *,
    genius_access_token: str | None = None,
    default_genre: str = "Pop",
    default_language: str = "English",
) -> dict[str, Any]:
    """Download URL audio and create an editable JSON manifest.

    Genius lookup is optional. Missing lyrics remain an empty ``lyrics`` field so
    the user can paste reviewed text directly into ``tracks.json``.
    """

    root = Path(dataset_dir).expanduser().resolve()
    audio_dir = root / "audio"
    manifest_path = root / "tracks.json"
    payload = _read_manifest(manifest_path)
    existing = {str(item.get("source_url")): dict(item) for item in payload["tracks"]}
    errors: list[dict[str, str]] = []

    for raw_url in urls:
        url = str(raw_url).strip()
        if not url:
            continue
        previous = existing.get(url)
        if previous and Path(str(previous.get("audio_path", ""))).is_file():
            continue
        try:
            info = _download_audio(url, audio_dir)
            video_id = str(info["id"])
            artist, title = _artist_title(info)
            prior_lyrics = str((previous or {}).get("lyrics") or "").strip()
            lyrics, lyrics_url = (prior_lyrics, (previous or {}).get("lyrics_source_url"))
            lyrics_error = None
            if not lyrics:
                try:
                    lyrics, lyrics_url = _search_genius_lyrics(title, artist, genius_access_token)
                except Exception as exc:
                    lyrics_error = f"{type(exc).__name__}: {exc}"
            existing[url] = {
                "id": video_id,
                "source_url": url,
                "audio_path": info["local_audio_path"],
                "artist": artist,
                "title": title,
                "genres": list((previous or {}).get("genres") or [default_genre]),
                "language": str((previous or {}).get("language") or default_language),
                "lyrics": lyrics,
                "lyrics_status": "found" if lyrics else "manual_required",
                "lyrics_source": "genius" if lyrics and not prior_lyrics else ("manual" if prior_lyrics else None),
                "lyrics_source_url": lyrics_url,
                "lyrics_error": lyrics_error,
                "duration_seconds": info.get("duration"),
                "youtube_title": info.get("title"),
            }
        except Exception as exc:
            errors.append({"source_url": url, "error": f"{type(exc).__name__}: {exc}"})

    tracks = sorted(existing.values(), key=lambda item: str(item.get("id", "")))
    payload = {
        "metadata": {
            "name": "mohim_local_youtube",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_tracks": len(tracks),
            "num_missing_lyrics": sum(not str(item.get("lyrics") or "").strip() for item in tracks),
        },
        "tracks": tracks,
        "errors": errors,
    }
    _write_manifest(manifest_path, payload)
    return payload


def load_local_tracks(manifest_path: str | Path, *, require_lyrics: bool = True) -> list[DaliTrack]:
    """Load the independent JSON schema through the existing processing interface."""

    path = Path(manifest_path).expanduser().resolve()
    payload = _read_manifest(path)
    tracks: list[DaliTrack] = []
    for item in payload["tracks"]:
        lyrics = str(item.get("lyrics") or "").strip()
        if require_lyrics and not lyrics:
            continue
        genres = item.get("genres") or ()
        if isinstance(genres, str):
            genres = [genres]
        tracks.append(
            DaliTrack(
                dali_id=str(item["id"]),
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
