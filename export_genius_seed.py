#!/usr/bin/env python3
"""Create genius_pop_seed.json on a local machine; never downloads audio."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import lyricsgenius
import requests
from bs4 import BeautifulSoup


GENIUS_BASE_URL = "https://genius.com"
HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}


def iter_tag_candidates(
    session: requests.Session,
    tag: str,
    max_pages: int,
    timeout: float,
) -> Iterable[dict[str, str]]:
    tag_slug = re.sub(r"[^a-z0-9]+", "-", tag.casefold()).strip("-")
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        response = session.get(f"{GENIUS_BASE_URL}/tags/{tag_slug}/all?page={page}", timeout=timeout)
        response.raise_for_status()
        links = BeautifulSoup(response.text, "html.parser").select("li.search_result a.song_link")
        if not links:
            break
        for link in links:
            title_node = link.select_one(".song_title")
            artist_nodes = link.select(".primary_artist_name")
            title = title_node.get_text(" ", strip=True) if title_node else ""
            artist = ", ".join(
                dict.fromkeys(node.get_text(" ", strip=True) for node in artist_nodes)
            )
            url = urljoin(GENIUS_BASE_URL, str(link.get("href") or "").strip())
            if title and artist and url not in seen:
                seen.add(url)
                yield {"artist": artist, "title": title, "candidate_url": url}


def write_seed(path: Path, payload: dict[str, Any]) -> None:
    payload["metadata"].update(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_tracks": len(payload["tracks"]),
            "num_errors": len(payload["errors"]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_seed(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("GENIUS_ACCESS_TOKEN") or getpass("GENIUS_ACCESS_TOKEN: ")
    if not token:
        raise ValueError("GENIUS_ACCESS_TOKEN is required")

    genius = lyricsgenius.Genius(token)
    genius.remove_section_headers = False
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
    session = requests.Session()
    session.headers.update(HEADERS)
    output = args.output.expanduser().resolve()
    payload: dict[str, Any] = {
        "metadata": {
            "name": "mohim_genius_seed",
            "tag": args.tag,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "interrupted": False,
        },
        "tracks": [],
        "errors": [],
    }
    write_seed(output, payload)
    print(f"seed JSON initialized: {output}", flush=True)

    try:
        for attempted, candidate in enumerate(
            iter_tag_candidates(session, args.tag, args.max_pages, args.timeout), 1
        ):
            if len(payload["tracks"]) >= args.max_tracks:
                break
            try:
                song = genius.search_song(candidate["title"], candidate["artist"])
                lyrics = str(getattr(song, "lyrics", "") or "").strip()
                if not lyrics:
                    raise RuntimeError("LyricsGenius returned no lyrics")
                source_url = str(getattr(song, "url", "") or candidate["candidate_url"])
                slug = urlparse(candidate["candidate_url"]).path.strip("/").removesuffix("-lyrics")
                payload["tracks"].append(
                    {
                        "id": slug,
                        "artist": candidate["artist"],
                        "title": candidate["title"],
                        "genres": [args.tag.title()],
                        "language": "English",
                        "lyrics": lyrics,
                        "lyrics_source": "genius",
                        "lyrics_source_url": source_url,
                        "candidate_url": candidate["candidate_url"],
                    }
                )
                status = f"saved ({len(payload['tracks'])}/{args.max_tracks})"
            except Exception as exc:
                payload["errors"].append({**candidate, "error": f"{type(exc).__name__}: {exc}"})
                status = f"failed: {type(exc).__name__}: {exc}"
            write_seed(output, payload)
            print(f"[{attempted}] {candidate['artist']} - {candidate['title']}: {status}", flush=True)
            if args.delay:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        payload["metadata"]["interrupted"] = True
        print("interrupted; completed tracks remain saved", flush=True)
    finally:
        session.close()
        write_seed(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "genius_pop_seed.json",
    )
    parser.add_argument("--tag", default="pop")
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_tracks <= 0 or args.max_pages <= 0:
        parser.error("--max-tracks and --max-pages must be positive")
    return args


if __name__ == "__main__":
    result = export_seed(parse_args())
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
