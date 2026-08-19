#!/usr/bin/env python3
"""Collect Billboard Pop Airplay candidates and optionally recrawl Genius lyrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import unicodedata
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from getpass import getpass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from urllib.parse import urlparse

import billboard
import lyricsgenius


DEFAULT_CHART_NAME = "pop-songs"
DEFAULT_START_DATE = date(2008, 1, 1)
DEFAULT_OUTPUT_NAME = "billboard_pop_genius_seed.json"
DEFAULT_RECRAWL_OUTPUT_NAME = "billboard_pop_genius_seed_recrawled.json"
GENIUS_FIELDS = {
    "id",
    "genius_id",
    "genius_artist",
    "genius_title",
    "lyrics",
    "lyrics_source",
    "lyrics_source_url",
    "candidate_url",
    "candidate_source",
}
T = TypeVar("T")


def with_retries(
    operation: Callable[[], T],
    *,
    retries: int,
    delay: float,
    label: str,
) -> T:
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception:
            if attempt >= retries:
                raise
            wait_seconds = delay * (2**attempt)
            print(
                f"{label} failed; retrying in {wait_seconds:g}s "
                f"({attempt + 1}/{retries})",
                flush=True,
            )
            if wait_seconds:
                time.sleep(wait_seconds)
    raise RuntimeError(f"{label} failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def previous_week(value: str) -> str:
    return (parse_iso_date(value) - timedelta(days=7)).isoformat()


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def candidate_key(title: str, artist: str) -> str:
    return f"{normalized_text(artist)}::{normalized_text(title)}"


def title_matches(expected: str, actual: str) -> bool:
    expected_normalized = normalized_text(expected)
    actual_normalized = normalized_text(actual)
    if not expected_normalized or not actual_normalized:
        return False
    if expected_normalized == actual_normalized:
        return True
    return SequenceMatcher(None, expected_normalized, actual_normalized).ratio() >= 0.82


def artist_matches(expected: str, actual: str) -> bool:
    expected_normalized = normalized_text(expected)
    actual_normalized = normalized_text(actual)
    if not expected_normalized or not actual_normalized:
        return False
    if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
        return True
    ignored = {"and", "feat", "featuring", "ft", "the", "with", "x"}
    expected_tokens = set(expected_normalized.split()) - ignored
    actual_tokens = set(actual_normalized.split()) - ignored
    if not expected_tokens or not actual_tokens:
        return False
    overlap = len(expected_tokens & actual_tokens)
    return overlap / min(len(expected_tokens), len(actual_tokens)) >= 0.5


def clean_lyrics(value: Any) -> str:
    lyrics = str(value or "").strip()
    _, marker, remainder = lyrics.partition("Read More")
    if marker:
        return remainder.lstrip(" \t\r\n\u00a0…").strip()
    _, marker, remainder = lyrics.partition("Lyrics")
    if marker:
        lyrics = remainder.lstrip(" \t\r\n\u00a0…")
    return lyrics.strip()


def read_seed_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    if not isinstance(payload.get("tracks"), list):
        raise ValueError(f"JSON must contain a tracks list: {path}")
    if not isinstance(payload.get("candidates", []), list):
        raise ValueError(f"JSON candidates must be a list: {path}")
    return dict(payload)


def candidate_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    candidate = {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in GENIUS_FIELDS
    }
    title = str(candidate.get("title") or "").strip()
    artist = str(candidate.get("artist") or "").strip()
    if not title or not artist:
        raise ValueError("Every Billboard record must contain artist and title")
    candidate["candidate_key"] = str(
        candidate.get("candidate_key") or candidate_key(title, artist)
    )
    return candidate


def new_recrawl_payload(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for record in [*source["tracks"], *source.get("candidates", [])]:
        candidate = candidate_from_record(record)
        unique[candidate["candidate_key"]] = candidate

    metadata = deepcopy(source.get("metadata") or {})
    metadata.update(
        {
            "name": "mohim_billboard_pop_genius_seed_recrawled",
            "generated_at": utc_now(),
            "recrawl_source": str(source_path),
            "interrupted": False,
        }
    )
    payload = {
        "metadata": metadata,
        "tracks": [],
        "candidates": list(unique.values()),
        "errors": [],
    }
    update_metadata(payload)
    return payload


def backup_existing_output(path: Path) -> Path | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def verify_lyrics_cleaner() -> None:
    assert clean_lyrics(
        "12 ContributorsSong LyricsDescription Read More [Verse 1]\nActual Lyrics line"
    ) == "[Verse 1]\nActual Lyrics line"
    assert clean_lyrics(
        "12 ContributorsTranslationsSong Lyrics[Intro]\nFirst line"
    ) == "[Intro]\nFirst line"
    assert clean_lyrics("[Verse 1]\nlowercase lyrics stay") == (
        "[Verse 1]\nlowercase lyrics stay"
    )


def new_payload(chart_name: str, start_date: date) -> dict[str, Any]:
    return {
        "metadata": {
            "name": "mohim_billboard_pop_genius_seed",
            "generated_at": utc_now(),
            "chart_name": chart_name,
            "start_date": start_date.isoformat(),
            "source_url": f"https://www.billboard.com/charts/{chart_name}/",
            "backfill_complete": False,
            "next_backfill_date": None,
            "processed_chart_dates": [],
            "interrupted": False,
        },
        "tracks": [],
        "candidates": [],
        "errors": [],
    }


def load_payload(path: Path, chart_name: str, start_date: date) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return new_payload(chart_name, start_date)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Output JSON is not empty but is invalid: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Output JSON must contain an object: {path}")
    payload.setdefault("metadata", {})
    payload.setdefault("tracks", [])
    payload.setdefault("candidates", [])
    payload.setdefault("errors", [])
    if not all(isinstance(payload[key], list) for key in ("tracks", "candidates", "errors")):
        raise ValueError(f"Output JSON has an incompatible structure: {path}")

    metadata = payload["metadata"]
    existing_chart = metadata.get("chart_name")
    if existing_chart and existing_chart != chart_name:
        raise ValueError(
            f"Existing output uses chart {existing_chart!r}; choose another --output "
            f"for {chart_name!r}"
        )
    existing_start = parse_iso_date(metadata.get("start_date", start_date.isoformat()))
    if start_date < existing_start:
        metadata["start_date"] = start_date.isoformat()
        metadata["backfill_complete"] = False
        processed = metadata.get("processed_chart_dates") or []
        metadata["next_backfill_date"] = previous_week(min(processed)) if processed else None
    elif start_date > existing_start:
        print(
            f"existing output already contains dates from {existing_start}; "
            f"keeping that earlier start date",
            flush=True,
        )
    metadata.update(
        {
            "chart_name": chart_name,
            "source_url": f"https://www.billboard.com/charts/{chart_name}/",
        }
    )
    metadata.setdefault("start_date", start_date.isoformat())
    metadata.setdefault("backfill_complete", False)
    metadata.setdefault("next_backfill_date", None)
    metadata.setdefault("processed_chart_dates", [])
    metadata.setdefault("interrupted", False)
    return payload


def update_metadata(payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    processed = sorted(set(metadata.get("processed_chart_dates") or []))
    metadata["processed_chart_dates"] = processed
    metadata.update(
        {
            "updated_at": utc_now(),
            "num_chart_weeks": len(processed),
            "num_tracks": len(payload["tracks"]),
            "num_pending_candidates": len(payload["candidates"]),
            "num_errors": len(payload["errors"]),
        }
    )
    if processed:
        metadata["oldest_chart_date"] = processed[0]
        metadata["latest_chart_date"] = processed[-1]


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    update_metadata(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def record_error(
    payload: dict[str, Any],
    *,
    stage: str,
    identity: str,
    details: dict[str, Any],
    exc: Exception,
) -> None:
    payload["errors"] = [
        error
        for error in payload["errors"]
        if not (error.get("stage") == stage and error.get("identity") == identity)
    ]
    payload["errors"].append(
        {
            "stage": stage,
            "identity": identity,
            **details,
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at": utc_now(),
        }
    )


def clear_error(payload: dict[str, Any], stage: str, identity: str) -> None:
    payload["errors"] = [
        error
        for error in payload["errors"]
        if not (error.get("stage") == stage and error.get("identity") == identity)
    ]


def chart_record_indexes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for collection_name in ("tracks", "candidates"):
        for record in payload[collection_name]:
            key = str(record.get("candidate_key") or "")
            if key:
                index[key] = record
    return index


def upsert_chart(
    payload: dict[str, Any],
    chart: billboard.ChartData,
    chart_name: str,
) -> None:
    chart_date = str(chart.date)
    records = chart_record_indexes(payload)
    for entry in chart.entries:
        title = str(entry.title or "").strip()
        artist = str(entry.artist or "").strip()
        if not title or not artist:
            continue
        key = candidate_key(title, artist)
        record = records.get(key)
        rank = int(entry.rank)
        if record is None:
            record = {
                "candidate_key": key,
                "artist": artist,
                "title": title,
                "genres": ["Pop"],
                "language": "English",
                "billboard_chart": chart_name,
                "billboard_peak_rank": rank,
                "first_chart_date": chart_date,
                "last_chart_date": chart_date,
                "chart_dates": [chart_date],
                "chart_weeks": 1,
            }
            payload["candidates"].append(record)
            records[key] = record
        else:
            dates = set(record.get("chart_dates") or [])
            dates.add(chart_date)
            record["chart_dates"] = sorted(dates)
            record["chart_weeks"] = len(record["chart_dates"])
            record["first_chart_date"] = min(record["chart_dates"])
            record["last_chart_date"] = max(record["chart_dates"])
            old_peak = int(record.get("billboard_peak_rank") or rank)
            record["billboard_peak_rank"] = min(old_peak, rank)


def chart_previous_date(chart: billboard.ChartData) -> str:
    value = str(getattr(chart, "previousDate", "") or "").strip()
    return value or previous_week(str(chart.date))


def fetch_chart(
    chart_name: str,
    chart_date: str | None,
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> billboard.ChartData:
    label = f"Billboard {chart_name} chart {chart_date or 'latest'}"
    return with_retries(
        lambda: billboard.ChartData(
            chart_name,
            date=chart_date,
            timeout=timeout,
            max_retries=retries,
        ),
        retries=retries,
        delay=retry_delay,
        label=label,
    )


def collect_billboard_charts(
    payload: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    metadata = payload["metadata"]
    start_date = parse_iso_date(metadata["start_date"])
    processed = set(metadata.get("processed_chart_dates") or [])
    starting_with_no_processed_weeks = not processed

    latest = fetch_chart(
        args.chart,
        None,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    chart = latest
    while parse_iso_date(str(chart.date)) >= start_date and str(chart.date) not in processed:
        upsert_chart(payload, chart, args.chart)
        processed.add(str(chart.date))
        metadata["processed_chart_dates"] = sorted(processed)
        next_date = chart_previous_date(chart)
        if metadata.get("next_backfill_date") is None:
            metadata["next_backfill_date"] = next_date
        clear_error(payload, "billboard", str(chart.date))
        write_payload(output, payload)
        print(
            f"[Billboard {chart.date}] entries={len(chart.entries)} "
            f"unique={len(payload['tracks']) + len(payload['candidates'])}",
            flush=True,
        )
        if args.chart_delay:
            time.sleep(args.chart_delay)
        if starting_with_no_processed_weeks:
            break
        if next_date in processed:
            break
        chart = fetch_chart(
            args.chart,
            next_date,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )

    if metadata.get("backfill_complete"):
        return
    next_backfill = metadata.get("next_backfill_date")
    if not next_backfill:
        next_backfill = chart_previous_date(latest)
        metadata["next_backfill_date"] = next_backfill

    while parse_iso_date(next_backfill) >= start_date:
        try:
            chart = fetch_chart(
                args.chart,
                next_backfill,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
        except Exception as exc:
            record_error(
                payload,
                stage="billboard",
                identity=next_backfill,
                details={"chart_name": args.chart, "requested_date": next_backfill},
                exc=exc,
            )
            write_payload(output, payload)
            raise
        actual_date = str(chart.date)
        if parse_iso_date(actual_date) < start_date:
            break
        if actual_date not in processed:
            upsert_chart(payload, chart, args.chart)
            processed.add(actual_date)
            metadata["processed_chart_dates"] = sorted(processed)
        clear_error(payload, "billboard", next_backfill)
        following_date = chart_previous_date(chart)
        if following_date == next_backfill:
            following_date = previous_week(next_backfill)
        metadata["next_backfill_date"] = following_date
        write_payload(output, payload)
        print(
            f"[Billboard {actual_date}] entries={len(chart.entries)} "
            f"unique={len(payload['tracks']) + len(payload['candidates'])}",
            flush=True,
        )
        if args.chart_delay:
            time.sleep(args.chart_delay)
        next_backfill = following_date

    metadata["backfill_complete"] = True
    metadata["next_backfill_date"] = None
    write_payload(output, payload)


def enrich_with_genius(
    payload: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    token: str,
) -> None:
    genius = lyricsgenius.Genius(
        token,
        remove_section_headers=False,
        skip_non_songs=True,
        timeout=args.timeout,
        retries=args.retries,
    )
    attempted = 0
    for candidate in list(payload["candidates"]):
        if args.max_genius_lookups and attempted >= args.max_genius_lookups:
            break
        attempted += 1
        key = candidate["candidate_key"]
        artist = candidate["artist"]
        title = candidate["title"]
        try:
            song = with_retries(
                lambda: genius.search_song(title, artist, get_full_info=False),
                retries=args.retries,
                delay=args.retry_delay,
                label=f"Genius search for {artist} - {title}",
            )
            if song is None:
                raise RuntimeError("Genius returned no search result")
            found_title = str(getattr(song, "title", "") or "").strip()
            found_artist = str(getattr(song, "artist", "") or "").strip()
            if not title_matches(title, found_title) or not artist_matches(artist, found_artist):
                raise RuntimeError(
                    f"Genius result mismatch: {found_artist} - {found_title}"
                )
            lyrics = clean_lyrics(getattr(song, "lyrics", ""))
            if not lyrics:
                raise RuntimeError("Genius returned no lyrics")
            genius_url = str(getattr(song, "url", "") or "").strip()
            slug = urlparse(genius_url).path.strip("/").removesuffix("-lyrics")
            track = {
                "id": slug or key,
                "genius_id": getattr(song, "id", None),
                **candidate,
                "genius_artist": found_artist,
                "genius_title": found_title,
                "lyrics": lyrics,
                "lyrics_source": "genius",
                "lyrics_source_url": genius_url,
                "candidate_url": genius_url,
                "candidate_source": f"billboard_chart:{args.chart}",
            }
            payload["tracks"].append(track)
            payload["candidates"].remove(candidate)
            clear_error(payload, "genius", key)
            status = f"saved ({len(payload['tracks'])} tracks)"
        except Exception as exc:
            record_error(
                payload,
                stage="genius",
                identity=key,
                details={"artist": artist, "title": title},
                exc=exc,
            )
            status = f"failed: {type(exc).__name__}: {exc}"
        write_payload(output, payload)
        print(f"[Genius {attempted}] {artist} - {title}: {status}", flush=True)
        if args.delay:
            time.sleep(args.delay)


def export_seed(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    payload = load_payload(output, args.chart, args.start_date)
    payload["metadata"]["interrupted"] = False
    write_payload(output, payload)
    print(f"output JSON ready: {output}", flush=True)
    print(
        f"resume: {len(payload['tracks'])} tracks, "
        f"{len(payload['candidates'])} pending candidates",
        flush=True,
    )
    try:
        collect_billboard_charts(payload, output, args)
    except KeyboardInterrupt:
        payload["metadata"]["interrupted"] = True
        print("interrupted; completed Billboard weeks remain saved", flush=True)
    finally:
        write_payload(output, payload)
    return payload


def recrawl_seed(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.recrawl_source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source JSON does not exist: {source_path}")
    if source_path == output_path:
        raise ValueError("--output must be different from --recrawl-source")

    verify_lyrics_cleaner()
    backup = backup_existing_output(output_path)
    if backup is not None:
        print(f"output backup: {backup}", flush=True)

    if output_path.is_file() and output_path.stat().st_size > 0:
        payload = read_seed_payload(output_path)
    else:
        payload = new_recrawl_payload(read_seed_payload(source_path), source_path)
        write_payload(output_path, payload)

    print(
        f"resume: {len(payload['tracks'])} saved, "
        f"{len(payload['candidates'])} remaining",
        flush=True,
    )
    token = os.environ.get("GENIUS_ACCESS_TOKEN") or getpass("GENIUS_ACCESS_TOKEN: ")
    if not token:
        raise ValueError("GENIUS_ACCESS_TOKEN is required")

    payload["metadata"]["interrupted"] = False
    try:
        enrich_with_genius(payload, output_path, args, token)
    except KeyboardInterrupt:
        payload["metadata"]["interrupted"] = True
        print("interrupted; completed lyrics remain saved", flush=True)
    finally:
        write_payload(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON; defaults depend on whether --recrawl-source is used",
    )
    parser.add_argument(
        "--recrawl-source",
        type=Path,
        help="skip Billboard collection and rebuild Genius lyrics from this seed JSON",
    )
    parser.add_argument("--chart", default=DEFAULT_CHART_NAME)
    parser.add_argument("--start-date", type=parse_iso_date, default=DEFAULT_START_DATE)
    parser.add_argument("--delay", type=float, default=1.0, help="delay between Genius lookups")
    parser.add_argument(
        "--chart-delay",
        type=float,
        default=0.25,
        help="delay between Billboard chart requests",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--max-genius-lookups",
        type=int,
        default=0,
        help="maximum Genius searches this run; 0 means all pending candidates",
    )
    args = parser.parse_args()
    if args.start_date > date.today():
        parser.error("--start-date cannot be in the future")
    if min(args.delay, args.chart_delay, args.retry_delay, args.timeout) < 0:
        parser.error("delays and timeout cannot be negative")
    if args.retries < 0 or args.max_genius_lookups < 0:
        parser.error("--retries and --max-genius-lookups cannot be negative")
    if args.output is None:
        default_name = (
            DEFAULT_RECRAWL_OUTPUT_NAME
            if args.recrawl_source is not None
            else DEFAULT_OUTPUT_NAME
        )
        args.output = Path(__file__).resolve().parent / default_name
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = recrawl_seed(arguments) if arguments.recrawl_source else export_seed(arguments)
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
