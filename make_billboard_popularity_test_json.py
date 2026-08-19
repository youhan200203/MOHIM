#!/usr/bin/env python3
"""Order downloaded Billboard tracks by chart popularity for motif tests."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKS_PATH = ROOT / "billboard_pop_dataset" / "tracks.json"
DEFAULT_SEED_PATH = ROOT / "billboard_pop_genius_seed_recrawled.json"
DEFAULT_OUTPUT_PATH = ROOT / "billboard_pop_dataset" / "tracks_popular.json"


def read_tracks_payload(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tracks"), list):
        raise ValueError(f"{label} JSON must contain a 'tracks' list: {path}")
    return dict(payload)


def integer_field(record: Mapping[str, Any], field: str, seed_id: str) -> int:
    value = record.get(field)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Seed {seed_id!r} has no valid {field!r}: {value!r}"
        ) from exc
    if number < 0 or (field == "billboard_peak_rank" and number == 0):
        raise ValueError(f"Seed {seed_id!r} has invalid {field!r}: {value!r}")
    return number


def popularity_index(seed_payload: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    index: dict[str, tuple[int, int]] = {}
    for record in seed_payload["tracks"]:
        if not isinstance(record, Mapping):
            raise ValueError("Every seed track must be a JSON object")
        seed_id = str(record.get("id") or "").strip()
        if not seed_id:
            raise ValueError("Every seed track must contain a non-empty 'id'")
        if seed_id in index:
            raise ValueError(f"Duplicate seed track id: {seed_id!r}")
        index[seed_id] = (
            integer_field(record, "chart_weeks", seed_id),
            integer_field(record, "billboard_peak_rank", seed_id),
        )
    return index


def rank_downloaded_tracks(
    tracks: list[Any],
    popularity: Mapping[str, tuple[int, int]],
    top: int,
) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, int, Mapping[str, Any]]] = []
    missing: list[str] = []
    for original_index, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            raise ValueError("Every downloaded track must be a JSON object")
        seed_id = str(track.get("seed_id") or "").strip()
        if not seed_id or seed_id not in popularity:
            missing.append(seed_id or "<missing seed_id>")
            continue
        chart_weeks, peak_rank = popularity[seed_id]
        ranked.append((-chart_weeks, peak_rank, original_index, track))

    if missing:
        preview = ", ".join(repr(value) for value in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"Could not match {len(missing)} downloaded track(s) to seed ids: "
            f"{preview}{suffix}"
        )

    ranked.sort(key=lambda item: item[:3])
    return [deepcopy(dict(item[3])) for item in ranked[:top]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=DEFAULT_TRACKS_PATH,
        help="downloaded tracks.json input",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="recrawled Billboard/Genius seed JSON input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="separate popularity-ordered JSON output",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="maximum number of downloaded tracks to write (default: 100)",
    )
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")
    return args


def main() -> None:
    args = parse_args()
    tracks_path = args.tracks.expanduser().resolve()
    seed_path = args.seed.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path in {tracks_path, seed_path}:
        raise ValueError("--output must differ from both input JSON paths")

    tracks_payload = read_tracks_payload(tracks_path, "Downloaded tracks")
    seed_payload = read_tracks_payload(seed_path, "Recrawled seed")
    ranked_tracks = rank_downloaded_tracks(
        tracks_payload["tracks"], popularity_index(seed_payload), args.top
    )

    output_payload = deepcopy(tracks_payload)
    output_payload["tracks"] = ranked_tracks
    metadata = output_payload.get("metadata")
    if isinstance(metadata, dict) and "num_tracks" in metadata:
        metadata["num_tracks"] = len(ranked_tracks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(ranked_tracks)} downloaded tracks: {output_path}")


if __name__ == "__main__":
    main()
