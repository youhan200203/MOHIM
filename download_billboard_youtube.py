#!/usr/bin/env python3
"""Download Billboard/Genius seed audio locally and upload it to Google Drive."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_SEED_NAME = "billboard_pop_genius_seed.json"
DEFAULT_DATASET_NAME = "billboard_pop_dataset"
DEFAULT_DRIVE_REMOTE = "gdrive:MOHIM/billboard_pop_dataset"


def verify_rclone(remote_path: str) -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is required for direct Google Drive uploads. "
            "Install it with: brew install rclone"
        )
    if ":" not in remote_path:
        raise ValueError("--drive-remote must look like gdrive:MOHIM/billboard_pop_dataset")
    remote_name = remote_path.split(":", 1)[0] + ":"
    configured = subprocess.run(
        ["rclone", "listremotes"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if remote_name not in configured:
        raise RuntimeError(
            f"rclone remote {remote_name!r} is not configured. "
            "Run 'rclone config', create a Google Drive remote named 'gdrive', "
            "and sign in once."
        )
    connection = subprocess.run(
        ["rclone", "lsd", remote_name, "--max-depth", "1"],
        capture_output=True,
        text=True,
    )
    if connection.returncode != 0:
        detail = connection.stderr.strip() or "unknown rclone error"
        raise RuntimeError(f"Could not connect to {remote_name}: {detail}")


def upload_checkpoint(dataset_dir: Path, remote_path: str) -> None:
    audio_dir = dataset_dir / "audio"
    manifest_path = dataset_dir / "tracks.json"
    if audio_dir.is_dir() and any(audio_dir.iterdir()):
        subprocess.run(
            [
                "rclone",
                "move",
                str(audio_dir),
                f"{remote_path.rstrip('/')}/audio",
                "--delete-empty-src-dirs",
                "--transfers",
                "4",
                "--checkers",
                "8",
                "--log-level",
                "ERROR",
            ],
            check=True,
        )
    subprocess.run(
        [
            "rclone",
            "copyto",
            str(manifest_path),
            f"{remote_path.rstrip('/')}/tracks.json",
            "--log-level",
            "ERROR",
        ],
        check=True,
    )


def count_seed_tracks(seed_path: Path) -> int:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"JSON must contain a 'tracks' list: {seed_path}")
    return len(tracks)


def load_ingest_function() -> Callable[..., dict[str, Any]]:
    module_path = Path(__file__).resolve().parent / "mohim" / "local_dataset.py"
    spec = importlib.util.spec_from_file_location("mohim_local_dataset", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the existing downloader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ingest_genius_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_SEED_NAME,
        help="Billboard/Genius seed JSON",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_DATASET_NAME,
        help="local output directory before upload",
    )
    parser.add_argument(
        "--drive-remote",
        default=DEFAULT_DRIVE_REMOTE,
        help="rclone destination (default: gdrive:MOHIM/billboard_pop_dataset)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="only process the first N seed tracks; default processes all tracks",
    )
    parser.add_argument("--search-results", type=int, default=8)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.max_tracks is not None and args.max_tracks <= 0:
        parser.error("--max-tracks must be positive")
    if args.search_results <= 0:
        parser.error("--search-results must be positive")
    if args.max_duration_seconds <= 0:
        parser.error("--max-duration-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    seed_path = args.seed.expanduser().resolve()
    if not seed_path.is_file():
        raise FileNotFoundError(f"Seed JSON does not exist: {seed_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")
    verify_rclone(args.drive_remote)
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run: "
            "./.venv-dataset/bin/pip install -r requirements.txt"
        ) from exc

    dataset_dir = args.dataset_dir.expanduser().resolve()
    ingest_genius_seed = load_ingest_function()
    ingest_genius_seed(
        seed_path,
        dataset_dir,
        max_tracks=args.max_tracks,
        search_results=args.search_results,
        max_duration_seconds=args.max_duration_seconds,
        checkpoint_callback=lambda _: upload_checkpoint(dataset_dir, args.drive_remote),
        allow_missing_audio=True,
    )


if __name__ == "__main__":
    main()
