"""Build the dual-stream JSON contract consumed by the MOHIM ACE-Step fork."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_dual_stream_manifest(
    processed_dir: str | Path,
    output_path: str | Path,
    *,
    caption_template: str = "English pop song with a recurring {motif_stem} motif",
) -> dict[str, Any]:
    root = Path(processed_dir).expanduser().resolve()
    samples: list[dict[str, Any]] = []
    skipped: list[str] = []

    for metadata_path in sorted(root.glob("*/metadata.json")):
        sample_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "accepted":
            continue
        lyrics_path = sample_dir / "lyrics.txt"
        motif_seed = sample_dir / metadata["motif_seed_file"]
        motif_target = sample_dir / metadata["motif_target_file"]
        vocal_target = sample_dir / metadata["vocal_target_file"]
        if not all(path.is_file() for path in (lyrics_path, motif_seed, motif_target, vocal_target)):
            skipped.append(metadata.get("track_id", sample_dir.name))
            continue
        lyrics = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics:
            skipped.append(metadata.get("track_id", sample_dir.name))
            continue

        motif_stem = metadata["motif_stem"]
        samples.append(
            {
                "motif_target_audio": str(motif_target),
                "motif_seed_audio": str(motif_seed),
                "vocal_target_audio": str(vocal_target),
                "audio_path": str(motif_target),
                "filename": f"{metadata['track_id']}_{motif_stem}{motif_target.suffix}",
                "caption": caption_template.format(motif_stem=motif_stem),
                "lyrics": lyrics,
                "bpm": None,
                "keyscale": "",
                "timesignature": "",
                "is_instrumental": False,
                "track_id": metadata["track_id"],
                "artist": metadata.get("artist", ""),
                "title": metadata.get("title", ""),
                "motif_stem": motif_stem,
                "motif_start_sec": metadata["motif_start_sec"],
                "motif_end_sec": metadata["motif_end_sec"],
            }
        )

    manifest = {
        "metadata": {
            "name": "mohim_youtube_dual_stream",
            "num_samples": len(samples),
            "skipped": skipped,
        },
        "samples": samples,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
