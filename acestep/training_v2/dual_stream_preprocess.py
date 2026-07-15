"""Audio-stem encoding helpers used by dual-stream preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
from acestep.training_v2.preprocess_vae import TARGET_SR, tiled_vae_encode


DUAL_STREAM_AUDIO_FIELDS = (
    "motif_seed_audio",
    "motif_target_audio",
    "vocal_target_audio",
)


def encode_stem_latents(
    path: str,
    vae: Any,
    dtype: torch.dtype,
    max_duration: float,
) -> torch.Tensor:
    """Load one aligned stem and return its VAE latent sequence on CPU."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Dual-stream stem not found: {audio_path}")
    audio, _ = load_audio_stereo(str(audio_path), TARGET_SR, max_duration)
    audio = audio.unsqueeze(0).to(device=next(vae.parameters()).device, dtype=vae.dtype)
    with torch.no_grad():
        return tiled_vae_encode(vae, audio, dtype).squeeze(0).cpu()
