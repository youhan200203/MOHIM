"""Joint Euler sampler for validating dual-stream motif/vocal adapters.

The caller supplies already encoded text conditions and a VAE-encoded motif
seed. Keeping audio/text encoding outside this module makes it usable from a
small research script without adding a Gradio or HTTP surface.
"""

from __future__ import annotations

from typing import Tuple

import torch

from acestep.training_v2.dual_stream import (
    MOTIF_ROLE,
    VOCAL_ROLE,
    DualStreamConditioner,
)


@torch.no_grad()
def sample_dual_stream_latents(
    decoder: torch.nn.Module,
    conditioner: DualStreamConditioner,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    motif_seed_latents: torch.Tensor,
    motif_seed_attention_mask: torch.Tensor,
    output_frames: int,
    latent_dim: int,
    steps: int = 50,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jointly denoise motif and vocal latent streams with Euler integration.

    Returns motif and vocal latents shaped ``[batch, output_frames, latent_dim]``.
    The caller can decode each stream with ACE-Step's VAE and optionally mix
    them only for listening; no full-mix model target is involved.
    """
    if output_frames < 1 or latent_dim < 1 or steps < 1:
        raise ValueError("output_frames, latent_dim, and steps must be positive")
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    shape = (encoder_hidden_states.shape[0], output_frames, latent_dim)
    motif = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    vocal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    stream_mask = torch.ones(shape[:2], device=device, dtype=dtype)
    context = torch.zeros(shape[0], shape[1], latent_dim * 2, device=device, dtype=dtype)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)

    for index in range(steps):
        t = times[index].expand(shape[0])
        dt = times[index + 1] - times[index]
        motif_condition, motif_condition_mask = conditioner.build_condition(
            encoder_hidden_states,
            encoder_attention_mask,
            motif_seed_latents,
            motif_seed_attention_mask,
            vocal,
            stream_mask,
            MOTIF_ROLE,
        )
        vocal_condition, vocal_condition_mask = conditioner.build_condition(
            encoder_hidden_states,
            encoder_attention_mask,
            motif_seed_latents,
            motif_seed_attention_mask,
            motif,
            stream_mask,
            VOCAL_ROLE,
        )
        motif_velocity = decoder(
            hidden_states=motif,
            timestep=t,
            timestep_r=t,
            attention_mask=stream_mask,
            encoder_hidden_states=motif_condition,
            encoder_attention_mask=motif_condition_mask,
            context_latents=context,
        )[0]
        vocal_velocity = decoder(
            hidden_states=vocal,
            timestep=t,
            timestep_r=t,
            attention_mask=stream_mask,
            encoder_hidden_states=vocal_condition,
            encoder_attention_mask=vocal_condition_mask,
            context_latents=context,
        )[0]
        motif = motif + dt * motif_velocity
        vocal = vocal + dt * vocal_velocity

    return motif, vocal
