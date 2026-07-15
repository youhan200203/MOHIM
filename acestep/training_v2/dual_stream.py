"""Dual-stream conditioning helpers for motif and vocal stem training.

The helpers keep ACE-Step's DiT unchanged. They turn the motif seed and the
other stream's current noisy latent into cross-attention tokens, so both
flow-matching branches can be denoised jointly with shared DiT weights.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MOTIF_ROLE = 0
VOCAL_ROLE = 1


def masked_adaptive_pool(
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool variable-length latent sequences without averaging padded frames."""
    if latents.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("latents must be [B, T, C] and attention_mask must be [B, T]")
    if latents.shape[:2] != attention_mask.shape:
        raise ValueError("latent and attention-mask time dimensions must match")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    token_count = min(max_tokens, latents.shape[1])
    mask = attention_mask.to(dtype=latents.dtype).unsqueeze(-1)
    values = F.adaptive_avg_pool1d((latents * mask).transpose(1, 2), token_count)
    weights = F.adaptive_avg_pool1d(mask.transpose(1, 2), token_count)
    pooled = (values / weights.clamp_min(torch.finfo(latents.dtype).eps)).transpose(1, 2)
    return pooled, weights.squeeze(1).gt(0)


class DualStreamConditioner(nn.Module):
    """Build role-aware motif and peer-stream condition tokens for a DiT."""

    def __init__(self, latent_dim: int, condition_dim: int, max_tokens: int = 64) -> None:
        """Initialize projections for motif seed and peer-stream latents."""
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.max_tokens = max_tokens
        self.seed_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, condition_dim))
        self.peer_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, condition_dim))
        self.role_embedding = nn.Embedding(2, condition_dim)
        self.seed_type_embedding = nn.Parameter(torch.zeros(1, 1, condition_dim))
        self.peer_type_embedding = nn.Parameter(torch.zeros(1, 1, condition_dim))

    def build_condition(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        motif_seed_latents: torch.Tensor,
        motif_seed_attention_mask: torch.Tensor,
        peer_latents: torch.Tensor,
        peer_attention_mask: torch.Tensor,
        role: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prefix base text conditions with motif-seed and peer-stream tokens."""
        if role not in (MOTIF_ROLE, VOCAL_ROLE):
            raise ValueError(f"unknown dual-stream role: {role}")
        seed_tokens, seed_mask = masked_adaptive_pool(
            motif_seed_latents, motif_seed_attention_mask, self.max_tokens
        )
        peer_tokens, peer_mask = masked_adaptive_pool(
            peer_latents, peer_attention_mask, self.max_tokens
        )
        seed_tokens = self.seed_projection(seed_tokens) + self.seed_type_embedding
        peer_tokens = self.peer_projection(peer_tokens) + self.peer_type_embedding
        batch_size = encoder_hidden_states.shape[0]
        role_ids = torch.full(
            (batch_size,), role, device=encoder_hidden_states.device, dtype=torch.long
        )
        role_token = self.role_embedding(role_ids).unsqueeze(1)
        role_mask = torch.ones(batch_size, 1, device=encoder_attention_mask.device, dtype=torch.bool)
        condition = torch.cat([role_token, seed_tokens, peer_tokens, encoder_hidden_states], dim=1)
        mask = torch.cat(
            [role_mask, seed_mask, peer_mask, encoder_attention_mask.to(torch.bool)], dim=1
        )
        return condition, mask.to(dtype=encoder_attention_mask.dtype)
