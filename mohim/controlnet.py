"""ControlNet-Transformer components for melody-conditioned ACE-Step training.

The implementation follows the ControlNet-Transformer topology used by
PixArt-delta and by *Editing Music with Melody and Text*: the pretrained DiT is
frozen, its first N blocks are copied, and zero-initialized linear projections
inject the copied-block outputs into the frozen backbone.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


MIDDLE_C_HZ = 261.2
MIDI_ZERO_HZ = 8.175798915643707
TOPK_CQT_SCHEMA = "topk_cqt_stereo_128bins_top4_hop512_highpass_middle_c_v1"


def _load_repeated_stereo(
    audio_path: str | Path,
    *,
    sample_rate: int,
    duration_seconds: float,
) -> np.ndarray:
    """Load a motif, force stereo, and tile it to the requested duration."""

    import librosa

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if audio.shape[-1] == 0:
        raise ValueError(f"Motif audio is empty: {audio_path}")

    target_samples = max(1, round(duration_seconds * sample_rate))
    repeats = math.ceil(target_samples / audio.shape[-1])
    return np.tile(audio, (1, repeats))[:, :target_samples]


def extract_repeated_topk_cqt(
    audio_path: str | Path,
    *,
    duration_seconds: float,
    sample_rate: int = 48_000,
    hop_length: int = 512,
    n_bins: int = 128,
    top_k: int = 4,
) -> torch.Tensor:
    """Return the paper's interleaved stereo top-k CQT pitch indices.

    The output is time-major with shape ``[frames, 2 * top_k]``.  Each value is
    an absolute CQT-bin index in ``[0, n_bins)``.  A second-order high-pass
    filter at Middle C is applied before CQT extraction, matching the paper.
    """

    import librosa
    from scipy.signal import butter, sosfiltfilt

    if top_k <= 0 or top_k > n_bins:
        raise ValueError(f"top_k must be in [1, {n_bins}], got {top_k}")
    audio = _load_repeated_stereo(
        audio_path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    sos = butter(2, MIDDLE_C_HZ, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, audio, axis=-1).astype(np.float32, copy=False)

    ranked_channels: list[np.ndarray] = []
    for channel in filtered:
        magnitude = np.abs(
            librosa.cqt(
                channel,
                sr=sample_rate,
                hop_length=hop_length,
                fmin=MIDI_ZERO_HZ,
                n_bins=n_bins,
                bins_per_octave=12,
            )
        )
        candidates = np.argpartition(magnitude, -top_k, axis=0)[-top_k:]
        candidate_values = np.take_along_axis(magnitude, candidates, axis=0)
        order = np.argsort(candidate_values, axis=0)[::-1]
        ranked_channels.append(np.take_along_axis(candidates, order, axis=0).T)

    frame_count = min(channel.shape[0] for channel in ranked_channels)
    left, right = (channel[:frame_count] for channel in ranked_channels)
    interleaved = np.empty((frame_count, top_k * 2), dtype=np.int64)
    interleaved[:, 0::2] = left
    interleaved[:, 1::2] = right
    return torch.from_numpy(interleaved)


class TopKCQTMelodyEncoder(nn.Module):
    """Pitch embeddings and 1-D convolutions from the music ControlNet paper."""

    def __init__(
        self,
        hidden_size: int,
        *,
        n_bins: int = 128,
        stereo_top_k: int = 8,
        pitch_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.stereo_top_k = stereo_top_k
        self.pitch_embedding = nn.Embedding(n_bins, pitch_embedding_dim)
        input_channels = stereo_top_k * pitch_embedding_dim
        self.convolutions = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size // 2, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, pitch_indices: torch.Tensor, target_length: int) -> torch.Tensor:
        if pitch_indices.ndim != 3 or pitch_indices.shape[-1] != self.stereo_top_k:
            raise ValueError(
                "pitch_indices must have shape "
                f"[batch, frames, {self.stereo_top_k}], got {tuple(pitch_indices.shape)}"
            )
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        if torch.any(pitch_indices < 0) or torch.any(pitch_indices >= self.n_bins):
            raise ValueError(f"pitch indices must be in [0, {self.n_bins})")

        embedded = self.pitch_embedding(pitch_indices.long())
        batch, frames, pitches, width = embedded.shape
        hidden = embedded.reshape(batch, frames, pitches * width).transpose(1, 2)
        hidden = self.convolutions(hidden)
        if hidden.shape[-1] != target_length:
            hidden = F.interpolate(hidden, size=target_length, mode="linear", align_corners=False)
        return hidden.transpose(1, 2)


class ControlTransformerBlock(nn.Module):
    """A pretrained DiT block copy with zero-initialized input/output bridges."""

    def __init__(self, base_block: nn.Module, hidden_size: int, block_index: int) -> None:
        super().__init__()
        self.block_index = block_index
        self.copied_block = copy.deepcopy(base_block)
        for parameter in self.copied_block.parameters():
            parameter.requires_grad_(True)
        self.before_proj = nn.Linear(hidden_size, hidden_size) if block_index == 0 else None
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        if self.before_proj is not None:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def prepare_first(self, frozen_hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.before_proj is None:
            raise RuntimeError("prepare_first is only valid for the first copied block")
        return frozen_hidden + self.before_proj(condition)


def _padding_mask(values: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
    valid = values[:, :key_length].to(torch.bool)
    return valid[:, None, None, :].expand(-1, 1, query_length, -1)


def _additive_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    result = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
    return result.masked_fill(~valid, torch.finfo(dtype).min)


class AceStepDiTControlNet(nn.Module):
    """Paper-faithful ControlNet-Transformer wrapper for an ACE-Step DiT decoder."""

    def __init__(
        self,
        base_decoder: nn.Module,
        *,
        copy_blocks: int = 12,
        pitch_embedding_dim: int = 64,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        total_blocks = len(base_decoder.layers)
        if copy_blocks <= 0 or copy_blocks >= total_blocks:
            raise ValueError(
                f"copy_blocks must be between 1 and {total_blocks - 1}, got {copy_blocks}"
            )
        hidden_size = int(base_decoder.config.hidden_size)
        self.base_decoder = base_decoder
        self.copy_blocks = copy_blocks
        self.gradient_checkpointing = gradient_checkpointing
        self.melody_encoder = TopKCQTMelodyEncoder(
            hidden_size,
            pitch_embedding_dim=pitch_embedding_dim,
        )
        self.control_blocks = nn.ModuleList(
            ControlTransformerBlock(base_decoder.layers[index], hidden_size, index)
            for index in range(copy_blocks)
        )
        for parameter in self.base_decoder.parameters():
            parameter.requires_grad_(False)
        self.base_decoder.eval()

    def train(self, mode: bool = True) -> "AceStepDiTControlNet":
        super().train(mode)
        self.base_decoder.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def control_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ace_step_dit_controlnet_topk_cqt_v1",
            "copy_blocks": self.copy_blocks,
            "melody_encoder": self.melody_encoder.state_dict(),
            "control_blocks": self.control_blocks.state_dict(),
        }

    def load_control_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        if int(state["copy_blocks"]) != self.copy_blocks:
            raise ValueError(
                f"Checkpoint has {state['copy_blocks']} copied blocks, model has {self.copy_blocks}"
            )
        self.melody_encoder.load_state_dict(state["melody_encoder"], strict=strict)
        self.control_blocks.load_state_dict(state["control_blocks"], strict=strict)

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        position_embeddings: Any,
        timestep_projection: torch.Tensor,
        self_attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        def call(current: torch.Tensor) -> torch.Tensor:
            return layer(
                current,
                position_embeddings,
                timestep_projection,
                self_attention_mask,
                position_ids,
                None,
                False,
                False,
                None,
                encoder_hidden_states,
                encoder_attention_mask,
            )[0]

        if self.gradient_checkpointing and self.training:
            return checkpoint(call, hidden, use_reentrant=False)
        return call(hidden)

    def _attention_masks(
        self,
        hidden: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor | None]:
        batch, sequence_length, _ = hidden.shape
        encoder_length = 0 if encoder_attention_mask is None else encoder_attention_mask.shape[-1]
        implementation = getattr(self.base_decoder.config, "_attn_implementation", "eager")

        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                batch, sequence_length, device=hidden.device, dtype=torch.bool
            )
        else:
            audio_attention_mask = audio_attention_mask.to(device=hidden.device, dtype=torch.float32)
            audio_attention_mask = F.max_pool1d(
                audio_attention_mask.unsqueeze(1),
                kernel_size=self.base_decoder.patch_size,
                stride=self.base_decoder.patch_size,
                ceil_mode=True,
            ).squeeze(1).to(torch.bool)[:, :sequence_length]

        if implementation == "flash_attention_2":
            cross = None if encoder_attention_mask is None else encoder_attention_mask.to(hidden.device)
            return {
                "full_attention": audio_attention_mask,
                "sliding_attention": audio_attention_mask,
            }, cross

        valid = _padding_mask(audio_attention_mask, sequence_length, sequence_length)
        full = _additive_mask(valid, hidden.dtype)
        sliding = full
        if getattr(self.base_decoder.config, "use_sliding_window", False):
            window = int(self.base_decoder.config.sliding_window)
            positions = torch.arange(sequence_length, device=hidden.device)
            local = (positions[:, None] - positions[None, :]).abs() <= window
            sliding = _additive_mask(valid & local[None, None], hidden.dtype)

        cross = None
        if encoder_length:
            encoder_valid = _padding_mask(
                encoder_attention_mask.to(hidden.device), sequence_length, encoder_length
            )
            cross = _additive_mask(encoder_valid, hidden.dtype)
        return {"full_attention": full, "sliding_attention": sliding}, cross

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        melody_pitch_indices: torch.Tensor,
        control_scale: float = 1.0,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        decoder = self.base_decoder
        temb_t, timestep_proj_t = decoder.time_embed(timestep)
        temb_r, timestep_proj_r = decoder.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_projection = timestep_proj_t + timestep_proj_r

        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_length = hidden_states.shape[1]
        pad_length = (-original_length) % decoder.patch_size
        if pad_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
        hidden_states = decoder.proj_in(hidden_states)
        encoder_hidden_states = decoder.condition_embedder(encoder_hidden_states)

        batch, sequence_length, _ = hidden_states.shape
        position_ids = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        position_embeddings = decoder.rotary_emb(hidden_states, position_ids)
        self_masks, cross_mask = self._attention_masks(
            hidden_states, attention_mask, encoder_attention_mask
        )
        melody = self.melody_encoder(
            melody_pitch_indices.to(hidden_states.device), sequence_length
        ).to(hidden_states.dtype)

        def run(layer: nn.Module, values: torch.Tensor) -> torch.Tensor:
            return self._run_layer(
                layer,
                values,
                position_embeddings,
                timestep_projection,
                self_masks[layer.attention_type],
                position_ids,
                encoder_hidden_states,
                cross_mask,
            )

        hidden_states = run(decoder.layers[0], hidden_states)
        control_hidden: torch.Tensor | None = None
        for frozen_index in range(1, self.copy_blocks + 1):
            control = self.control_blocks[frozen_index - 1]
            if frozen_index == 1:
                control_input = control.prepare_first(hidden_states, melody)
            else:
                if control_hidden is None:
                    raise RuntimeError("Control hidden state was not initialized")
                control_input = control_hidden
            control_hidden = run(control.copied_block, control_input)
            residual = control.after_proj(control_hidden) * control_scale
            hidden_states = run(decoder.layers[frozen_index], hidden_states + residual)

        for frozen_index in range(self.copy_blocks + 1, len(decoder.layers)):
            hidden_states = run(decoder.layers[frozen_index], hidden_states)

        shift, scale = (decoder.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (
            decoder.norm_out(hidden_states) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = decoder.proj_out(hidden_states)[:, :original_length]
        return hidden_states, None


def make_silence_context(
    silence_latent: torch.Tensor,
    *,
    batch_size: int,
    target_length: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ACE-Step's standard text-to-music Source+Mask context."""

    silence = silence_latent
    if silence.ndim == 2:
        silence = silence.unsqueeze(0)
    if silence.ndim != 3 or silence.shape[-1] != 64:
        raise ValueError(f"Expected silence latent [B, T, 64], got {tuple(silence.shape)}")
    if silence.shape[1] < target_length:
        repeats = math.ceil(target_length / silence.shape[1])
        silence = silence.repeat(1, repeats, 1)
    silence = silence[:, :target_length].to(device=device, dtype=dtype)
    if silence.shape[0] == 1 and batch_size > 1:
        silence = silence.expand(batch_size, -1, -1)
    elif silence.shape[0] != batch_size:
        raise ValueError(f"Silence batch {silence.shape[0]} does not match batch {batch_size}")
    return torch.cat([silence, torch.ones_like(silence)], dim=-1)


__all__ = [
    "AceStepDiTControlNet",
    "MIDDLE_C_HZ",
    "MIDI_ZERO_HZ",
    "TOPK_CQT_SCHEMA",
    "TopKCQTMelodyEncoder",
    "extract_repeated_topk_cqt",
    "make_silence_context",
]
