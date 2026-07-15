"""Per-step joint flow matching for dual-stream motif/vocal training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict

import torch

from acestep.training_v2.dual_stream import MOTIF_ROLE, VOCAL_ROLE
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps


def run_dual_stream_training_step(module: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute weighted flow loss for mutually conditioned motif and vocal streams."""
    if module.device_type in ("cuda", "xpu", "mps") and module.dtype != torch.float32:
        autocast_ctx = torch.autocast(device_type=module.device_type, dtype=module.dtype)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        nb = module.transfer_non_blocking
        motif_target = batch["motif_target_latents"].to(module.device, module.dtype, non_blocking=nb)
        motif_mask = batch["motif_target_attention_mask"].to(module.device, module.dtype, non_blocking=nb)
        vocal_target = batch["vocal_target_latents"].to(module.device, module.dtype, non_blocking=nb)
        vocal_mask = batch["vocal_target_attention_mask"].to(module.device, module.dtype, non_blocking=nb)
        motif_seed = batch["motif_seed_latents"].to(module.device, module.dtype, non_blocking=nb)
        motif_seed_mask = batch["motif_seed_attention_mask"].to(module.device, module.dtype, non_blocking=nb)
        base_condition = batch["encoder_hidden_states"].to(module.device, module.dtype, non_blocking=nb)
        base_condition_mask = batch["encoder_attention_mask"].to(module.device, module.dtype, non_blocking=nb)

        t, _ = sample_timesteps(
            batch_size=motif_target.shape[0],
            device=module.device,
            dtype=module.dtype,
            data_proportion=module._data_proportion,
            timestep_mu=module._timestep_mu,
            timestep_sigma=module._timestep_sigma,
            use_meanflow=False,
        )
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        motif_noise, vocal_noise = torch.randn_like(motif_target), torch.randn_like(vocal_target)
        motif_xt = t_ * motif_noise + (1.0 - t_) * motif_target
        vocal_xt = t_ * vocal_noise + (1.0 - t_) * vocal_target
        if module.force_input_grads_for_checkpointing:
            motif_xt, vocal_xt = motif_xt.requires_grad_(True), vocal_xt.requires_grad_(True)

        conditioner = _get_conditioner(module)
        motif_condition, motif_condition_mask = conditioner.build_condition(
            base_condition, base_condition_mask, motif_seed, motif_seed_mask,
            vocal_xt, vocal_mask, MOTIF_ROLE,
        )
        vocal_condition, vocal_condition_mask = conditioner.build_condition(
            base_condition, base_condition_mask, motif_seed, motif_seed_mask,
            motif_xt, motif_mask, VOCAL_ROLE,
        )
        if module._null_cond_emb is not None and module._cfg_ratio > 0.0:
            motif_condition = apply_cfg_dropout(motif_condition, module._null_cond_emb, module._cfg_ratio)
            vocal_condition = apply_cfg_dropout(vocal_condition, module._null_cond_emb, module._cfg_ratio)

        motif_pred = _decode(module, motif_xt, motif_mask, motif_condition, motif_condition_mask, t)
        vocal_pred = _decode(module, vocal_xt, vocal_mask, vocal_condition, vocal_condition_mask, t)
        loss = (
            module.training_config.motif_loss_weight * _masked_flow_loss(motif_pred, motif_noise - motif_target, motif_mask)
            + module.training_config.vocal_loss_weight * _masked_flow_loss(vocal_pred, vocal_noise - vocal_target, vocal_mask)
        )

    loss = loss.float()
    module.training_losses.append(loss.item())
    return loss


def _decode(module: Any, latents: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor, condition_mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Run one stream through the shared DiT with text2music-style zero context."""
    context = torch.zeros(
        latents.shape[0], latents.shape[1], latents.shape[2] * 2,
        device=module.device, dtype=module.dtype,
    )
    return module.model.decoder(
        hidden_states=latents, timestep=t, timestep_r=t, attention_mask=mask,
        encoder_hidden_states=condition, encoder_attention_mask=condition_mask,
        context_latents=context,
    )[0]


def _masked_flow_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return MSE over valid audio frames, excluding collate padding."""
    weights = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    return ((prediction - target).square() * weights).sum() / (
        weights.sum() * prediction.shape[-1]
    ).clamp_min(1.0)


def _get_conditioner(module: Any) -> Any:
    """Find the conditioner through optional Fabric wrappers."""
    decoder = module.model.decoder
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    conditioner = getattr(decoder, "dual_stream_conditioner", None)
    if conditioner is None:
        raise RuntimeError("Dual-stream conditioner is not attached to the decoder.")
    return conditioner
