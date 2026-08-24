"""Training utilities for the ACE-Step melody ControlNet experiment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .controlnet import AceStepDiTControlNet, TOPK_CQT_SCHEMA


class ControlNetTensorDataset(Dataset):
    """Join existing target tensors with separate CQT and text-condition caches."""

    def __init__(
        self,
        tensor_dir: str | Path,
        cqt_dir: str | Path,
        condition_dir: str | Path,
        *,
        paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.tensor_dir = Path(tensor_dir)
        self.cqt_dir = Path(cqt_dir)
        self.condition_dir = Path(condition_dir)
        self.paths = [Path(path) for path in paths] if paths is not None else sorted(self.tensor_dir.glob("*.pt"))
        if not self.paths:
            raise ValueError(f"No target tensors found in {self.tensor_dir}")
        missing = [
            path.name
            for path in self.paths
            if not (self.cqt_dir / path.name).is_file()
            or not (self.condition_dir / path.name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing CQT/text caches for {len(missing)} tensors: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        target = torch.load(path, map_location="cpu", weights_only=True)
        cqt = torch.load(self.cqt_dir / path.name, map_location="cpu", weights_only=True)
        condition = torch.load(self.condition_dir / path.name, map_location="cpu", weights_only=True)
        if cqt.get("schema") != TOPK_CQT_SCHEMA:
            raise ValueError(f"Unexpected CQT schema in {self.cqt_dir / path.name}")
        context = target.get("context_latents")
        if context is None:
            raise ValueError(f"Missing ACE-Step source context_latents in {path}")
        if context.ndim != 2 or context.shape[-1] != 128:
            raise ValueError(
                f"Expected context_latents shaped [T, 128] in {path}, got {tuple(context.shape)}"
            )
        if context.shape[0] != target["target_latents"].shape[0]:
            raise ValueError(
                f"Source/target length mismatch in {path}: "
                f"{context.shape[0]} != {target['target_latents'].shape[0]}"
            )
        return {
            "target_latents": target["target_latents"],
            "context_latents": context,
            "attention_mask": target["attention_mask"],
            "encoder_hidden_states": condition["encoder_hidden_states"],
            "encoder_attention_mask": condition["encoder_attention_mask"],
            "melody_pitch_indices": cqt["melody_pitch_indices"],
            "path": str(path),
        }


def _pad_time(tensor: torch.Tensor, length: int, *, value: float = 0.0) -> torch.Tensor:
    if tensor.shape[0] >= length:
        return tensor[:length]
    return F.pad(tensor, (0, 0, 0, length - tensor.shape[0]), value=value)


def collate_controlnet_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    target_length = max(sample["target_latents"].shape[0] for sample in samples)
    encoder_length = max(sample["encoder_hidden_states"].shape[0] for sample in samples)
    melody_length = max(sample["melody_pitch_indices"].shape[0] for sample in samples)
    return {
        "target_latents": torch.stack(
            [_pad_time(sample["target_latents"], target_length) for sample in samples]
        ),
        "context_latents": torch.stack(
            [_pad_time(sample["context_latents"], target_length) for sample in samples]
        ),
        "attention_mask": torch.stack(
            [F.pad(sample["attention_mask"], (0, target_length - sample["attention_mask"].shape[0])) for sample in samples]
        ),
        "encoder_hidden_states": torch.stack(
            [_pad_time(sample["encoder_hidden_states"], encoder_length) for sample in samples]
        ),
        "encoder_attention_mask": torch.stack(
            [F.pad(sample["encoder_attention_mask"], (0, encoder_length - sample["encoder_attention_mask"].shape[0])) for sample in samples]
        ),
        "melody_pitch_indices": torch.stack(
            [_pad_time(sample["melody_pitch_indices"], melody_length) for sample in samples]
        ),
        "paths": [sample["path"] for sample in samples],
    }


def split_paths(
    paths: Sequence[str | Path], *, validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    values = [Path(path) for path in paths]
    random.Random(seed).shuffle(values)
    validation_size = max(1, round(len(values) * validation_fraction))
    return values[validation_size:], values[:validation_size]


def load_silence_latent(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        candidates = [tensor for tensor in value.values() if torch.is_tensor(tensor)]
        if len(candidates) != 1:
            raise ValueError(f"Could not identify silence tensor in {path}")
        value = candidates[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Silence latent is not a tensor: {path}")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 3 and value.shape[-1] == 64:
        return value.contiguous()
    if value.ndim == 3 and value.shape[1] == 64:
        return value.transpose(1, 2).contiguous()
    if value.ndim != 3:
        raise ValueError(f"Unexpected silence latent shape: {tuple(value.shape)}")
    raise ValueError(f"Expected one 64-channel axis in silence latent: {tuple(value.shape)}")


def move_batch(
    batch: dict[str, Any], device: torch.device | str, dtype: torch.dtype
) -> dict[str, Any]:
    return {
        **batch,
        "target_latents": batch["target_latents"].to(device=device, dtype=dtype),
        "context_latents": batch["context_latents"].to(device=device, dtype=dtype),
        "attention_mask": batch["attention_mask"].to(device=device),
        "encoder_hidden_states": batch["encoder_hidden_states"].to(device=device, dtype=dtype),
        "encoder_attention_mask": batch["encoder_attention_mask"].to(device=device),
        "melody_pitch_indices": batch["melody_pitch_indices"].to(device=device),
    }


def flow_matching_step(
    model: AceStepDiTControlNet,
    batch: dict[str, Any],
    *,
    timestep_mu: float,
    timestep_sigma: float,
    control_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
) -> torch.Tensor:
    target = batch["target_latents"]
    batch_size, target_length, channels = target.shape
    if channels != 64:
        raise ValueError(f"Expected 64-channel target latents, got {channels}")
    if noise is None:
        noise = torch.randn_like(target)
    elif noise.shape != target.shape:
        raise ValueError(
            f"Fixed noise shape {tuple(noise.shape)} does not match target {tuple(target.shape)}"
        )
    else:
        noise = noise.to(device=target.device, dtype=target.dtype)
    if timestep is None:
        timestep = torch.sigmoid(
            torch.randn(batch_size, device=target.device, dtype=target.dtype) * timestep_sigma
            + timestep_mu
        )
    elif timestep.shape != (batch_size,):
        raise ValueError(
            f"Fixed timestep must have shape {(batch_size,)}, got {tuple(timestep.shape)}"
        )
    else:
        timestep = timestep.to(device=target.device, dtype=target.dtype)
    if torch.any(timestep < 0.0) or torch.any(timestep > 1.0):
        raise ValueError("Fixed timestep values must be in [0, 1]")
    amount = timestep[:, None, None]
    noised = amount * noise + (1.0 - amount) * target
    context = batch["context_latents"]
    if context.shape != (batch_size, target_length, 128):
        raise ValueError(
            "Expected context_latents shaped "
            f"{(batch_size, target_length, 128)}, got {tuple(context.shape)}"
        )
    prediction = model(
        hidden_states=noised,
        timestep=timestep,
        timestep_r=timestep,
        attention_mask=batch["attention_mask"],
        encoder_hidden_states=batch["encoder_hidden_states"],
        encoder_attention_mask=batch["encoder_attention_mask"],
        context_latents=context,
        melody_pitch_indices=batch["melody_pitch_indices"],
        control_scale=control_scale,
    )[0]
    flow = noise - target
    weights = batch["attention_mask"].to(prediction.dtype).unsqueeze(-1)
    return ((prediction - flow).square() * weights).sum() / (weights.sum() * channels)


def create_adamw8bit(
    model: AceStepDiTControlNet,
    *,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required. In Colab run `%pip install -q bitsandbytes`."
        ) from exc
    return bnb.optim.AdamW8bit(
        model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def create_inverse_lr(
    optimizer: torch.optim.Optimizer, *, inverse_gamma: int = 10_000, power: float = 0.5
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1.0 + step / inverse_gamma) ** (-power)
    )


def cuda_memory_summary() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }


def memory_smoke_test(
    model: AceStepDiTControlNet,
    batch: dict[str, Any],
    *,
    timestep_mu: float,
    timestep_sigma: float,
    learning_rate: float = 5e-5,
) -> dict[str, float]:
    """Run a real AdamW8bit update so optimizer-state memory is included."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    optimizer = create_adamw8bit(model, learning_rate=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_step(
        model,
        batch,
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    result = {"loss": float(loss.detach())}
    result.update(cuda_memory_summary())
    del optimizer, loss
    return result


def save_training_checkpoint(
    path: str | Path,
    *,
    model: AceStepDiTControlNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    history: list[dict[str, float]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "controlnet": model.control_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        temporary,
    )
    temporary.replace(destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: AceStepDiTControlNet,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_control_state_dict(state["controlnet"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def write_history(path: str | Path, history: list[dict[str, float]]) -> None:
    Path(path).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ControlNetTensorDataset",
    "collate_controlnet_batch",
    "create_adamw8bit",
    "create_inverse_lr",
    "cuda_memory_summary",
    "flow_matching_step",
    "load_silence_latent",
    "load_training_checkpoint",
    "memory_smoke_test",
    "move_batch",
    "save_training_checkpoint",
    "split_paths",
    "write_history",
]
