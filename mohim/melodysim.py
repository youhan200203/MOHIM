"""MelodySim inference for one selected motif against aligned four-bar windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


MELODYSIM_MODEL_ID = "m-a-p/MERT-v1-95M"
MELODYSIM_CHECKPOINT_REPO = "amaai-lab/MelodySim"
MELODYSIM_CHECKPOINT_FILE = "siamese_net_20250328.ckpt"


def phase_aligned_four_bar_segments(
    downbeats: Iterable[float],
    candidate_start_sec: float,
    *,
    bars: int = 4,
    audio_duration_sec: float | None = None,
) -> list[dict[str, float | int | bool]]:
    """Return complete bar windows sharing the selected candidate's phase."""
    values = np.asarray(list(downbeats), dtype=np.float64).reshape(-1)
    if bars <= 0:
        raise ValueError("bars must be positive.")
    if len(values) <= bars:
        return []
    candidate_index = int(np.argmin(np.abs(values - candidate_start_sec)))
    if not np.isclose(values[candidate_index], candidate_start_sec, atol=0.05):
        raise ValueError("Selected motif start does not match a detected downbeat.")

    rows: list[dict[str, float | int | bool]] = []
    first_index = candidate_index % bars
    for start_index in range(first_index, len(values) - bars, bars):
        end_index = start_index + bars
        start_sec = float(values[start_index])
        end_sec = float(values[end_index])
        if end_sec <= start_sec:
            continue
        if audio_duration_sec is not None and end_sec > audio_duration_sec:
            continue
        rows.append(
            {
                "segment_index": len(rows),
                "start_downbeat_index": start_index,
                "end_downbeat_index": end_index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "is_source": start_index == candidate_index,
            }
        )
    return rows


class _ResidualBlock:
    """Factory namespace that keeps torch optional until MelodySim is used."""

    @staticmethod
    def create(in_channels: int, out_channels: int) -> Any:
        import torch.nn as nn

        class ResidualBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.shortcut = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1, 1),
                        nn.BatchNorm1d(out_channels),
                    )
                    if in_channels != out_channels
                    else nn.Sequential()
                )

            def forward(self, inputs: Any) -> Any:
                output = self.relu(self.bn1(self.conv1(inputs)))
                output = self.bn2(self.conv2(output)) + self.shortcut(inputs)
                return self.relu(output)

        return ResidualBlock()


def _create_similarity_head() -> Any:
    import torch
    import torch.nn as nn

    class SiameseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = _ResidualBlock.create(3072, 512)
            self.layer2 = _ResidualBlock.create(512, 256)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(256, 128)

        def forward(self, inputs: Any) -> Any:
            output = self.layer2(self.layer1(inputs))
            return self.fc(self.global_pool(output).flatten(1))

    class MelodySimHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.siamese_net = SiameseNet()
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def encode(self, features: Any) -> Any:
            return self.siamese_net(features)

        def compare(self, query: Any, references: Any) -> Any:
            query_batch = query.expand(references.shape[0], -1)
            different_probability = torch.sigmoid(
                self.classifier(torch.abs(query_batch - references)).squeeze(-1)
            )
            return 1.0 - different_probability

    return MelodySimHead()


class MelodySimEncoder:
    """Official MelodySim MERT-95M backbone and released similarity head."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.torch = torch
        self.device = torch.device(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MELODYSIM_MODEL_ID)
        self.sample_rate = int(self.processor.sampling_rate)
        self.backbone = AutoModel.from_pretrained(
            MELODYSIM_MODEL_ID,
            trust_remote_code=True,
        ).to(self.device).eval()
        resolved_checkpoint = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else Path(
                hf_hub_download(
                    repo_id=MELODYSIM_CHECKPOINT_REPO,
                    filename=MELODYSIM_CHECKPOINT_FILE,
                )
            )
        )
        checkpoint = torch.load(
            resolved_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        head_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("siamese_net.") or key.startswith("classifier.")
        }
        if not head_state:
            raise ValueError("MelodySim checkpoint does not contain the similarity head.")
        self.head = _create_similarity_head().to(self.device)
        self.head.load_state_dict(head_state, strict=True)
        self.head.eval()
        self.time_reduce = torch.nn.AvgPool1d(
            kernel_size=10,
            stride=10,
            count_include_pad=False,
        ).to(self.device)

    def _prepare_audio(self, audio: Any, sample_rate: int, target_samples: int) -> Any:
        import torchaudio.functional as audio_functional

        values = audio.detach().cpu() if hasattr(audio, "detach") else self.torch.as_tensor(audio)
        values = values.float()
        if values.ndim == 2:
            values = values.mean(dim=0)
        if values.ndim != 1:
            raise ValueError("MelodySim audio must be mono or channel-first stereo.")
        if sample_rate != self.sample_rate:
            values = audio_functional.resample(values, sample_rate, self.sample_rate)
        if len(values) < target_samples:
            values = self.torch.nn.functional.pad(values, (0, target_samples - len(values)))
        else:
            values = values[:target_samples]
        processed = self.processor(
            values.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )["input_values"]
        return processed.squeeze(0)

    @property
    def identity(self) -> dict[str, str]:
        return {
            "model_id": MELODYSIM_MODEL_ID,
            "checkpoint_repo": MELODYSIM_CHECKPOINT_REPO,
            "checkpoint_file": MELODYSIM_CHECKPOINT_FILE,
        }

    def _encode_batch(self, waveforms: Any) -> Any:
        with self.torch.inference_mode():
            hidden_states = self.backbone(
                waveforms.to(self.device),
                output_hidden_states=True,
            ).hidden_states
            selected = [
                self.time_reduce(hidden.transpose(1, 2)).transpose(1, 2)
                for hidden in hidden_states[2::3]
            ]
            if len(selected) != 4:
                raise ValueError(f"Expected four MERT layers, received {len(selected)}.")
            features = self.torch.stack(selected, dim=1).permute(0, 1, 3, 2)
            features = features.flatten(start_dim=1, end_dim=2)
            return self.head.encode(features)

    def compare_candidate_to_segments(
        self,
        candidate_audio: Any,
        segment_audios: Iterable[Any],
        *,
        sample_rate: int,
        batch_size: int = 8,
    ) -> list[float]:
        """Encode the candidate once and compare it with every aligned segment."""
        segments = list(segment_audios)
        if not segments:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        candidate_values = (
            candidate_audio.detach().cpu()
            if hasattr(candidate_audio, "detach")
            else self.torch.as_tensor(candidate_audio)
        )
        target_samples = round(candidate_values.shape[-1] * self.sample_rate / sample_rate)
        query_waveform = self._prepare_audio(candidate_audio, sample_rate, target_samples)
        query_embedding = self._encode_batch(query_waveform.unsqueeze(0))

        scores: list[float] = []
        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start : batch_start + batch_size]
            waveforms = self.torch.stack(
                [self._prepare_audio(audio, sample_rate, target_samples) for audio in batch]
            )
            reference_embeddings = self._encode_batch(waveforms)
            with self.torch.inference_mode():
                similarities = self.head.compare(query_embedding, reference_embeddings)
            scores.extend(float(value) for value in similarities.detach().cpu())
        return scores
