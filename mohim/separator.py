"""Lazy-loading HTDemucs six-source separator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class StemSeparator:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "htdemucs_6s",
        shifts: int = 0,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.shifts = shifts
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from demucs.pretrained import get_model

            model = get_model(self.model_name)
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    @property
    def sample_rate(self) -> int:
        return int(self.model.samplerate)

    def _load_audio(self, audio_path: str | Path) -> Any:
        import torchaudio

        wav, sample_rate = torchaudio.load(str(audio_path))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        return wav.cpu()

    def separate(self, audio_path: str | Path) -> tuple[dict[str, Any], int, Any]:
        """Return stem tensors, sample rate, and the resampled stereo mixture."""

        return self.separate_many([audio_path])[0]

    def separate_many(
        self,
        audio_paths: Iterable[str | Path],
    ) -> list[tuple[dict[str, Any], int, Any]]:
        """Separate multiple tracks in one padded GPU batch and preserve input order."""

        import torch
        from demucs.apply import apply_model

        paths = list(audio_paths)
        if not paths:
            return []
        mixtures = [self._load_audio(path) for path in paths]
        lengths = [wav.shape[-1] for wav in mixtures]
        max_length = max(lengths)
        batch = torch.stack(
            [
                torch.nn.functional.pad(wav, (0, max_length - length))
                for wav, length in zip(mixtures, lengths)
            ]
        )

        with torch.inference_mode():
            batch_sources = apply_model(
                self.model,
                batch.to(self.device),
                device=self.device,
                shifts=self.shifts,
            )

        results = []
        for song_index, (mixture, length) in enumerate(zip(mixtures, lengths)):
            sources = batch_sources[song_index, :, :, :length].cpu()
            stems = {name: sources[index] for index, name in enumerate(self.model.sources)}
            results.append((stems, self.sample_rate, mixture))
        return results


def save_audio(path: str | Path, audio: Any, sample_rate: int, *, audio_format: str = "flac") -> Path:
    """Save a tensor as PCM-16 WAV or FLAC to keep prepared datasets compact."""

    import numpy as np
    import soundfile as sf

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    if array.ndim == 2:
        array = array.T
    subtype = "PCM_16"
    sf.write(str(output), array, sample_rate, subtype=subtype, format=audio_format.upper())
    return output
