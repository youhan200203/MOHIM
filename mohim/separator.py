"""Lazy-loading HTDemucs six-source separator."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class StemSeparator:
    def __init__(self, device: str = "cuda", model_name: str = "htdemucs_6s") -> None:
        self.device = device
        self.model_name = model_name
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

    def separate(self, audio_path: str | Path) -> tuple[dict[str, Any], int, Any]:
        """Return stem tensors, sample rate, and the resampled stereo mixture."""

        import torch
        import torchaudio
        from demucs.apply import apply_model

        wav, sample_rate = torchaudio.load(str(audio_path))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        with torch.inference_mode():
            sources = apply_model(self.model, wav.to(self.device)[None], device=self.device)[0]
        stems = {name: sources[index].cpu() for index, name in enumerate(self.model.sources)}
        return stems, self.sample_rate, wav.cpu()


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
