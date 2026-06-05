from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import torch
import torchaudio as ta

from .cache import LineCache, update_line_entry
from .model import ScriptLine, VoiceConfig

COMPOSITE_VOICE_OFFSET_MS = 25


class TTSModel(Protocol):
    sr: int

    def generate(self, **kwargs: object) -> torch.Tensor: ...


def load_tts_model(device: str) -> TTSModel:
    from chatterbox.tts import ChatterboxTTS

    return cast("TTSModel", ChatterboxTTS.from_pretrained(device=device))


def _as_cpu_channels_first(wav: torch.Tensor) -> torch.Tensor:
    if wav.device.type != "cpu":
        wav = wav.detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError("generated audio must be a 1D or 2D tensor")
    return wav


def _silence(sr: int, ms: int) -> torch.Tensor:
    samples = int(sr * (ms / 1000.0))
    return torch.zeros(1, samples)


def _delay_waveform(wav: torch.Tensor, sr: int, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    return torch.cat([_silence(sr, ms), wav], dim=1)


def _mix_waveforms(wavs: list[torch.Tensor]) -> torch.Tensor:
    if not wavs:
        raise ValueError("cannot mix an empty waveform list")

    channels = max(wav.shape[0] for wav in wavs)
    max_len = max(wav.shape[1] for wav in wavs)
    mixed = torch.zeros(channels, max_len)

    for wav in wavs:
        if wav.shape[0] == 1 and channels > 1:
            wav = wav.expand(channels, -1)
        elif wav.shape[0] != channels:
            raise ValueError(
                "generated composite voices must have compatible channel counts"
            )
        if wav.shape[1] < max_len:
            wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
        mixed = mixed + wav

    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def _generate_voice(model: TTSModel, text: str, voice: VoiceConfig) -> torch.Tensor:
    kwargs = {"text": text, **voice.generation_kwargs()}
    return _as_cpu_channels_first(model.generate(**kwargs))


def synthesize_line(
    model: TTSModel, line: ScriptLine, cache: LineCache, manifest: dict[str, object]
) -> Path:
    cache.ensure_dirs()
    out_path = cache.line_path(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    line_voices = line.voices or (line.voice,)
    if len(line_voices) == 1:
        wav = _generate_voice(model, line.normalized_text, line_voices[0])
    else:
        wavs = [
            _delay_waveform(
                _generate_voice(model, line.normalized_text, voice),
                model.sr,
                index * COMPOSITE_VOICE_OFFSET_MS,
            )
            for index, voice in enumerate(line_voices)
        ]
        wav = _mix_waveforms(wavs)

    ta.save(str(out_path), wav, model.sr)
    update_line_entry(line, manifest)
    return out_path
