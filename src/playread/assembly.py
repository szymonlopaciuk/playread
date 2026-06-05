from __future__ import annotations

from pathlib import Path

import torch
import torchaudio as ta

from .cache import LineCache
from .model import Script, ScriptLine

INITIAL_PAUSE_MS = 300
NARRATOR_PAUSE_MS = 450
LINE_PAUSE_MS = 280
ESTIMATED_CHARS_PER_SECOND = 14.0


def silence(sr: int, ms: int, channels: int = 1) -> torch.Tensor:
    samples = int(sr * (ms / 1000.0))
    return torch.zeros(channels, samples)


def line_pause_ms(character: str) -> int:
    return NARRATOR_PAUSE_MS if character.upper() == "NARRATOR" else LINE_PAUSE_MS


def estimate_duration_samples(text: str, sr: int) -> int:
    seconds = max(0.35, len(text.strip()) / ESTIMATED_CHARS_PER_SECOND)
    return int(sr * seconds)


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = ta.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr


def _match_channels(wav: torch.Tensor, channels: int) -> torch.Tensor:
    if wav.shape[0] == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if wav.shape[0] == 1:
        return wav.repeat(channels, 1)
    return wav[:channels, :]


def _route_for_focus(
    wav: torch.Tensor, line: ScriptLine, focus_character: str, focus_channel: str
) -> torch.Tensor:
    mono = _match_channels(wav, 1)
    routed = torch.zeros(2, mono.shape[1])
    focus_index = 0 if focus_channel == "left" else 1
    other_index = 1 - focus_index
    routed[focus_index if line.character == focus_character else other_index, :] = mono[
        0
    ]
    return routed


def _silenced_line(
    line: ScriptLine, cache: LineCache, sr: int, multiplier: float, channels: int
) -> torch.Tensor:
    line_path = cache.line_path(line)
    if line_path.exists():
        wav, cached_sr = read_wav(line_path)
        samples = wav.shape[1]
        if cached_sr != sr:
            samples = int(samples * sr / cached_sr)
    else:
        samples = estimate_duration_samples(line.raw_text, sr)
    return torch.zeros(channels, max(1, int(samples * multiplier)))


def _line_audio(
    line: ScriptLine,
    cache: LineCache,
    sr: int,
    silence_characters: set[str],
    silence_multiplier: float,
    focus_character: str | None,
    focus_channel: str,
) -> torch.Tensor:
    channels = 2 if focus_character else 1
    if line.character in silence_characters:
        return _silenced_line(line, cache, sr, silence_multiplier, channels)

    wav, wav_sr = read_wav(cache.line_path(line))
    if wav_sr != sr:
        raise ValueError(f"{line.selector} has sample rate {wav_sr}, expected {sr}")
    if focus_character:
        return _route_for_focus(wav, line, focus_character, focus_channel)
    return _match_channels(wav, 1)


def assemble_scene(
    script: Script,
    scene_name: str,
    cache: LineCache,
    out_dir: Path,
    sr: int,
    silence_characters: set[str] | None = None,
    silence_multiplier: float = 1.2,
    skip_characters: set[str] | None = None,
    focus_character: str | None = None,
    focus_channel: str = "left",
) -> Path:
    silence_characters = silence_characters or set()
    skip_characters = skip_characters or set()
    channels = 2 if focus_character else 1
    segments: list[torch.Tensor] = [silence(sr, INITIAL_PAUSE_MS, channels)]

    for line in script.scenes[scene_name]:
        if line.character in skip_characters:
            continue
        segments.append(
            _line_audio(
                line,
                cache,
                sr,
                silence_characters,
                silence_multiplier,
                focus_character,
                focus_channel,
            )
        )
        segments.append(silence(sr, line_pause_ms(line.character), channels))

    final_wav = torch.cat(segments, dim=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenes" / f"{scene_name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out_path), final_wav, sr)
    return out_path


def assemble_play(scene_paths: list[Path], out_dir: Path, sr: int) -> Path:
    segments: list[torch.Tensor] = []
    channels = 1
    for scene_path in scene_paths:
        wav, wav_sr = read_wav(scene_path)
        if wav_sr != sr:
            raise ValueError(f"{scene_path} has sample rate {wav_sr}, expected {sr}")
        channels = max(channels, wav.shape[0])
        segments.append(wav)
    matched = [_match_channels(wav, channels) for wav in segments]
    final_wav = torch.cat(matched, dim=1) if matched else silence(sr, 0, channels)
    out_path = out_dir / "play.wav"
    ta.save(str(out_path), final_wav, sr)
    return out_path
