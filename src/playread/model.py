from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceConfig:
    character: str
    audio_prompt_path: Path | None
    cache_prompt_path: str | None = None
    cfg_weight: float | None = None
    exaggeration: float | None = None

    def generation_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "cfg_weight": self.cfg_weight if self.cfg_weight is not None else 0.5,
            "exaggeration": self.exaggeration if self.exaggeration is not None else 0.5,
        }
        if self.audio_prompt_path is not None and self.audio_prompt_path.exists():
            kwargs["audio_prompt_path"] = str(self.audio_prompt_path)
        return kwargs

    def key_data(self) -> dict[str, Any]:
        return {
            "audio_prompt_path": self.cache_prompt_path,
            "cfg_weight": self.cfg_weight,
            "exaggeration": self.exaggeration,
        }


@dataclass(frozen=True)
class ScriptLine:
    scene: str
    number: int
    character: str
    raw_text: str
    normalized_text: str
    voice: VoiceConfig
    voices: tuple[VoiceConfig, ...] | None = None

    def __post_init__(self) -> None:
        if self.voices is None:
            object.__setattr__(self, "voices", (self.voice,))
        elif not self.voices:
            raise ValueError("line must have at least one voice")

    @property
    def selector(self) -> str:
        return f"{self.scene}:{self.number}"

    @property
    def cache_filename(self) -> str:
        safe_character = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.character)
        return f"{self.number:03d}_{safe_character}.wav"


@dataclass(frozen=True)
class Script:
    path: Path
    voices: dict[str, VoiceConfig]
    play: list[str]
    scenes: dict[str, list[ScriptLine]]

    @property
    def lines(self) -> list[ScriptLine]:
        return [line for scene_name in self.play for line in self.scenes[scene_name]]


@dataclass(frozen=True)
class LineSelector:
    scene: str
    number: int

    @classmethod
    def parse(cls, value: str) -> "LineSelector":
        scene, sep, number_text = value.rpartition(":")
        if not sep or not scene or not number_text.isdecimal():
            raise ValueError(f"line selector must use SCENE:NUMBER format: {value}")
        number = int(number_text)
        if number < 1:
            raise ValueError(f"line selector number must be 1 or greater: {value}")
        return cls(scene=scene, number=number)
