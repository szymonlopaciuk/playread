from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import yaml

from .model import LineSelector, Script, ScriptLine, VoiceConfig


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resolve_path(path_value: object, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def normalize_for_tts(character: str, text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)

    character_key = character.upper()
    if character_key == "NARRATOR":
        t = t.replace(". ", "... ")
    elif character_key == "ALEX":
        t = t.replace("Well, ", "Well... ")
        t = t.replace("Sorry, what?", "Sorry... what?")
        t = t.replace("No, not really...", "No... not really...")
    elif character_key == "PETER":
        t = t.replace("Indeed! So,", "Indeed. So,")
    elif character_key == "MELINDA":
        pass

    return (
        t.replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "...")
        .replace("\u2014", "...")
    )


def composite_characters(character: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in character.split(" and ") if part.strip())
    return parts if len(parts) > 1 else ()


def load_script(script_path: Path) -> Script:
    script_path = script_path.expanduser().resolve()
    with script_path.open("r", encoding="utf-8") as script_file:
        script_data = _require_mapping(yaml.safe_load(script_file), str(script_path))

    script_dir = script_path.parent
    voices_data = _require_mapping(script_data.get("voices"), "voices")
    play_data = script_data.get("play")
    if not isinstance(play_data, list) or not all(isinstance(name, str) for name in play_data):
        raise ValueError("play must be a list of scene names")

    voices: dict[str, VoiceConfig] = {}
    for character, raw_cfg in voices_data.items():
        cfg = _require_mapping(raw_cfg, f"voices.{character}")
        prompt_path = _resolve_path(cfg["file"], script_dir) if cfg.get("file") else None
        character_name = str(character)
        voices[character_name] = VoiceConfig(
            character=character_name,
            audio_prompt_path=prompt_path,
            cfg_weight=cfg.get("cfg_weight"),
            exaggeration=cfg.get("exaggeration"),
        )

    scenes: dict[str, list[ScriptLine]] = {}
    for scene_name in play_data:
        raw_lines = script_data.get(scene_name)
        if not isinstance(raw_lines, list):
            raise ValueError(f"{scene_name} must be a list of lines")

        lines: list[ScriptLine] = []
        for line_index, raw_line in enumerate(raw_lines, start=1):
            if not isinstance(raw_line, dict) or len(raw_line) != 1:
                raise ValueError(f"{scene_name}[{line_index}] must contain one character-to-line mapping")
            character, text = next(iter(raw_line.items()))
            character_name = str(character)
            if not isinstance(text, str):
                raise ValueError(f"{scene_name}[{line_index}] text must be a string")
            if character_name in voices:
                line_voices = (voices[character_name],)
            else:
                component_names = composite_characters(character_name)
                if not component_names:
                    raise ValueError(f"{scene_name}[{line_index}] has no voice configured for {character_name}")
                missing_components = [name for name in component_names if name not in voices]
                if missing_components:
                    missing_text = ", ".join(missing_components)
                    raise ValueError(
                        f"{scene_name}[{line_index}] has no voice configured for composite speaker component(s): "
                        f"{missing_text}"
                    )
                line_voices = tuple(voices[name] for name in component_names)
            lines.append(
                ScriptLine(
                    scene=scene_name,
                    number=line_index,
                    character=character_name,
                    raw_text=text,
                    normalized_text=normalize_for_tts(character_name, text),
                    voice=line_voices[0],
                    voices=line_voices,
                )
            )

        scenes[scene_name] = lines

    return Script(path=script_path, voices=voices, play=list(play_data), scenes=scenes)


def parse_selectors(values: tuple[str, ...] | list[str]) -> list[LineSelector]:
    return [LineSelector.parse(value) for value in values]


def find_line(script: Script, selector: LineSelector) -> ScriptLine:
    lines = script.scenes.get(selector.scene)
    if lines is None:
        raise ValueError(f"unknown scene in selector: {selector.scene}")
    if selector.number > len(lines):
        raise ValueError(f"line selector out of range: {selector.scene}:{selector.number}")
    return lines[selector.number - 1]


def render_numbered_markdown(script: Script) -> str:
    parts = [f"# {script.path.name}", ""]
    for scene_name in script.play:
        parts.extend([f"## {scene_name}", ""])
        for line in script.scenes[scene_name]:
            parts.append(f"- `{line.selector}` **{line.character}**: {line.raw_text}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
