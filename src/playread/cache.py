from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ScriptLine

SYNTHESIS_SETTINGS = {
    "engine": "chatterbox-tts",
    "normalizer": "playread.script.normalize_for_tts",
}


@dataclass(frozen=True)
class LineCache:
    cache_dir: Path

    @property
    def lines_dir(self) -> Path:
        return self.cache_dir / "lines"

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def line_path(self, line: ScriptLine) -> Path:
        return self.lines_dir / line.scene / line.cache_filename

    def ensure_dirs(self) -> None:
        self.lines_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)


def default_output_dir(input_path: Path) -> Path:
    return input_path.expanduser().resolve().parent / f"{input_path.stem}-out"


def default_cache_dir(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    return input_path.parent / f"{input_path.stem}-cache"


def load_manifest(cache: LineCache) -> dict[str, Any]:
    if not cache.manifest_path.exists():
        return {"lines": {}}
    with cache.manifest_path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, dict) or not isinstance(data.get("lines"), dict):
        return {"lines": {}}
    return data


def save_manifest(cache: LineCache, manifest: dict[str, Any]) -> None:
    cache.ensure_dirs()
    with cache.manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_metadata(path: Path | None, cache_path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    metadata_path = cache_path if cache_path is not None else path.name
    if not path.exists():
        return {"path": metadata_path, "exists": False}
    stat = path.stat()
    return {
        "path": metadata_path,
        "exists": True,
        "size": stat.st_size,
        "sha256": file_sha256(path),
    }


def cache_key_data(line: ScriptLine) -> dict[str, Any]:
    voices = line.voices or (line.voice,)
    return {
        "scene": line.scene,
        "line_number": line.number,
        "character": line.character,
        "raw_text": line.raw_text,
        "normalized_text": line.normalized_text,
        "voice": line.voice.key_data(),
        "voices": [voice.key_data() for voice in voices],
        "prompt_files": [
            prompt_metadata(voice.audio_prompt_path, voice.cache_prompt_path)
            for voice in voices
        ],
        "synthesis_settings": SYNTHESIS_SETTINGS,
    }


def line_cache_key(line: ScriptLine) -> str:
    encoded = json.dumps(
        cache_key_data(line), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_entry(line: ScriptLine, cache_key: str) -> dict[str, Any]:
    voices = line.voices or (line.voice,)
    return {
        "scene": line.scene,
        "line_number": line.number,
        "character": line.character,
        "raw_text": line.raw_text,
        "normalized_text": line.normalized_text,
        "voice": line.voice.key_data(),
        "voices": [voice.key_data() for voice in voices],
        "prompt_files": [
            prompt_metadata(voice.audio_prompt_path, voice.cache_prompt_path)
            for voice in voices
        ],
        "synthesis_settings": SYNTHESIS_SETTINGS,
        "cache_key": cache_key,
    }


def is_line_stale(line: ScriptLine, cache: LineCache, manifest: dict[str, Any]) -> bool:
    entry = manifest.get("lines", {}).get(line.selector)
    if not isinstance(entry, dict):
        return True
    if entry.get("cache_key") != line_cache_key(line):
        return True
    return not cache.line_path(line).exists()


def update_line_entry(line: ScriptLine, manifest: dict[str, Any]) -> None:
    manifest.setdefault("lines", {})[line.selector] = manifest_entry(
        line, line_cache_key(line)
    )
