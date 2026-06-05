from __future__ import annotations

from pathlib import Path
import os

import torchaudio as ta
import torch

from playread.cache import LineCache, default_cache_dir, is_line_stale, line_cache_key, load_manifest, save_manifest, update_line_entry
from playread.script import load_script
from playread.synthesis import synthesize_line


class FakeModel:
    sr = 1000
    calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.ones(1, 75)


def write_script(tmp_path: Path, text: str = "Hello.", cfg_weight: float = 0.5) -> Path:
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
    script_path = tmp_path / "script.yaml"
    script_path.write_text(
        f"""
voices:
  A:
    file: voice.wav
    cfg_weight: {cfg_weight}
play:
  - scene_1
scene_1:
  - A: "{text}"
""",
        encoding="utf-8",
    )
    return script_path


def cache_current_line(tmp_path: Path, script_path: Path) -> tuple[LineCache, dict[str, object]]:
    script = load_script(script_path)
    line = script.lines[0]
    cache = LineCache(tmp_path / "out")
    cache.line_path(line).parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(cache.line_path(line)), torch.zeros(1, 100), 1000)
    manifest = load_manifest(cache)
    update_line_entry(line, manifest)
    save_manifest(cache, manifest)
    return cache, manifest


def test_cache_hit_when_file_and_key_are_current(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    line = load_script(script_path).lines[0]

    assert not is_line_stale(line, cache, manifest)


def test_cache_invalidates_text_change(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    changed = load_script(write_script(tmp_path, text="Changed.")).lines[0]

    assert is_line_stale(changed, cache, manifest)


def test_cache_invalidates_voice_config_change(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    changed = load_script(write_script(tmp_path, cfg_weight=0.7)).lines[0]

    assert is_line_stale(changed, cache, manifest)


def test_cache_ignores_prompt_file_mtime_change(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    voice = tmp_path / "voice.wav"
    os.utime(voice, None)
    changed = load_script(script_path).lines[0]

    assert not is_line_stale(changed, cache, manifest)


def test_cache_invalidates_prompt_file_content_change(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"changed voice")
    changed = load_script(script_path).lines[0]

    assert is_line_stale(changed, cache, manifest)


def test_manifest_uses_relative_prompt_paths_and_hashes(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, script_path)
    line = load_script(script_path).lines[0]
    entry = manifest["lines"][line.selector]

    assert cache.cache_dir == tmp_path / "out"
    assert default_cache_dir(script_path) == tmp_path / "script-cache"
    assert entry["voice"]["audio_prompt_path"] == "voice.wav"
    assert entry["prompt_files"][0]["path"] == "voice.wav"
    assert "sha256" in entry["prompt_files"][0]
    assert "mtime_ns" not in entry["prompt_files"][0]
    assert str(tmp_path) not in entry["voice"]["audio_prompt_path"]


def test_force_rerender_updates_manifest_key(tmp_path: Path) -> None:
    old_script_path = write_script(tmp_path)
    cache, manifest = cache_current_line(tmp_path, old_script_path)
    new_line = load_script(write_script(tmp_path, text="Fresh text.")).lines[0]

    old_key = manifest["lines"]["scene_1:1"]["cache_key"]
    synthesize_line(FakeModel(), new_line, cache, manifest)

    assert manifest["lines"]["scene_1:1"]["cache_key"] == line_cache_key(new_line)
    assert manifest["lines"]["scene_1:1"]["cache_key"] != old_key
    assert not is_line_stale(new_line, cache, manifest)


def test_synthesize_composite_line_generates_and_mixes_component_voices(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
    script_path = tmp_path / "script.yaml"
    script_path.write_text(
        """
voices:
  A:
    file: voice.wav
    cfg_weight: 0.4
  B:
    file: voice.wav
    cfg_weight: 0.7
play:
  - scene_1
scene_1:
  - A and B: "Together."
""",
        encoding="utf-8",
    )
    line = load_script(script_path).lines[0]
    model = FakeModel()
    cache = LineCache(tmp_path / "out")
    manifest = load_manifest(cache)

    out_path = synthesize_line(model, line, cache, manifest)
    wav, sr = ta.load(str(out_path))

    assert sr == 1000
    assert wav.shape == (1, 100)
    assert [call["cfg_weight"] for call in model.calls] == [0.4, 0.7]
    assert manifest["lines"]["scene_1:1"]["voices"][0]["cfg_weight"] == 0.4
    assert manifest["lines"]["scene_1:1"]["voices"][1]["cfg_weight"] == 0.7
