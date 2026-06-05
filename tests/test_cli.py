from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchaudio as ta
from click.testing import CliRunner

from playread.cache import LineCache, default_cache_dir, load_manifest, save_manifest, update_line_entry
from playread.cli import main
from playread.script import load_script


class FakeModel:
    sr = 1000

    def generate(self, **kwargs: object) -> torch.Tensor:
        return torch.ones(1, 80)


def write_script(tmp_path: Path) -> Path:
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
    script_path = tmp_path / "script.yaml"
    script_path.write_text(
        """
voices:
  A:
    file: voice.wav
  B:
    file: voice.wav
play:
  - scene_1
scene_1:
  - A: "one"
  - B: "two"
  - A: "three"
""",
        encoding="utf-8",
    )
    return script_path


def seed_cache(script_path: Path, cache_dir: Path) -> None:
    script = load_script(script_path)
    cache = LineCache(cache_dir)
    manifest = load_manifest(cache)
    for line in script.lines:
        path = cache.line_path(line)
        path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(path), torch.ones(1, 50), 1000)
        update_line_entry(line, manifest)
    save_manifest(cache, manifest)


def test_cli_script_writes_markdown(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(main, ["script", str(script_path), "--output-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "`scene_1:3` **A**" in (out_dir / "script.md").read_text(encoding="utf-8")


def test_cli_rerender_lines_forces_selected_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = write_script(tmp_path)
    cache_dir = tmp_path / "custom-cache"
    monkeypatch.setattr("playread.cli.load_tts_model", lambda device: FakeModel())

    result = CliRunner().invoke(
        main,
        [
            "rerender-lines",
            str(script_path),
            "scene_1:1",
            "scene_1:3",
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (cache_dir / "lines" / "scene_1" / "001_A.wav").exists()
    assert (cache_dir / "lines" / "scene_1" / "003_A.wav").exists()
    assert not (tmp_path / "out" / "scenes").exists()


def test_cli_render_assembles_from_default_shared_cache(tmp_path: Path) -> None:
    script_path = write_script(tmp_path)
    cache_dir = default_cache_dir(script_path)
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    seed_cache(script_path, cache_dir)

    result_a = CliRunner().invoke(main, ["render", str(script_path), "--output-dir", str(out_a), "--device", "cpu"])
    result_b = CliRunner().invoke(main, ["render", str(script_path), "--output-dir", str(out_b), "--device", "cpu"])

    assert result_a.exit_code == 0, result_a.output
    assert result_b.exit_code == 0, result_b.output
    assert "Line cache is current." in result_a.output
    assert "Line cache is current." in result_b.output
    assert (cache_dir / "lines" / "scene_1" / "001_A.wav").exists()
    assert not (out_a / "cache").exists()
    assert not (out_b / "cache").exists()
    assert (out_a / "scenes" / "scene_1.wav").exists()
    assert (out_b / "play.wav").exists()
    assert (out_b / "script.md").exists()


def test_cli_render_uses_custom_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = write_script(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "custom-cache"
    monkeypatch.setattr("playread.cli.load_tts_model", lambda device: FakeModel())

    result = CliRunner().invoke(
        main,
        ["render", str(script_path), "--output-dir", str(out_dir), "--cache-dir", str(cache_dir), "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    assert (cache_dir / "lines" / "scene_1" / "001_A.wav").exists()
    assert (cache_dir / "manifest.json").exists()
    assert not (out_dir / "cache").exists()
    assert (out_dir / "play.wav").exists()
