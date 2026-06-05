from __future__ import annotations

from pathlib import Path

import torch
import torchaudio as ta

from playread.assembly import assemble_scene, estimate_duration_samples
from playread.cache import LineCache
from playread.script import load_script

SR = 1000


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
  - A: "AAAA"
  - B: "BBBBBBBB"
""",
        encoding="utf-8",
    )
    return script_path


def write_line_wavs(script_path: Path, out_dir: Path) -> tuple[object, LineCache]:
    script = load_script(script_path)
    cache = LineCache(out_dir)
    for line, samples, value in zip(
        script.lines, [100, 200], [0.25, 0.75], strict=True
    ):
        path = cache.line_path(line)
        path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(path), torch.full((1, samples), value), SR)
    return script, cache


def test_assemble_default_mono(tmp_path: Path) -> None:
    script, cache = write_line_wavs(write_script(tmp_path), tmp_path / "out")

    out = assemble_scene(script, "scene_1", cache, tmp_path / "out", SR)
    wav, sr = ta.load(str(out))

    assert sr == SR
    assert wav.shape == (1, 300 + 100 + 280 + 200 + 280)


def test_assemble_stereo_focus_left_and_right(tmp_path: Path) -> None:
    script, cache = write_line_wavs(write_script(tmp_path), tmp_path / "out")

    left_path = assemble_scene(
        script,
        "scene_1",
        cache,
        tmp_path / "left",
        SR,
        focus_character="A",
        focus_channel="left",
    )
    right_path = assemble_scene(
        script,
        "scene_1",
        cache,
        tmp_path / "right",
        SR,
        focus_character="A",
        focus_channel="right",
    )
    left, _ = ta.load(str(left_path))
    right, _ = ta.load(str(right_path))

    assert left.shape[0] == 2
    assert torch.all(left[0, 300:400] > 0)
    assert torch.all(left[1, 300:400] == 0)
    assert torch.all(left[1, 680:880] > 0)
    assert torch.all(right[1, 300:400] > 0)
    assert torch.all(right[0, 680:880] > 0)


def test_assemble_silenced_character_uses_cached_duration(tmp_path: Path) -> None:
    script, cache = write_line_wavs(write_script(tmp_path), tmp_path / "out")

    out = assemble_scene(
        script,
        "scene_1",
        cache,
        tmp_path / "out",
        SR,
        silence_characters={"A"},
        silence_multiplier=2.0,
    )
    wav, _ = ta.load(str(out))

    assert wav.shape == (1, 300 + 200 + 280 + 200 + 280)
    assert torch.all(wav[:, 300:500] == 0)


def test_assemble_silenced_character_estimates_duration_without_cache(
    tmp_path: Path,
) -> None:
    script, cache = write_line_wavs(write_script(tmp_path), tmp_path / "out")
    cache.line_path(script.lines[0]).unlink()

    out = assemble_scene(
        script,
        "scene_1",
        cache,
        tmp_path / "out",
        SR,
        silence_characters={"A"},
        silence_multiplier=2.0,
    )
    wav, _ = ta.load(str(out))

    expected_a = int(estimate_duration_samples("AAAA", SR) * 2.0)
    assert wav.shape == (1, 300 + expected_a + 280 + 200 + 280)


def test_assemble_skipped_character_removes_line_and_pause(tmp_path: Path) -> None:
    script, cache = write_line_wavs(write_script(tmp_path), tmp_path / "out")

    out = assemble_scene(
        script, "scene_1", cache, tmp_path / "out", SR, skip_characters={"A"}
    )
    wav, _ = ta.load(str(out))

    assert wav.shape == (1, 300 + 200 + 280)
