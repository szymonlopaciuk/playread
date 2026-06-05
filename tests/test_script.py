from __future__ import annotations

from pathlib import Path

import pytest

from playread.script import (
    find_line,
    load_script,
    normalize_for_tts,
    parse_selectors,
    render_numbered_markdown,
)


def write_script(tmp_path: Path, body: str, create_voice: bool = True) -> Path:
    if create_voice:
        (tmp_path / "voice.wav").write_bytes(b"voice")
    script_path = tmp_path / "script.yaml"
    script_path.write_text(body, encoding="utf-8")
    return script_path


def valid_script(tmp_path: Path) -> Path:
    return write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
    cfg_weight: 0.4
    exaggeration: 0.6
  NARRATOR:
    file: voice.wav
play:
  - scene_1
scene_1:
  - A: " Hello   there. "
  - NARRATOR: "A pause. Then action."
""",
    )


def test_load_script_validates_and_normalizes(tmp_path: Path) -> None:
    script = load_script(valid_script(tmp_path))

    assert script.play == ["scene_1"]
    assert script.scenes["scene_1"][0].normalized_text == "Hello there."
    assert script.scenes["scene_1"][1].normalized_text == "A pause... Then action."
    assert script.voices["A"].audio_prompt_path == (tmp_path / "voice.wav").resolve()


def test_normalize_for_tts_applies_generic_text_cleanup() -> None:
    assert (
        normalize_for_tts("A", " \u201cHello\u201d \u2014  then\u2013now ")
        == '"Hello" ... then...now'
    )


def test_load_script_resolves_composite_speaker_voices(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
  B:
    file: voice.wav
play:
  - scene_1
scene_1:
  - A and B: "together"
""",
    )

    script = load_script(script_path)
    line = script.lines[0]

    assert line.character == "A and B"
    assert [voice.character for voice in line.voices or ()] == ["A", "B"]


def test_load_script_rejects_unknown_composite_component(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
play:
  - scene_1
scene_1:
  - A and B: "together"
""",
    )

    with pytest.raises(ValueError, match="composite speaker component"):
        load_script(script_path)


def test_load_script_rejects_malformed_line_entry(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
play:
  - scene_1
scene_1:
  - A: "one"
    B: "two"
""",
    )

    with pytest.raises(ValueError, match="one character-to-line mapping"):
        load_script(script_path)


def test_load_script_rejects_missing_scene(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
play:
  - scene_1
""",
    )

    with pytest.raises(ValueError, match="scene_1 must be a list"):
        load_script(script_path)


def test_load_script_rejects_unknown_character(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: voice.wav
play:
  - scene_1
scene_1:
  - B: "no voice"
""",
    )

    with pytest.raises(ValueError, match="no voice configured"):
        load_script(script_path)


def test_load_script_rejects_missing_voice_file(tmp_path: Path) -> None:
    script_path = write_script(
        tmp_path,
        """
voices:
  A:
    file: missing.wav
play:
  - scene_1
scene_1:
  - A: "hello"
""",
        create_voice=False,
    )

    with pytest.raises(ValueError, match="voices.A file does not exist"):
        load_script(script_path)


def test_parse_and_find_selectors(tmp_path: Path) -> None:
    script = load_script(valid_script(tmp_path))
    selectors = parse_selectors(["scene_1:1", "scene_1:2"])

    assert [selector.number for selector in selectors] == [1, 2]
    assert find_line(script, selectors[1]).character == "NARRATOR"

    with pytest.raises(ValueError, match="SCENE:NUMBER"):
        parse_selectors(["scene_1"])


def test_render_numbered_markdown_includes_selectors(tmp_path: Path) -> None:
    script = load_script(valid_script(tmp_path))

    markdown = render_numbered_markdown(script)

    assert "`scene_1:1` **A**" in markdown
    assert "`scene_1:2` **NARRATOR**" in markdown
