from __future__ import annotations

from pathlib import Path

import click

from .assembly import assemble_play, assemble_scene
from .cache import LineCache, default_output_dir, is_line_stale, load_manifest, save_manifest
from .script import find_line, load_script, parse_selectors, render_numbered_markdown
from .synthesis import load_tts_model, synthesize_line


def _output_dir(input_path: Path, output_dir: str | None) -> Path:
    return Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(input_path)


def _validate_character_options(script_characters: set[str], *character_sets: set[str]) -> None:
    for characters in character_sets:
        unknown = sorted(characters - script_characters)
        if unknown:
            raise click.ClickException(f"unknown character option: {', '.join(unknown)}")


@click.group()
def main() -> None:
    """Render play scripts with cached line audio."""


@main.command()
@click.argument("input_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False), default=None)
@click.option("--silence-character", multiple=True)
@click.option("--silence-multiplier", type=float, default=1.2, show_default=True)
@click.option("--skip-character", multiple=True)
@click.option("--focus-character", default=None)
@click.option("--focus-channel", type=click.Choice(["left", "right"]), default="left", show_default=True)
@click.option("--device", default="mps", show_default=True)
def render(
    input_yaml: Path,
    output_dir: str | None,
    silence_character: tuple[str, ...],
    silence_multiplier: float,
    skip_character: tuple[str, ...],
    focus_character: str | None,
    focus_channel: str,
    device: str,
) -> None:
    """Render cached lines, scene WAVs, and the full play WAV."""
    try:
        script = load_script(input_yaml)
        out_dir = _output_dir(input_yaml, output_dir)
        cache = LineCache(out_dir)
        manifest = load_manifest(cache)
        script_characters = set(script.voices) | {line.character for line in script.lines}
        silence_characters = set(silence_character)
        skip_characters = set(skip_character)
        focus_characters = {focus_character} if focus_character else set()
        _validate_character_options(script_characters, silence_characters, skip_characters, focus_characters)

        stale_lines = [line for line in script.lines if is_line_stale(line, cache, manifest)]
        if stale_lines:
            click.echo(f"Rendering {len(stale_lines)} stale or missing line(s) on {device}.")
            model = load_tts_model(device)
            for line in stale_lines:
                click.echo(f"[{line.selector}] {line.character}: {line.normalized_text}")
                synthesize_line(model, line, cache, manifest)
            save_manifest(cache, manifest)
            sr = model.sr
        else:
            click.echo("Line cache is current.")
            if not script.lines:
                raise click.ClickException("script contains no lines")
            import torchaudio as ta

            _, sr = ta.load(str(cache.line_path(script.lines[0])))

        scene_paths = [
            assemble_scene(
                script,
                scene_name,
                cache,
                out_dir,
                sr,
                silence_characters=silence_characters,
                silence_multiplier=silence_multiplier,
                skip_characters=skip_characters,
                focus_character=focus_character,
                focus_channel=focus_channel,
            )
            for scene_name in script.play
        ]
        play_path = assemble_play(scene_paths, out_dir, sr)
        script_md_path = out_dir / "script.md"
        script_md_path.write_text(render_numbered_markdown(script), encoding="utf-8")
        click.echo(f"Saved play: {play_path}")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("rerender-lines")
@click.argument("input_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("line", nargs=-1, required=True)
@click.option("--output-dir", type=click.Path(file_okay=False), default=None)
@click.option("--device", default="mps", show_default=True)
def rerender_lines(input_yaml: Path, line: tuple[str, ...], output_dir: str | None, device: str) -> None:
    """Force regeneration of selected cached line WAVs."""
    try:
        script = load_script(input_yaml)
        out_dir = _output_dir(input_yaml, output_dir)
        cache = LineCache(out_dir)
        manifest = load_manifest(cache)
        selected = [find_line(script, selector) for selector in parse_selectors(line)]
        model = load_tts_model(device)
        for selected_line in selected:
            click.echo(f"[{selected_line.selector}] {selected_line.character}: {selected_line.normalized_text}")
            synthesize_line(model, selected_line, cache, manifest)
        save_manifest(cache, manifest)
        click.echo(f"Rerendered {len(selected)} line(s).")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("script")
@click.argument("input_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False), default=None)
def script_command(input_yaml: Path, output_dir: str | None) -> None:
    """Write a numbered Markdown reference for line selectors."""
    try:
        script = load_script(input_yaml)
        out_dir = _output_dir(input_yaml, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "script.md"
        out_path.write_text(render_numbered_markdown(script), encoding="utf-8")
        click.echo(f"Saved script: {out_path}")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
