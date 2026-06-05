# playread

`playread` renders simple play-script YAML files into spoken audio. It uses
Chatterbox TTS to synthesize each line, then assembles audio files for each
scene and for the full play.

It includes options for line-learning practice, such as silencing a single
character or putting one character on one audio channel, for example left, and
all other characters on the other channel. With the latter option, you can keep
one earphone in for practice and add the other earphone when you need a prompt.

Generation can take a while depending on the accelerator device, so generated
lines are cached. This lets you render several versions of the same play quickly
without regenerating unchanged dialogue.

## Requirements

- Python 3.14 or newer
- `uv`
- FFmpeg must be available on the system
- A device supported by your installed PyTorch build
  - macOS Apple Silicon: `--device mps` is the default
  - CUDA: `--device cuda`
  - CPU fallback: `--device cpu`

Install or sync the project dependencies:

```sh
uv sync
```

Run the CLI through `uv`:

```sh
uv run playread --help
```

## Script Format

A script is a YAML file with:

- `voices`: a mapping from character names to voice prompt files and optional
  generation settings
  - the settings are passed to Chatterbox TTS; see
    https://github.com/resemble-ai/chatterbox
  - sample voices from the "CSTR VCTK Corpus" are included in the examples; see
    https://datashare.ed.ac.uk/handle/10283/2950
  - any roughly 10-second audio prompt can be used; see Chatterbox for details
- `play`: the ordered list of scenes to render.
- one list per scene, where every item maps one or more speakers to a line.

Example:

```yaml
voices:
  ALICE:
    file: voices/alice.wav
    cfg_weight: 0.5
    exaggeration: 0.45
  BOB:
    file: voices/bob.wav
  narrator:
    file: voices/narrator.wav
    cfg_weight: 0.3
    exaggeration: 0.35

play:
  - scene_1

scene_1:
  - narrator: "Alice and Bob enter."
  - ALICE: "Hello, Bob."
  - BOB: "Hello, Alice."
  - ALICE and BOB: "Hello, everyone!"
```

The composite line is generated with both voices and mixed into one cached line.

## Rendering

Render a complete play:

```sh
uv run playread render examples/Wurzel-Flummery/wurzel-flummery.yaml
```

By default this writes:

- line cache: `<yaml-dir>/<yaml-stem>-cache/`
- rendered output: `<yaml-dir>/<yaml-stem>-out/`
- scene WAVs: `<output-dir>/scenes/<scene>.wav`
- full play WAV: `<output-dir>/play.wav`
- numbered line reference: `<output-dir>/script.md`

Use a custom output directory:

```sh
uv run playread render play.yaml --output-dir out/full-cast
```

Use a custom cache directory:

```sh
uv run playread render play.yaml --cache-dir shared-cache --output-dir out/full-cast
```

Render a rehearsal version with one character replaced by silence:

```sh
uv run playread render play.yaml \
  --output-dir out/no-alice \
  --silence-character ALICE
```

Skip a character entirely:

```sh
uv run playread render play.yaml \
  --output-dir out/without-narrator \
  --skip-character narrator
```

Create a stereo focus mix where one character is routed to one side and all
other lines are routed to the other side:

```sh
uv run playread render play.yaml \
  --output-dir out/alice-left \
  --focus-character ALICE \
  --focus-channel left
```

## Regenerating Single Lines

TTS output can occasionally be glitchy, even when the input line and voice are
fine. `playread` lets you regenerate individual cached lines instead of
rerendering the whole play.

To find the line identifiers, generate a numbered Markdown reference:

```sh
uv run playread script play.yaml
```

This writes `script.md` to the default output directory, or to `--output-dir` if
provided. Line identifiers use `scene:number`, for example `scene_1:12`.

Regenerate one or more cached lines by passing those identifiers to
`rerender-lines`:

```sh
uv run playread rerender-lines play.yaml scene_1:3 scene_2:12
```

This updates only the cached line WAVs. It does not rebuild existing scene WAVs
or `play.wav`. After regenerating lines, run `render` again to assemble updated
scene and play outputs:

```sh
uv run playread render play.yaml
```

Use the same cache override when rerendering lines from a non-default cache:

```sh
uv run playread rerender-lines play.yaml scene_1:3 --cache-dir shared-cache
uv run playread render play.yaml --cache-dir shared-cache
```

## Cache Behavior

The line cache stores one WAV per script line plus a manifest:

```text
play-cache/
  manifest.json
  lines/
    scene_1/
      001_narrator.wav
      002_ALICE.wav
```

Cache keys are portable between machines. They are based on:

- scene and line number
- character and line text
- normalized text sent to TTS
- generation settings
- YAML-relative prompt file paths
- SHA-256 hashes of prompt file contents

Absolute paths and file modification timestamps are not part of the cache key.
Changing prompt file bytes invalidates affected lines; moving the project to a
different machine should not.

## Development

Run the test suite:

```sh
uv run pytest
```

Useful focused checks:

```sh
uv run pytest tests/test_cache.py tests/test_cli.py
```
