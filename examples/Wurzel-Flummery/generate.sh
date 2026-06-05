#!/bin/bash

uv run playread render wurzel-flummery.yaml --output-dir full-cast
uv run playread render wurzel-flummery.yaml --output-dir no-viola --silence-character VIOLA
uv run playread render wurzel-flummery.yaml --focus-character RICHARD --focus-channel right --output-dir richard-on-right


