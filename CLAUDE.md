# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AGENTS.md is the single source of agent guidance for this repository (commands, where each topic is
documented, architecture, conventions). It is imported here so that Claude Code and other agents
read the same instructions; edit AGENTS.md, not this file, when they change.

@AGENTS.md

## Additions for Claude Code

- CI runs on Python 3.12 and 3.13. To run the same commands on 3.13 locally without disturbing
  `.venv`, use a second environment (uv puts a `.gitignore` inside it, so git ignores it):
  `UV_PROJECT_ENVIRONMENT=$PWD/.venv-313 uv sync --locked --python 3.13`, then prefix the CI
  commands with the same `UV_PROJECT_ENVIRONMENT=...`.
- To try an unpushed checkout from a dataset repo (e.g. `rea-l-ch1-anemoi-dataset`), overlay it with
  `uv run --no-sync --with-editable "<path to this repo>[fdb]" hive ...` after `unset VIRTUAL_ENV`.
  `--with <path>` reuses a cached build and silently runs an old hive. The dataset repos' own
  `run_hive.sh` installs `HIVE_REF` from GitHub, so it only sees pushed commits.
