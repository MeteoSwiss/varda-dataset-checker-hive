# AGENTS.md

Guidance for coding agents in this repository. The docs are the source of truth; this file points
to them and adds only what they don't say. Read the linked section before changing the area it
covers, and update the docs when behaviour changes.

## Where things are documented

| topic | read |
|---|---|
| what the hive is, the bees, CLI usage, exit codes | [README.md](README.md) (top, *Usage*) |
| findings vs assertions, what belongs in the hive vs the dataset repo | [README.md#findings-not-assertions](README.md#findings-not-assertions), [README.md#what-belongs-where](README.md#what-belongs-where) |
| spec fields | [README.md#spec-fields](README.md#spec-fields); authoritative schema: `src/varda_hive/honeycomb/spec.py`; every field with defaults: `templates/spec.example.yaml` |
| meta-bee checks, meta files, `compare`, reference policy | [docs/meta-bee.md](docs/meta-bee.md) |
| stats-bee sampling, stats files, tolerances, calibration, limits | [docs/stats-bee.md](docs/stats-bee.md) |
| CI gating, `--strict`, publishing workflow, CSCS-CI job | [docs/ci.md](docs/ci.md) |
| adding a check | [README.md#adding-a-check](README.md#adding-a-check) |
| source plug-ins (spot-bee) | `templates/source_plugin_example.py` |
| dev setup, style, commit messages, PR process | [CONTRIBUTING.md](CONTRIBUTING.md), [.github/pull_request_template.md](.github/pull_request_template.md) |

## Commands

Setup, tests and quality tools: [README.md#development](README.md#development) and
[CONTRIBUTING.md](CONTRIBUTING.md#code-style-and-quality). CI (`.github/workflows/CI_test.yaml`)
runs, on Python 3.12 and 3.13:

```bash
uv sync --locked
uv run pytest -s --cov=varda_hive --cov-fail-under=80
uv run pylint -rn --output-format=parseable src/varda_hive
uv run mypy src/varda_hive
uv run yapf --diff --recursive src tests      # fix with: uv run yapf -r -i src tests
```

Single test: `uv run pytest tests/test_bees.py::<test_name>`. The `faulty` marker is deselected by
default (`addopts` in `pyproject.toml`); `uv run pytest -m faulty` runs on Varda-tiny on /scratch
and skips when the data is missing (paths overridable via `HIVE_FAULTY_DIR` / `HIVE_CLEAN_DS`, see
`tests/test_faulty_tiny.py`).

## Architecture in brief

- `src/varda_hive/honeycomb/`: the shared core (dataset access, `Finding`/`Report`, spec, check
  registry and runner, seeded sampling, `Source` interface). Module docstrings explain each part.
- `src/varda_hive/bees/<bee>/checks.py`: a bee's checks, registered with `@check("<bee>")`. The
  registry imports a bee's `checks` module lazily (`honeycomb/registry.py:_load_bee`); bee-specific
  subcommands (`dump`, `compare`, `calibrate`) live in that bee's own `cli.py`. The top-level
  `src/varda_hive/cli.py` docstring describes the entry points and exit codes.
- **Bees must stay independent**: a bee may import the honeycomb but never another bee.
  `tests/test_independence.py` enforces this by running each bee with the others unimportable.
- `honeycomb/dataset.py` reads the zarr v2 store directly, not through anemoi-datasets
  (optional extra), so checks see exactly what is on disk. meta-bee never reads `data`.

## Conventions not obvious from the docs

- Checks yield findings and never assert or raise for a dataset fault; the runner turns a crash
  into an ERROR finding. A check that lacks a spec field reports "skipped" (fails under `--strict`).
- Every new check needs a test that passes on the clean synthetic dataset (`tests/conftest.py`
  `build_dataset`) and fires on a broken one, as in `tests/test_bees.py`.
- When adding a check or spec field, update the checks table / spec fields in the matching doc
  (`README.md`, `docs/<bee>.md`) and `templates/spec.example.yaml`.
- New files carry the BSD-3-Clause license header used in existing sources.
- spot-bee is still a template (only `matches_source`); see the README status note.
