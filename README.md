# varda-dataset-checker-hive

A hive of slim tools ("bees") that check anemoi datasets for faults. Each bee covers one layer:

| command | layer | reads |
|---|---|---|
| `meta-bee` | metadata and structure: variables, dates, frequency, grid, reference diff | `.zattrs` and small arrays only |
| `stats-bee` | values: NaNs, constant fields, physical ranges, stored vs recomputed statistics, reference stats file ([docs](docs/stats-bee.md)) | sampled time steps |
| `spot-bee` | sampled (time, variable, grid points) vs the original source | sampled time steps + source |
| `hive` | runs all bees and merges their findings | |

All of them share the same core, the `honeycomb`:
- dataset access
- `Finding` / `Report`
- the spec schema
- the check registry
- seeded sampling
- the `Source` interface

> Status: the stats-bee is implemented; meta-bee and spot-bee still have one template check each
> (`variables_match_spec`, `matches_source`).

## Usage

```bash
uv sync
uv run hive <dataset.zarr> --spec <spec.yaml>          # all bees
uv run meta-bee <dataset.zarr> --spec <spec.yaml>      # one bee
uv run hive --list                                     # available checks
uv run stats-bee dump <dataset.zarr> --spec <spec.yaml> -o stats.yaml   # stats file
uv run stats-bee compare stats.yaml reference.yaml     # what is identical, what differs
uv run hive <dataset.zarr> --spec <spec.yaml> --format json --fail-on warning
```

Exit code is 0 when no finding reaches `--fail-on` (default `error`), and 1 otherwise.

## Findings, not assertions

Checks yield `Finding`s with a severity of `ERROR`, `WARNING` or `INFO`. They never assert.
The same check therefore works from the CLI, in CI, in pytest or in a notebook, and expected
noise can be reported as a warning instead of a failure. Examples of such noise are run-to-run
ICON differences and metadata churn between anemoi-datasets versions. A check that crashes is
reported as an `ERROR` finding, so it can't hide the results of the other checks.

## What belongs where

The hive is generic. Everything dataset-specific lives in the **dataset's own repo**:

- **The spec** (`checks/<dataset>.yaml`): expected variables, dates, frequency, grid, constant
  fields, physical ranges, sampling and tolerances. Every field is optional, and unknown keys are
  rejected. See `templates/spec.example.yaml`.
- **The source plug-in**: a class with `fetch(variable, date) -> np.ndarray | None` that spot-bee
  uses to get the original values, e.g. from GRIB via a grib-index DB for ICH1. The spec names it
  by import path:

  ```yaml
  source:
    plugin: ich1_source.py:GribIndexSource     # or package.module:Class
    args: {ana_db: ..., fg_db: ...}
  ```

  See `templates/source_plugin_example.py`, which reads another zarr.

## Adding a check

```python
from varda_hive.honeycomb import CheckContext, check, error

@check("stats")
def no_negative_precipitation(ctx: CheckContext):
    """Precipitation is never negative."""
    ...
    yield error("negative values", variable="TOT_PREC_1H", time="2020-03-05T04:00:00")
```

Put it in `src/varda_hive/bees/<bee>/checks.py`. Then add a test showing that it passes on a
clean synthetic dataset and fires on a broken one, as in `tests/test_bees.py`.

## Development

```bash
uv sync && uv run pytest
uv run pytest -m faulty    # stats-bee on Varda-tiny and its faulty variants (needs /scratch)
```
