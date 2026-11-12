# varda-dataset-checker-hive

[![CI_test](https://github.com/MeteoSwiss/varda-dataset-checker-hive/actions/workflows/CI_test.yaml/badge.svg)](https://github.com/MeteoSwiss/varda-dataset-checker-hive/actions/workflows/CI_test.yaml)

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
uv run meta-bee check <dataset.zarr> --spec <spec.yaml>   # one bee
uv run hive --list                                     # available checks
uv run stats-bee dump <dataset.zarr> --spec <spec.yaml> -o stats.yaml   # stats file
uv run stats-bee compare stats.yaml reference.yaml     # what is identical, what differs
uv run stats-bee calibrate <dataset.zarr> --spec <spec.yaml> -o tolerances.yaml   # calibrated tolerances
uv run hive <dataset.zarr> --spec <spec.yaml> --format json --fail-on warning
```

Exit codes:

| code | meaning |
|---|---|
| 0 | no finding reaches `--fail-on` (default `error`); with `--strict`, also no check was skipped |
| 1 | the dataset failed the checks (a crash inside one check counts here, as an ERROR finding) |
| 2 | usage or configuration error: missing dataset, invalid spec or ranges file, unknown `--select` name |
| 3 | internal error of the tool |

## Release CI

Gate a release on the dataset's spec and its committed reference stats file:

```bash
uv run hive $DATASET --spec checks/<dataset>.yaml --strict --report hive.json
```

- `--strict` fails when a check is skipped, e.g. because the spec has no `ranges_file`. Otherwise a
  misconfigured job stays green.
- `--report` writes the JSON report as a CI artifact, while the text goes to the log.
- Known, accepted warnings can be demoted to INFO in the spec, so `--fail-on warning` stays usable:
  `ignore_warnings: {statistics_consistent: [insolation]}`.

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

## Setting up a dataset

The spec is written by hand, once per dataset, and every bee reads the same file. Only the
reference stats file is generated.

1. **Copy the templates** into the dataset repo:

   ```bash
   cp templates/spec.example.yaml            <dataset-repo>/checks/<dataset>.yaml
   cp templates/physical_ranges.example.yaml <dataset-repo>/checks/physical_ranges.yaml
   ```

2. **Fill in the expected structure** (`variables`, `start_date`, `end_date`, `frequency`,
   `grid_points`, `constant_fields`) from the anemoi **recipe**, not from the dataset you are
   about to check. A spec copied from the dataset agrees with it by construction and catches
   nothing in that build.

3. **Write the ranges file.** Keep the bounds loose: they should catch unit errors and garbage,
   not unusual weather. Variables without a range are listed as INFO.

4. **Choose the sampling.** `n_dates` random dates over the whole time range, `n_points` grid
   points on each of them (10k–100k, or `all` for the full domain), plus `include_dates` for time
   indices you always want checked. `all` for both gives an exact full pass, affordable only for
   small datasets. Leave the `tolerances` at their defaults unless a check tells you otherwise.
   Optionally calibrate them for this sampling with `stats-bee calibrate`, see
   [docs/stats-bee.md](docs/stats-bee.md#calibrated-tolerances).

5. **Check the dataset** and review every finding:

   ```bash
   uv run hive data.zarr --spec checks/<dataset>.yaml
   ```

   Fix the dataset or the spec. Add a warning to `ignore_warnings` only once you have understood
   and accepted it, e.g. `statistics_consistent` on insolation, whose diurnal cycle a small
   sample can't resolve.

6. **Write the reference stats file** once this version of the dataset is accepted:

   ```bash
   uv run stats-bee dump data.zarr --spec checks/<dataset>.yaml -o checks/<dataset>.stats.yaml
   ```

   Uncomment `reference_stats: <dataset>.stats.yaml` in the spec, and commit the spec, the ranges
   file and the stats file.

7. **Gate releases in CI** with `--strict`, see [Release CI](#release-ci). Each rebuild is then
   compared exactly with the accepted version. See
   [docs/stats-bee.md](docs/stats-bee.md#stats-files) for the stats file.

### Spec fields

Every field is optional. A check that needs a missing field reports "skipped", which fails the run
under `--strict`. Unknown keys are rejected, so a typo fails when the spec is loaded. Relative
paths resolve against the spec file's directory.

| field | used by | meaning |
|---|---|---|
| `name` | none | for humans |
| `variables` | meta-bee | exact set of variables |
| `start_date`, `end_date`, `frequency`, `grid_points` | meta-bee (planned) | expected dates and grid |
| `constant_fields` | stats-bee | fields that must not change in time (used when the dataset lists none) |
| `ranges_file` | stats-bee | YAML of `{variable: {min, max, units}}`, see `templates/physical_ranges.example.yaml` |
| `reference_stats` | stats-bee | stats file of the accepted version, written by `stats-bee dump` |
| `calibrated_tolerances` | stats-bee | tolerance file for `statistics_consistent`, written by `stats-bee calibrate` |
| `sampling` | stats-bee, spot-bee | `seed`, `n_dates` and `n_points` (int or `all`), `include_dates`; the same dates x points for both bees |
| `tolerances` | stats-bee, spot-bee | `statistics_rtol`, `z_threshold`, `compare_atol_stdev` (stats-bee); `source_rtol`, `source_atol` (spot-bee) |
| `source` | spot-bee | source plug-in, see above |
| `ignore_warnings` | all bees | `{check: [variables]}` (or `"*"` for all variables): warnings demoted to INFO |

`templates/spec.example.yaml` shows every field with its default value, and
`src/varda_hive/honeycomb/spec.py` is the authoritative schema.

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

Quality tools:

```bash
uv run pylint src/varda_hive
uv run mypy src/varda_hive
uv run yapf --diff --recursive src tests
```

## Ownership

| role | who |
|---|---|
| code owner | Daniel Hupp (see `.github/CODEOWNERS`) |
| maintainer | open |

Pull requests are reviewed by the code owner or a maintainer. Questions and bug reports: open an issue
in this repository.
