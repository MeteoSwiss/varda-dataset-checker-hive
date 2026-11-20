# varda-dataset-checker-hive

[![CI_test](https://github.com/MeteoSwiss/varda-dataset-checker-hive/actions/workflows/CI_test.yaml/badge.svg)](https://github.com/MeteoSwiss/varda-dataset-checker-hive/actions/workflows/CI_test.yaml)

A hive of slim tools ("bees") that check anemoi datasets for faults. Each bee covers one layer:

| command | layer | reads |
|---|---|---|
| `meta-bee` | metadata and structure: internal consistency, variables, dates, frequency, grid, per-variable metadata, reference diff ([docs](docs/meta-bee.md)) | group attributes (`.zattrs` or `zarr.json`) and small arrays only |
| `stats-bee` | values: NaNs, constant fields, physical ranges, stored vs recomputed statistics, reference statistics ([docs](docs/stats-bee.md)) | sampled time steps |
| `spot-bee` | sampled (time, variable, grid points) vs the original source, e.g. GRIB via a grib-index DB: what is identical, what differs ([docs](docs/spot-bee.md)) | sampled time steps + source |
| `hive` | runs all bees and merges their findings | |

All of them share the same core, the `honeycomb`:
- dataset access
- `Finding` / `Report`
- the spec schema
- the check registry
- seeded sampling
- the `Source` interface

> Status: all three bees are implemented. spot-bee ships two GRIB sources: anemoi grib-index
> databases (`GribIndexSource`, `grib` extra) and FDB (`FdbSource`, `fdb` extra). Derived variables, computed forcings and constant fields are
> reported as not comparable, see [docs/spot-bee.md](docs/spot-bee.md#derived-variables-and-forcings).

## Usage

```bash
uv sync                                                # add --extra grib or --extra fdb for spot-bee's sources
uv run hive <dataset.zarr> --spec <spec.yaml>          # all bees
uv run hive <dataset.zarr> --spec <spec.yaml> --strict --out hive/   # + report.md, report.json, ref.yaml
uv run meta-bee check <dataset.zarr> --spec <spec.yaml>   # one bee
uv run hive --list                                     # available checks
uv run hive compare hive/ref.yaml checks/data.ref.yaml # two dumps: what is identical, what differs
uv run stats-bee calibrate <dataset.zarr> --spec <spec.yaml> -o tolerances.yaml   # calibrated tolerances
uv run spot-bee check <dataset.zarr> --spec <spec.yaml>   # sampled values vs the source
uv run hive <dataset.zarr> --spec <spec.yaml> --format json --fail-on warning   # or --format markdown
```

A check is a **dump plus a comparison**. Each bee builds its dump once: the facts it compares,
i.e. meta-bee the metadata and structure, stats-bee the statistics of the seeded sample (defined by
the spec alone, so every rebuild that meets the spec samples the same dates). Its checks compare the dump with the spec (expected
values) and with the reference (the accepted dump). `--out DIR` writes the reports (`report.md`,
`report.json`) and the dump (`ref.yaml`, one section per bee), also when checks fail; without
`--out` nothing is written. `ref.yaml` is the reference to commit (see
[Setting up a dataset](#setting-up-a-dataset)). `hive compare` compares two dumps without the data,
with the same code as the reference checks.

Exit codes:

| code | meaning |
|---|---|
| 0 | no finding reaches `--fail-on` (default `error`); with `--strict`, also no check was skipped |
| 1 | the dataset failed the checks (a crash inside one check counts here, as an ERROR finding) |
| 2 | usage or configuration error: missing dataset, invalid spec, unknown `--select` name |
| 3 | internal error of the tool |

## Release CI

Gate a release on the dataset's spec and its committed reference, in the dataset repo's CI:

```bash
uv run hive $DATASET --spec checks/<dataset>.yaml --strict --out hive/
```

Keep `hive/` as the job's artifact: `report.md` for the merge request, `report.json` for tools,
`ref.yaml` to accept as the new reference.

`--strict` fails when a check is skipped, so a misconfigured job can't stay green. See
[docs/ci.md](docs/ci.md) for the publishing workflow, the flags and an example CSCS-CI job built on
[varda-cscs-ci-workflows](https://github.com/MeteoSwiss/varda-cscs-ci-workflows).

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
- **The source configuration**: where spot-bee gets the original values. For datasets built from
  GRIB with anemoi's `grib-index` source, the generic plug-in in the hive reads the same index DBs.
  The dataset repo only adds the mapping of renamed or rescaled variables:

  ```yaml
  source:
    plugin: varda_hive.sources.grib_index:GribIndexSource   # or file.py:Class in the dataset repo
    args: {root: ..., indexes: {ana: grib-index-ana.db, fg: grib-index-fg.db}, variables: {...}}
  ```

  Other sources need a plug-in, a class with `fetch(variable, date) -> np.ndarray | None`, see
  [docs/spot-bee.md](docs/spot-bee.md#source-plug-ins) and `templates/source_plugin_example.py`.

## Setting up a dataset

A dataset needs two files: the spec, written by hand once, which every bee reads, and the ref
file, the dump of the accepted version, written by `--out`. All three workflows are the same
command:

| workflow | command | then |
|---|---|---|
| reference exists (CI, release) | `hive data.zarr --spec S --strict --out hive/` | spec and reference checks; exit 0 or 1 |
| first reference | the same | the reference checks report "skipped: reference … not found" (exit 1 under `--strict`); review, commit `hive/ref.yaml` as the spec's `reference` |
| intended change | the same | `hive/report.md` lists the differences to the reference; after review, `cp hive/ref.yaml checks/<dataset>.ref.yaml` |

1. **Copy the template** into the dataset repo:

   ```bash
   cp templates/spec.example.yaml <dataset-repo>/checks/<dataset>.yaml
   ```

2. **Fill in the expected structure** (`start_date`, `end_date`, `frequency`, `grid_points`,
   `resolution`, `bbox`, `missing_dates`) and the `variables` with what is expected of each: the
   variables_metadata keys (`param`, `levtype`, `levelist`, `process`, ...), `constant: true`,
   `allow_nans: true`. Take them from the anemoi **recipe**, not from the dataset you are about to
   check. A spec copied from the dataset agrees with it by construction and catches nothing in
   that build.

3. **Add the physical ranges** (`min`, `max`, `units`) to the variables. Keep the bounds loose:
   they should catch unit errors and garbage, not unusual weather. Variables without a range are
   listed as INFO. YAML anchors share a range between variables, e.g. the levels of one field.

4. **Choose the sampling.** `n_dates` random dates over the spec's period (`start_date` to
   `end_date` every `frequency`), `n_points` grid points on each of them for mean and stdev
   (10k–100k, or `all`; min/max and NaN/inf always cover the full domain), plus `include_dates` for
   dates you always want checked. The sample depends on the spec only, so every rebuild is sampled
   on the same dates and compares with the reference; changing the sampling or the period is an
   intended change (accept the new `ref.yaml`). `all` for both gives an exact full pass, affordable only for
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

6. **Accept the first reference.** Set `reference: <dataset>.ref.yaml` in the spec before the file
   exists; the reference checks then skip ("reference … not found"), they are not a configuration
   error. Once this version of the dataset is accepted:

   ```bash
   uv run hive data.zarr --spec checks/<dataset>.yaml --strict --out hive/
   cp hive/ref.yaml checks/<dataset>.ref.yaml
   ```

   Commit the spec and the ref file. Accepting an intended change later is the same copy, after
   reviewing `hive/report.md`.

7. **Gate releases in CI** with `--strict`, see [Release CI](#release-ci). Each rebuild is then
   compared with the accepted version. The ref file holds meta-bee's dump (`meta`) and stats-bee's
   dump (`stats`), see [docs/meta-bee.md](docs/meta-bee.md) and
   [docs/stats-bee.md](docs/stats-bee.md#stats-files).

### Spec fields

Every field is optional. A check that needs a missing field reports "skipped", which fails the run
under `--strict`. Unknown keys are rejected, so a typo fails when the spec is loaded. Relative
paths resolve against the spec file's directory.

| field | used by | meaning |
|---|---|---|
| `name` | none | for humans |
| `variables` | all bees | exact set of variables (meta-bee): a list, or a mapping `{name: {...}}` with the per-variable fields below |
| `variables.<name>.min`, `.max`, `.units` | stats-bee, meta-bee | physical range (`physical_ranges`); units, compared with variables_metadata |
| `variables.<name>.allow_nans` | meta-bee, stats-bee | `true`: NaNs allowed in this variable (default `false`) |
| `variables.<name>.constant` | stats-bee | `true`: never changes in time (used when the dataset lists no `constant_fields`) |
| `variables.<name>.<key>` | meta-bee | any other key (`param`, `levtype`, `levelist`, `process`, ...): expected in `variables_metadata.<name>`, at its top or in `mars` |
| `anchors` | none | a place for YAML anchors shared by several variables |
| `start_date`, `end_date`, `frequency` | meta-bee | expected first and last date, and step |
| `grid_points`, `resolution`, `bbox` | meta-bee | expected grid size, `resolution` attribute, extent `{lat_min, lat_max, lon_min, lon_max, tolerance}` |
| `allow_nans` | meta-bee, stats-bee | `true`, `false` or a list of variables; with a `variables` mapping, set it per variable instead |
| `missing_dates` | meta-bee | expected `missing_dates` |
| `reference` | meta-bee, stats-bee | ref file (`ref.yaml` of `--out`) of the accepted version, or a zarr (meta-bee only); if it does not exist yet, the reference checks are skipped (the first reference) |
| `reference_policy` | meta-bee | `{ignore, error, warning, info: [path globs]}`: severity of differences to the reference |
| `calibrated_tolerances` | stats-bee | tolerance file for `statistics_consistent`, written by `stats-bee calibrate` |
| `sampling` | stats-bee, spot-bee | `seed`, `n_dates` and `n_points` (int or `all`), `include_dates` (dates); drawn on the spec's period (the dataset's dates without one); the same dates x points for both bees |
| `tolerances` | stats-bee, spot-bee | `statistics_rtol`, `z_threshold`, `compare_atol_stdev` (stats-bee); `source_rtol`, `source_atol`, `source_per_variable: {VAR: {rtol, atol}}` (spot-bee) |
| `source` | spot-bee | source plug-in and its args, see [docs/spot-bee.md](docs/spot-bee.md#source-plug-ins) |
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

Datasets in zarr format 2 and 3 are both supported; the hive needs zarr-python >= 3.1.

```bash
uv sync && uv run pytest
uv run pytest -m faulty    # all bees on Varda-tiny and its faulty variants (needs /scratch)
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
