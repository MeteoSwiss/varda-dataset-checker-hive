# spot-bee

spot-bee compares a seeded sample of (date, variable, grid points) of an anemoi dataset with the
original source data, e.g. the GRIB files the dataset was built from. For every sampled variable it
reports whether the values are identical to the source, within tolerance, or different. meta-bee
and stats-bee check a dataset against itself and against an accepted version. spot-bee is the only
bee that checks it against the data it was built from, so it also catches faults that
leave the statistics unchanged, such as a field rolled in space or two variables swapped.

```bash
uv sync --extra grib                                            # eccodes, for the grib-index source
uv run spot-bee check data.zarr --spec checks/data.yaml
uv run spot-bee check data.zarr --spec checks/data.yaml --format json   # per-variable details
```

## Check

| check | needs | catches |
|---|---|---|
| `matches_source` | `source` | sampled values that differ from the source; variables the source does not have |

Findings of `matches_source`:

| severity | when |
|---|---|
| ERROR | values of a variable differ from the source on a date: one finding per (variable, date) with `n_differ`, `n_nan_mismatch`, `max_abs_diff`, the first differing grid points and, for ensembles, the members |
| ERROR | the source failed for a variable (e.g. two GRIB messages match, wrong grid size): once per variable, which is then not compared further |
| WARNING | the source has no data for a variable on some sampled dates, with the dates |
| INFO | a variable is not comparable with the source, with the reason (see [derived variables](#derived-variables-and-forcings)) |
| INFO | `spot summary: 95 identical, 0 within tolerance, 0 differ, 19 not comparable, 0 unavailable (22 dates × 10000 points)` |

The `details` of the summary list the variables of each group, with the largest absolute difference
of each and, for differing variables, the dates. Use `--format json` or `--report` to get them.

Values count as **identical** when they are equal in the dataset's precision: the source values are
cast to the dataset dtype (float32) first, so a GRIB value decoded as float64 that was stored
unchanged is identical. **Within tolerance** means `|dataset - source| <= source_atol + source_rtol *
|source|` everywhere, and NaN in both counts as equal.

## Sampling

spot-bee uses the spec's `sampling`, the same as stats-bee:
- **Dates:** the stratified random dates plus `include_dates`, without the missing dates.
- **Points:** `n_points` grid points, the same on every date, or `all`.

Both bees therefore check the same cells. Reading the source is usually the cost: one GRIB message
per (variable, date), whatever `n_points` is. `n_points: all` costs little more than a subset and
also finds faults on a few grid points, such as spikes.

## Spec fields

```yaml
sampling: {seed: 0, n_dates: 20, n_points: all, include_dates: []}
tolerances:
  source_rtol: 1.0e-4
  source_atol: 1.0e-4
  source_per_variable:                 # overrides of rtol and/or atol
    TOT_PREC_1H: {atol: 1.0e-6}
source:
  plugin: varda_hive.sources.grib_index:GribIndexSource   # or file.py:Class, relative to the spec
  args: {...}                                             # passed to the constructor
```

Without `source`, the check reports "skipped", which fails under `--strict`.

## Source plug-ins

A source is a class named by `source.plugin`: `package.module:Class`, or `path/file.py:Class`
relative to the spec file. It is built with `Class(**source.args)`. If its constructor takes a
`base_dir` argument, it also gets the spec file's directory, to resolve relative paths in `args`.
The contract (`src/varda_hive/honeycomb/sources.py`):

| member | required | does |
|---|---|---|
| `fetch(variable, date) -> np.ndarray \| None` | yes | the field at `date` in dataset grid order, shape `(grid,)` or `(ensemble, grid)`; `None` if the source has no data for it |
| `raise NotComparable(reason)` from `fetch` | no | the variable cannot be compared one-to-one; reported once as INFO and not fetched again |
| `bind(dataset)` | no | called once before the first `fetch`, e.g. to read `variables_metadata` |

Any other exception from `fetch` gives one ERROR for that variable. The other variables are still
compared.

Two plug-ins ship with the hive:

- `varda_hive.sources.grib_index:GribIndexSource`: GRIB files through anemoi `grib-index`
  databases, see below.
- `templates/source_plugin_example.py:ZarrSource`: another anemoi zarr, e.g. a trusted copy.

A dataset with another kind of source (FDB/polytope, a different archive) can bring its own plug-in
in its repo.

### GribIndexSource

`anemoi-datasets grib-index` writes an SQLite index of GRIB messages. It has a `paths(id, path)`
table, and a `grib_index` table with the byte offset, the length and the keys of every message
(`valid_datetime`, `shortName`, `level`, `typeOfLevel`, ...). The same indexes the dataset build
reads can drive spot-bee, so nothing has to be re-indexed.

`GribIndexSource` finds the message of a (variable, date) by `valid_datetime` and a set of
key = value matches. It reads its bytes and decodes them with eccodes, so it needs the `grib` extra.
It runs one query per date and index, so a sample of ~100 variables per date stays fast.

```yaml
source:
  plugin: varda_hive.sources.grib_index:GribIndexSource
  args:
    root: /scratch/mch/huppd/varda/ich1-analysis-anemoi-dataset   # base of the DB paths and of the GRIB paths in them
    indexes:                                   # name -> DB, or a list of DBs (e.g. per-day indexes)
      ana: ml_test/processed/grib-index-ana.db
      fg: ml_test/processed/grib-index-fg.db
    default_index: ana                         # default: the first one
    variables:                                 # overrides of the default mapping
      W_SO: {index: fg, match: {shortName: W_SO, "level:d": "0.01"}}
      TOT_PREC_1H: {index: fg, match: {shortName: TOT_PREC}, scale: 0.001}   # renamed and rescaled in the recipe
      VMAX_10M_1H: {index: fg, match: {shortName: VMAX_10M}}
    skip:                                      # not comparable, with the reason
      TOT_PREC_6H: 6 h accumulation of TOT_PREC
    # pl_type_of_level: isobaricInhPa
```

**Mapping.** Without an override, a variable's messages come from its
`variables_metadata[var].mars` entry:

| mars entry | matched keys |
|---|---|
| `levtype: pl` | `shortName = param`, `level = levelist`, `typeOfLevel = pl_type_of_level` |
| any other | `shortName = param` |

An override gives:

| key | meaning | default |
|---|---|---|
| `index` | which index to search | `default_index` |
| `match` | the index columns to match | |
| `scale`, `offset` | applied as `value * scale + offset` | `1`, `0` |

The `match` keys must be columns of the index. Wrong keys fail when the source is bound, with an
ERROR that names the variable. Index values are text, so write numbers the way the index stores them
(`"0.01"`, `850`).

**Not comparable without an override:**
- variables in `skip`
- computed forcings (`computed_forcing`)
- derived fields (`process`, e.g. accumulations)
- constant fields (`constant_in_time`): they are read from a separate file, not from the
  time-indexed archive
- variables without mars metadata

**Errors** that the ICH1 tests used to pass silently:
- more than one message matching a (variable, date)
- a message whose `numberOfDataPoints` differs from the dataset grid

**Ensembles:** with several ensemble members, `number` = 0, 1, ... is matched as well.

GRIB values are returned in the order of the message. That is the dataset grid order for an
unstructured ICON grid read with anemoi's `grid_definition: icon`, which keeps the GRIB order. A
recipe that crops, thins or reorders the grid needs its own plug-in.

## Derived variables and forcings

Some dataset variables are not a GRIB field. spot-bee reports them as "not comparable" (INFO), not
as missing. Each kind is covered by another check:

| kind | examples (ICH1) | covered by |
|---|---|---|
| accumulations and other derived fields | `TOT_PREC_6H`, `ssrd_6h`, `strd_1h` | stats-bee's reference comparison; the derivation tests in the dataset repo (`tests/test_computed.py`) and of the filter (anemoi-plugins-meteoswiss) |
| computed forcings | `cos_latitude`, `sin_julian_day`, `cos_local_time`, `insolation` | the tests of their implementation upstream (earthkit-data `forcings`, earthkit-meteo `cos_solar_zenith_angle`, used by anemoi); stats-bee for the stored values |
| constant fields | `FIS`, `FR_LAND`, `SSO_*` | stats-bee's `constant_fields`; meta-bee's reference comparison |

Computed forcings are deliberately not reimplemented in the hive. A copy of the formulas would test
the copy, not the implementation the dataset was built with.

Derived variables need care when they are first generated in an accepted reference dataset. After
that, stats-bee's comparison with the reference catches changes.

Options for later:
- **Recipes in the source config:** declarative recipes such as scale, sum of fields (`ssrd = (ASWDIFD_S + ASWDIR_S) * period`) or an accumulation window (`TOT_PREC_6H` = the sum of six `TOT_PREC` values).
- **Stored references:** values computed by hand for a few fixed dates, kept as files in the dataset repo.
- **Calling the upstream forcing implementation:** earthkit as an optional dependency, instead of a copy.
- **One source for build and check:** the same source code used to build the dataset and to check it, so the mapping lives in one place.

## Example: Varda-tiny (ICH1 test dataset)

With the spec above, `n_dates: 20`, `n_points: 10000` and `include_dates: [99, 100]`:

```text
info    spot.matches_source: not comparable with source: 6 h accumulation of TOT_PREC [TOT_PREC_6H]
info    spot.matches_source: not comparable with source: computed forcing [insolation]
...
info    spot.matches_source: spot summary: 95 identical, 0 within tolerance, 0 differ, 19 not comparable, 0 unavailable (22 dates × 10000 points)
```

All 95 GRIB variables are bit-identical, and the run takes 40 s. The 19 not comparable are 9
constants, 9 forcings and `TOT_PREC_6H`.

On the faulty variants, spot-bee finds every value fault of a GRIB variable at the faulty date:
- `d-step-nan`, `-zero`, `-spike`, `-unit`, `-swap`, `-stale`, `-subtle`
- `d-step-shift`: rolled in space, so the statistics are unchanged
- `d-shift-consistent`, `m-variables-swap`, `m-varmeta-param`

It does not find:
- `d-step-constant`: FIS is a constant field, which stats-bee catches
- metadata-only faults: meta-bee
- statistics-only faults: stats-bee

`tests/test_spot_ich1.py` runs these with `uv run pytest -m faulty`. `HIVE_ICH1_ROOT`,
`HIVE_FAULTY_DIR` and `HIVE_CLEAN_DS` point it at other copies.
