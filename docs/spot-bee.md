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
| ERROR | values of a variable differ from the source: one finding per variable, at its first differing date, with the totals `n_differ`, `n_nan_mismatch`, `max_abs_diff` (`"inf"` if one side is infinite), the first differing grid points, for ensembles the members (dataset member positions 0, 1, ..., not the source's numbering, e.g. GRIB `number`), and in `dates` each differing date with its `n_differ` and `max_abs_diff` |
| ERROR | the source failed for a variable (e.g. two GRIB messages match, wrong grid size or coordinates, a bad override): once per variable, which is then not compared further; the summary lists the dates skipped after it |
| ERROR | the source has no data for a variable on any sampled date: usually a wrong mapping, `root` or date format. If the source really lacks it, add it to `GribIndexSource`'s `skip`, or, for any plug-in, to `ignore_warnings: {matches_source: [VAR]}`, which keeps it a WARNING demoted to INFO |
| WARNING | the source has no data for a variable on some sampled dates, with the dates |
| INFO | a variable is not comparable with the source, with the reason (see [derived variables](#derived-variables-and-forcings)) |
| INFO, skipped | no sampled value was compared at all (every variable not comparable, unavailable or failed): fails under `--strict` |
| INFO | `spot summary: 95 identical, 0 within tolerance, 0 differ, 19 not comparable, 0 unavailable (22 dates × 10000 points)` |

The `details` of the summary list the variables of each group, with the largest absolute difference
of each and, for differing variables, the dates. Use `--format json` or `--report` to get them.

Values count as **identical** when they are equal in the dataset's precision: the source values are
cast to the dataset dtype (float32) first, so a GRIB value decoded as float64 that was stored
unchanged is identical. **Within tolerance** means `|dataset - source| <= source_atol + source_rtol *
|source|` everywhere, and NaN in both counts as equal. The default `source_atol` is 0: an absolute
default would hide faults in fields with small values (kg/kg, m). Give a variable its own `atol` when
its source values near zero are rescaled or shifted: a precipitation in m from mm, or a temperature
in °C from K (`offset: -273.15`), where float32 rounding of the offset is about 1e-5 at values near 0,
so give it e.g. `atol: 1.0e-4`.

A rescaled variable (`scale`, `offset`) is often "within tolerance" rather than "identical": the
source applies the factor in float64, the dataset build may have applied it in float32, and the two
can differ in the last bit. That is a label, not a fault.

## Sampling

spot-bee uses the spec's `sampling`, the same as stats-bee:
- **Dates:** the stratified random dates plus `include_dates`, without the missing dates.
- **Points:** `n_points` grid points, the same on every date, or `all`.

Both bees therefore check the same cells (one definition: `honeycomb/sampling.py:sampled_dates`).
With a `reference` in the spec, stats-bee samples the reference's dates instead, while spot-bee
keeps the spec's. They are the same dates as long as the reference was dumped with the same
`sampling` from a dataset of the same length; spot-bee does not read stats files, to stay
independent of stats-bee. Reading the source is usually the cost: one GRIB message
per (variable, date), whatever `n_points` is. `n_points: all` costs little more than a subset and
also finds faults on a few grid points, such as spikes.

## Spec fields

```yaml
sampling: {seed: 0, n_dates: 20, n_points: all, include_dates: []}   # defaults: n_dates 5, n_points 1000
tolerances:
  source_rtol: 1.0e-4                  # default
  source_atol: 0.0                     # default
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
compared. The field may be a masked array (masked values count as NaN); anything that is not
numeric fails that variable. An exception from the constructor or from `bind` stops the whole check
("check crashed"), so a plug-in should keep per-variable configuration errors for `fetch`, as
`GribIndexSource` does.

Three plug-ins ship with the hive:

- `varda_hive.sources.grib_index:GribIndexSource`: GRIB files through anemoi `grib-index`
  databases, see below.
- `varda_hive.sources.fdb:FdbSource`: GRIB fields from an FDB, see [FdbSource](#fdbsource).
- `templates/source_plugin_example.py:ZarrSource`: another anemoi zarr, e.g. a trusted copy.

A dataset with another kind of source (polytope, a different archive) can bring its own plug-in
in its repo. Both GRIB sources decode and validate messages the same way
(`src/varda_hive/sources/_grib.py`): the number of grid points, the grid check below, and missing
points from the bitmap as NaN.

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
    root: /scratch/mch/huppd/varda/ich1-analysis-anemoi-dataset   # base of the DB and GRIB paths; relative to the spec, default: the spec's directory
    indexes:                                   # name -> DB, or a list of DBs (e.g. per-day indexes)
      ana: ml_test/processed/grib-index-ana.db
      fg: ml_test/processed/grib-index-fg.db
    default_index: ana                         # default: the first one
    variables:                                 # overrides of the default mapping
      W_SO: {index: fg, match: {shortName: W_SO, "level:d": "0.01"}}
      TOT_PREC_1H: {index: fg, match: {shortName: TOT_PREC}, scale: 0.001}   # renamed and rescaled in the recipe
    skip:                                      # not comparable, with the reason
      TOT_PREC_6H: 6 h accumulation of TOT_PREC
    # pl_type_of_level: isobaricInhPa
    # members: [1, 2, 3]                       # GRIB `number` of each dataset member, default 0, 1, ...
    # check_grid: true                         # compare GRIB coordinates with the dataset's, where eccodes has them
```

**Mapping.** Without an override, a variable's messages come from its
`variables_metadata[var].mars` entry:

| mars entry | matched keys |
|---|---|
| `levtype: pl` | `shortName = param`, `level = levelist`, `typeOfLevel = pl_type_of_level` (an ERROR without `levelist`) |
| any other | `shortName = param`, and `level = levelist` if `levelist` is set and the index has a `level` column |

The `level` match of non-`pl` variables assumes the index writes `level` like mars writes `levelist`.
Where it does not (soil depths, scaled levels), the variable is "not available on any sampled date";
give it an override with the right `match`, as `W_SO` above.

An override gives:

| key | meaning | default |
|---|---|---|
| `index` | which index to search | `default_index` |
| `match` | the index columns to match | |
| `scale`, `offset` | applied as `value * scale + offset` | `1`, `0` |

The `match` keys must be columns of the index. A configuration error of one variable (unknown key or
index, a column the index lacks, an empty `match`, a `pl` variable without `levelist`) gives one ERROR
for that variable, and the other variables are still compared. Errors of the source as a whole (a DB
that does not exist, an index without DBs, `members` of the wrong length) stop the check. Index values
are text, so write numbers the way the index stores them (`"0.01"`, `850`).

**Not comparable without an override:**
- variables in `skip`
- computed forcings (`computed_forcing`)
- derived fields (`process`, e.g. accumulations)
- constant fields (`constant_in_time`): they are read from a separate file, not from the
  time-indexed archive
- variables without mars metadata

**Errors** rather than wrong comparisons:
- more than one message matching a (variable, date)
- a message whose `numberOfDataPoints` differs from the dataset grid

**Ensembles:** with several ensemble members, `number` is matched as well: 0, 1, ... by default, or
the numbers listed in `members` (e.g. `[1, 2, 3]` for a dataset without the control member). With
`members`, `number` is matched for a single-member dataset too, e.g. `members: [5]` for a dataset
built from one member of an ensemble archive. Every index a variable uses then needs a `number`
column.

GRIB values are returned in the order of the message. That is the dataset grid order for an
unstructured ICON grid read with anemoi's `grid_definition: icon`, which keeps the GRIB order. A
recipe that crops, thins or reorders the grid needs its own plug-in.

**Grid check:** for grids whose coordinates eccodes computes (`regular_ll`, `reduced_ll`,
`regular_gg`, `reduced_gg`), the latitudes and longitudes of each distinct grid (`md5GridSection`) are
compared once with the dataset's, within 0.001°. A mismatch, e.g. a cropped or reordered recipe, is an
ERROR of the variable that names the first deviating point, instead of "differ" everywhere.
Unstructured grids (ICON: no coordinates in the GRIB) and rotated grids are not checked.
`check_grid: false` turns the check off.

**eccodes** is imported when the source is bound: without it the check fails once, with the install
hint.

### FdbSource

`FdbSource` retrieves one GRIB field per (variable, date) from an FDB with pyfdb and decodes it with
eccodes, so it needs the `fdb` extra and the FDB library, e.g. from the FDB uenv. Its args follow
the anemoi `fdb` source of the recipe, so they can be copied from it:

```yaml
source:
  plugin: varda_hive.sources.fdb:FdbSource
  args:
    fdb_config: null             # dict as in the recipe; null: pyfdb's default (FDB5_CONFIG_FILE)
    fdb_userconfig: null
    request: {class: rd, stream: reanl, expver: r001, model: icon-rea-l-ch1, type: cf}
    offset_from_date: true       # date=YYYYMMDD, time=0000, step=offset from midnight
    step_zero_from_previous_date: [VMAX_10M]   # step 0 -> previous date, step 24 (true: all variables)
    variables:                   # overrides, merged on top of the default request
      VMAX_10M: {}               # `process: maximum`, but the raw GRIB field: compare it
      CLCL: {request: {levtype: pl, levelist: 800}}   # requested as pl, mapped to sfc by a flavour
      CLCM: {request: {levtype: pl, levelist: 400}}
      CLCH: {request: {levtype: pl, levelist: 0}}
    skip: {CIN_MU: masked below -999 by apply_mask_fields}
    members: [1, 2]              # GRIB `number` of each dataset member, default 0, 1, ... (none for one member)
    check_grid: true
```

The request of a variable is `request`, plus from its `variables_metadata`: `param` = `grib.paramId`
(FDB indexes paramIds, as the recipe's `param_id_map`), else `mars.param`; `levtype` = `mars.levtype`;
`levelist` = `mars.levelist` unless null. Then the time keys, as the anemoi source computes them:
with `offset_from_date`, `date` of the valid time, `time=0000` and the step since midnight, spelled
as FDB indexes it (`0`, `10m`, `6`, `23h30m`); without it, `time=HHMM`. An override's `request` is
merged on top, and its `scale` and `offset` apply to the values (value * scale + offset).

Not comparable, unless the variable has an override: `skip`, computed forcings, derived fields
(`process`), constant fields, no mars metadata (the same rules as GribIndexSource), and
`levtype: unknown`, e.g. a field interpolated from model to pressure levels. An override, even an
empty one, makes the variable comparable, e.g. a `process: maximum` that is the raw GRIB field.

No data for a request is `None` (reported as missing); several messages for one request are an ERROR
of the variable. Values are in message order, which is the dataset order for an ICON grid read
with `grid_definition: icon`; the grid check is that of GribIndexSource.

**Environment.** Run it as the build ran, e.g. in the FDB uenv with the dataset repo's eccodes
definitions, and `FDB5_CONFIG_FILE` pointing at the FDB root the dataset was built from:

```bash
uenv run --view fdb:rea-l-ch1 fdb/5.21:v2 -- bash -c '
  export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
  uv run --with "varda-dataset-checker-hive[fdb] @ git+https://github.com/MeteoSwiss/varda-dataset-checker-hive@<ref>" \
    hive data.zarr --spec checks/<dataset>.yaml --strict'
```

In that uenv, importing pyfdb before eccodes crashes the interpreter; `FdbSource` imports eccodes
first. `tests/test_fdb_realch1.py` (`pytest -m fdb`) runs spot-bee on the REA-L-CH1 10 min CI
dataset against the FDB.

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
- **Batched or point requests:** an optional `prefetch(date, variables)` or a `points` argument in the
  source contract, for FDB/polytope sources where one request per field is slow (`FdbSource` makes
  one retrieve per variable and date).
- **A small dataset view for `bind`:** variables, metadata, dates, members, grid size and lat/lon,
  instead of the internal `Dataset`.
- **Source errors at spec load:** build the plug-in when the spec is loaded, so a wrong plug-in or
  path exits with 2 (configuration) instead of 1.
- **`spot-bee mapping`:** print variable → index, match and reason without a full run.
- **Pruning per-day DBs by date** in `GribIndexSource`, once long lists of DBs make it slow.

## Limits

- **Sampled dates only.** A fault on one time step is found only if that step is sampled: with
  `n_dates` of N steps, the chance is about n/N. Put known-risky steps in `include_dates`, or use
  `n_dates: all` for a release.
- **Not compared with the source:** constant fields (stats-bee's `constant_fields` only checks that
  they don't change in time, not that they are the right field), derived fields and computed
  forcings, and the coordinates (`latitudes`, `longitudes`).
- **Grid order is trusted on unstructured and rotated grids.** There only the number of grid points is
  checked; a source in another order shows up as "differ" everywhere, not as a mapping error.
- **The mapping comes from the dataset's own `variables_metadata`.** A fault that changes both the
  metadata and the data the same way is not seen; overrides in the spec avoid this for the
  variables they list.
- **A failure stops a variable.** After the first source failure a variable is not compared on
  later dates (listed in the summary's `not_compared_after_failure`).

## Example: Varda-tiny (ICH1 test dataset)

With the spec above (and `source_per_variable: {TOT_PREC_1H: {atol: 1.0e-6}}`, as in
`tests/test_spot_ich1.py`), `n_dates: 20`, `n_points: 10000` and `include_dates: [99, 100]`:

```text
info    spot.matches_source: not comparable with source: 6 h accumulation of TOT_PREC [TOT_PREC_6H]
info    spot.matches_source: not comparable with source: computed forcing [insolation]
...
info    spot.matches_source: spot summary: 95 identical, 0 within tolerance, 0 differ, 19 not comparable, 0 unavailable (22 dates × 10000 points)
```

All 95 GRIB variables are bit-identical, and the run takes about 10 s. The 19 not comparable are 9
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
