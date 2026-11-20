# meta-bee

meta-bee checks the metadata and structure of an anemoi dataset. It reads the group attributes
(`.zattrs` in zarr format 2, the root `zarr.json` in format 3) and the small arrays (dates,
coordinates, statistics), never `data`, so it takes about a second even on large datasets. It does
not need anemoi-datasets.

```bash
uv run meta-bee check data.zarr --spec checks/data.yaml           # all meta checks
uv run meta-bee check data.zarr --spec checks/data.yaml --out hive/   # + reports and ref.yaml (meta section)
uv run hive compare hive/ref.yaml checks/data.ref.yaml             # what is identical, what differs
uv run hive compare hive/ref.yaml checks/data.ref.yaml --all       # every item, identical ones too
```

## Checks

| check | needs | catches |
|---|---|---|
| `dates_valid` | nothing | unset (NaT), duplicate, unordered or irregularly spaced dates |
| `frequency_consistent` | nothing | `frequency` attribute not equal to the step between the dates |
| `date_attrs_consistent` | nothing | `start_date`/`end_date` attributes not the first/last date |
| `missing_dates_valid` | nothing | `missing_dates` entries that are not dates of the dataset (and lists the others as INFO) |
| `shape_consistent` | nothing | `shape`, `variables`, `field_shape`, dates, coordinates or statistics not matching `data` |
| `coordinates_valid` | nothing | non-finite or out-of-range latitudes/longitudes, coordinates in radians |
| `variables_metadata_consistent` | nothing | `variables_metadata`/`constant_fields` naming unknown variables; `constant_fields` disagreeing with `constant_in_time` |
| `variables_match_spec` | `variables` | variables missing or not expected |
| `dates_match_spec` | `start_date`, `end_date`, `frequency` | other date range or step |
| `grid_matches_spec` | `grid_points`, `resolution`, `bbox` | other grid size, resolution or extent (shifted or swapped coordinates) |
| `flags_match_spec` | `allow_nans` (or a `variables` mapping), `missing_dates` | NaNs allowed in other variables than the spec's, dates unexpectedly listed as missing |
| `variables_metadata_match_spec` | per-variable keys and `units` in `variables` | wrong mars param/levtype/levelist; units differing from the spec (WARNING when the dataset stores none) |
| `matches_reference` | `reference` or `--reference` | any difference to the accepted version, see below |

The first group replaces `anemoi-datasets validate`, which catches 3 of the 13 metadata faults of
Varda-tiny. meta-bee reports all of them: 12 as ERROR, and `m-stats-dates` as a WARNING against the
reference (its ERROR is stats-bee's `statistics_window_valid`). See `tests/test_faulty_tiny.py`.

## Spec fields

```yaml
variables:
  T_2M: {param: T_2M, levtype: sfc, units: K}
  T_850: {param: T, levtype: pl, levelist: 850}
  CIN_MU: {param: CIN_MU, levtype: sfc, allow_nans: true}
start_date: 2020-03-01T00:00:00
end_date: 2020-03-30T23:00:00
frequency: 1h
grid_points: 30676
resolution: 2km
bbox: {lat_min: 45.43, lat_max: 48.19, lon_min: 5.25, lon_max: 11.03, tolerance: 0.01}
missing_dates: []
reference: data.ref.yaml                # a ref file (`ref.yaml` of `--out`) or a zarr; skipped while missing
reference_policy: {ignore: [], error: [], warning: [], info: []}
```

Write them from the anemoi **recipe**, not from the dataset you check: a spec copied from the
dataset agrees with it by construction.

Each key of a variable other than `min`, `max`, `units`, `allow_nans` and `constant` is expected in
its `variables_metadata` entry, looked up at the top level and then in `mars`; `units` is compared
too. A variable without such keys is not checked.

`allow_nans` is the set of variables with `allow_nans: true` (with a `variables` mapping) or the
spec's `allow_nans`: `true`, `false` or a list. anemoi writes a list of variables or a bool; a list
is compared as a set (one ERROR per variable allowed by only one side), `false` is the empty list,
and a list against `true` is an ERROR.

## The reference: the meta section and `compare`

meta-bee's dump, the `meta` section of `ref.yaml` (written by `--out`, about 30 kB), holds all group
attributes and, for every array, its shape, dtype, a sha256 of its values and a short summary. `data`
gets only its shape and dtype. It is built once per run: `matches_reference` compares it with the
reference's `meta` section, and `--out` writes it. Commit `ref.yaml` next to the spec once a version
is accepted (with stats-bee's dump in its `stats` section); a rebuild is then compared with it
without needing access to the published dataset. The dump does
not depend on the zarr format (2 or 3) and does not record it: the same content gives the same
`meta` section, so a dataset rewritten in format 3 compares clean against a reference taken from
format 2. A zarr can be given as `--reference` instead: it is
fingerprinted on the fly. A spec `reference` that does not exist yet makes `matches_reference` skip
("reference … not found"): that run's `ref.yaml` is the first reference.

The comparison walks both on dotted paths (`variables_metadata.T_850.mars.levelist`,
`arrays.latitudes.values`). Every item is `identical`, `differs`, `added` (only in the dataset),
`missing` (only in the reference) or `ignored`. The severity of a difference:

| severity | paths (defaults) |
|---|---|
| ERROR | structure: `variables` (set and order), `shape`, `dtype`, `field_shape`, `resolution`, `frequency`, `start_date`, `end_date`, `last_date`, `arrays.*.shape`, `arrays.*.dtype`, the values of `dates`, `latitudes`, `longitudes` |
| WARNING | everything else: `variables_metadata.*`, `constant_fields`, `missing_dates`, `allow_nans`, statistics dates, added or missing attributes and arrays |
| INFO | `description`, `attribution`, `licence`; the values of the statistics arrays, as rebuilds are not bit-reproducible (stats-bee compares them with tolerances) |
| ignored | churn: `uuid`, `latest_write_timestamp`, `provenance_load`, `total_size`, `total_number_of_files`, `version`, `history`, `recipe`, `order_by`, `chunks`, `data_request` |

The spec's `reference_policy` comes first: globs in `ignore`, `error`, `warning` and `info`, checked
in this order. For example, to accept a new anemoi version that adds keys to every variable:

```yaml
reference_policy:
  ignore: ["variables_metadata.*.grib"]
  error: [constant_fields]
```

`hive compare a.ref.yaml b.ref.yaml` runs the same comparison on two dumps and prints the differing
items as a table (`--all` adds the identical and ignored ones, which also shows the paths to use in
the policy). It exits 1 when a difference reaches `--fail-on` (default `error`), and with `--strict`
also when a section is in only one of the files. `--format json` and `--format markdown` are
available, as for every bee.

## In CI

With `reference` in the spec, `hive --strict` (or `meta-bee check`) runs the reference comparison
as the `matches_reference` check, so a release needs no separate `compare` step. When a change is
intended (a new variable, a new period), review the differences in `hive/report.md` of
`hive ... --out hive/`, then `cp hive/ref.yaml checks/data.ref.yaml` and commit it with the spec change. See [ci.md](ci.md) for the whole
workflow and an example CSCS-CI job.
