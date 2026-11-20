# stats-bee

Run stats-bee on a freshly generated anemoi dataset to make sure that its stored statistics are
consistent with its data, and that the values are plausible: in physical range, no NaNs,
constant fields constant. When a dataset is regenerated, a reference stats file shows what
changed compared to the last accepted version.

It reads a seeded sample of `n_dates` dates, reduces every field to summary statistics per date
(min, max, NaN and inf counts over the full domain; mean and stdev on `n_points` grid points),
and checks those. It
needs only the dataset, not the original source data: comparing grid-point values with the
source is spot-bee's job.

| check | what | severity |
|---|---|---|
| `statistics_consistent` | stored mean/stdev/min/max are self-consistent (no NaN, min <= mean <= max, stdev 0 exactly when min == max) and match the ones recomputed from the sample | ERROR / WARNING, see [tolerances](#how-the-tolerances-work) |
| `physical_ranges` | per-date and stored min/max inside the `min`/`max` of the spec's `variables` | ERROR |
| `finite_values` | no inf on the sampled dates and points, and NaN only in the variables allowed to have them, see [NaNs](#nans) | ERROR |
| `statistics_window_valid` | `statistics_start_date` <= `statistics_end_date`, both within the dataset's dates | ERROR / WARNING if missing |
| `constant_fields` | fields in `constant_fields` (the dataset's, else the spec's variables with `constant: true`) don't change between dates (ERROR); other fields aren't identical on all dates, or uniform in space on one date (WARNING, computed forcings exempt) | ERROR / WARNING |
| `matches_reference_stats` | stored and sample statistics (the dump) match the stats section of the spec's `reference` (a ref file), on the same sample (the spec's); skipped while the reference does not exist, see [stats files](#stats-files) | ERROR / WARNING |

```bash
uv run stats-bee check <dataset.zarr> --spec <spec.yaml>            # run the checks
uv run stats-bee check <dataset.zarr> --spec <spec.yaml> --out hive/   # + reports and ref.yaml (stats section)
uv run hive compare hive/ref.yaml checks/data.ref.yaml               # compare two dumps
uv run stats-bee calibrate <dataset.zarr> --spec <spec.yaml> -o tol.yaml  # calibrated tolerances
```

`--help` on each command lists the options. This page explains the concepts behind them.

## How stats-bee reads a dataset

stats-bee reads the zarr store directly, not through anemoi-datasets, so the checks see exactly
what is on disk. It reads the `.zattrs` keys that anemoi writes:
- `variables`
- `statistics_start_date` / `statistics_end_date`: the window the stored statistics cover
- `allow_nans`
- `missing_dates`: never sampled, and left out of the population the statistics are estimated for
- `constant_fields`
- `variables_metadata[...].computed_forcing`

It also reads the arrays `data`, `dates`, `mean`, `stdev`, `minimum` and `maximum`.

### NaNs

anemoi's `allow_nans` is a bool or the list of variables that may contain NaNs (e.g. fields
undefined over part of the domain). `finite_values` allows NaNs in:
- the dataset's list, if `allow_nans` is a list
- otherwise the spec's: the variables with `allow_nans: true` in a `variables` mapping, or the
  spec's `allow_nans` list
- every variable (NaNs not checked, an INFO) only for `allow_nans: true` without a list in the spec

meta-bee's `flags_match_spec` checks that the dataset's list is the spec's. The stored statistics of
a variable with NaNs cover its finite values; stats-bee recomputes them the same way, weighting
each date by its number of finite values. A variable that is NaN everywhere has NaN statistics: an
ERROR of `statistics_consistent`.

**The sample.** The `n_dates` random dates are drawn stratified (one per equal block) over the
spec's **whole period** (`start_date` to `end_date` every `frequency`, without the spec's
`missing_dates`; the dataset's dates if the spec has no period), plus the `include_dates` (dates), so the value checks also see the dates after the
statistics window (often the validation and test period). Only the random dates inside the window
are used to estimate the statistics. Every sampled date is read whole, and then summarised:
- **minimum, maximum, NaN and inf counts over the full domain**, so `physical_ranges`,
  `finite_values` and the min/max checks see every grid point: a single bad point is found;
- **mean and stdev on `n_points` seeded grid points**, the same on every date (`all` for the full
  domain). These are the expensive part, and the estimate tolerates the sampling error.

**Cost.** Every sampled date costs one read of one chunk, which holds all variables and points of
that time step, so `n_points` does not reduce the I/O, only the time to summarise. Measured per
date on 1 km grids (about 120 variables x 1.15M points, 0.55 GB): read 0.4–0.5 s, full-domain
min/max/NaN/inf 0.07 s, mean/stdev on 100k points 0.15–0.3 s, or on the full domain 0.4–1 s.
`n_points` in the 10k–100k range is a good default.

## Stats files

stats-bee's dump, the `stats` section of `ref.yaml` (written by `--out`), is a stats file: a small fingerprint of a dataset's values on its seeded sample. Its size depends
only on the number of variables (and the list of sampled dates), not on the size of the dataset or
of the sample. It is built once per run, from the sample all checks share:
`matches_reference_stats` compares it with the reference's `stats` section, and `--out` writes it.

| key | content |
|---|---|
| `dataset`, `created`, `generator` | where it comes from |
| `sampling` | seed, `n_dates`, `n_points`, `include_dates`, grid size, statistics window, and the sampled `dates` (with `random`: the ones inside the window that estimate the statistics) |
| `stored` | the dataset's stored minimum/maximum/mean/stdev per variable |
| `sample` | per variable over all sampled dates: minimum, maximum, NaN and inf counts of the full domain; mean and stdev of the finite values on the sampled points |

Floats are written with `repr`, which round-trips exactly, and NaN and inf are written as `null`.
The current format is `varda-hive-stats/5`. Files of older formats are rejected: regenerate them
with `hive ... --out`.

### Reference workflow in a dataset repo

The stats file is the `stats` section of the dataset's ref file, written together with meta-bee's
dump by `--out` (see [ci.md](ci.md)). With stats-bee alone:

```
fresh dataset ──► stats-bee check data.zarr --spec S --out hive/   "is this dataset internally OK?"
                     │ OK (matches_reference_stats skipped: no reference yet)
                     ▼
                  cp hive/ref.yaml checks/data.ref.yaml   ──► commit in the dataset repo

later: dataset regenerated (new anemoi version, recipe change, rebuild)
                  stats-bee check new.zarr --spec S --out hive/   "what changed?" (hive/report.md)
                  intended? cp hive/ref.yaml checks/data.ref.yaml
```

Some faults are invisible from inside the dataset. For example, +2 K on T_850 with the stored
statistics updated to match leaves the dataset perfectly consistent. Only the comparison with the
accepted version shows them.

```bash
# once, after the dataset is built and checked: commit the result
uv run hive <dataset.zarr> --spec checks/<dataset>.yaml --strict --out hive/
cp hive/ref.yaml checks/<dataset>.ref.yaml       # and `reference: <dataset>.ref.yaml` in the spec

# in CI, or after rebuilding the dataset: the same command, exit 1 if anything differs
uv run hive <dataset.zarr> --spec checks/<dataset>.yaml --strict --out hive/
uv run hive compare hive/ref.yaml checks/<dataset>.ref.yaml   # the differences again, without the data
```

The sample depends on the spec only (see varda_hive/honeycomb/sampling.py), so a rebuild that meets
the spec is sampled on the reference's dates. Dates are matched by value, so a dataset with extra
time steps still lines up; sampled dates the dataset lacks are listed in `sampling.missing_dates`.
Changing the sampling or the period in the spec changes the sample: the comparison then reports
"sampled dates differ" and compares only the stored statistics, until the new `ref.yaml` is accepted. The points follow from the seed, so the grid,
`seed` and `n_points` must match the reference's (not checked for `n_points: all`). The comparison is
then cell by cell, `stored` and `sample` statistics of every variable, and needs no statistics.
A fault on one sampled date changes the sample statistics, so it is found exactly, but the finding
names the variable and statistic, not the date. Each value is:
- *identical*: bitwise equal
- *close*: within `compare_atol_stdev` x the variable's stdev (default 0, i.e. off). The tolerance
  is scaled by stdev rather than by the value, because relative tolerances break for fields near
  zero, like wind components or precipitation.
- *differs*: anything else

`hive compare` runs the same comparison as the `matches_reference_stats` check on two stored dumps. See [ci.md](ci.md) for the whole publishing workflow
with all bees.

## How the tolerances work

`statistics_consistent` compares the stored statistics with statistics recomputed from a
sample, so the tolerance has to cover the sampling error. Two parts contribute:
- **dates**: the dates are the sampling unit, because grid points within one time step are
  strongly correlated. They are drawn stratified over the time range; the ones inside the
  statistics window estimate the statistics.
- **points**: the point subset is the same on every date, so its error does not average out over
  the dates. It is a fixed bias of the sample, zero for `n_points: all`.

Default tolerances:
- **ERROR**: the mean is off by more than `z_threshold` x a distribution-free bound on its
  sampling error: stdev x (sqrt(1 - n/N) / sqrt(n) + sqrt(1 - k/G) / sqrt(k)), for n of N dates
  and k of G points. The per-date means can't spread more than the variable itself, so this bound
  always holds, and the check is very unlikely to fire by chance. It is loose, because stdev
  includes the spatial variance, while the error from sampling dates depends only on how much the
  domain mean varies in time.
- **WARNING**: the mean or stdev is off by more than `z_threshold` x (the standard error estimated
  from the sample itself + the point term). The estimate is analytic for the mean and a bootstrap
  over the dates for the stdev. It is tighter, but a skewed sample can fool it. On Varda-tiny, 16
  of 20 random dates fell at night, and the mean of insolation looked inconsistent.
- **Minimum and maximum** can only be bounded by a sample: a sampled value below the stored
  minimum or above the stored maximum is an ERROR. Exact checks without a tolerance also apply:
  min <= mean <= max, and stdev is 0 exactly when min == max.
- **Full pass**: with `n_dates: all` and `n_points: all` the recomputation is exact, and only
  floating-point error remains: `statistics_rtol` x (|value| + stdev). That is affordable for small
  datasets like Varda-tiny (about 1 min), not for multi-year ones.

The INFO finding of the check lists the smallest detectable difference of each mean, both as an
ERROR and as a WARNING. That shows when a fault is below what the sample can resolve. In that
case a reference stats file still catches it, because that comparison is exact.

### Calibrated tolerances

Instead of the distribution-free bound, the tolerance can be measured:

```bash
uv run stats-bee calibrate data.zarr --spec checks/<dataset>.yaml --n-calib 300 \
    -o checks/<dataset>.tolerances.yaml
# then, in the spec:  calibrated_tolerances: <dataset>.tolerances.yaml
```

1. Read `--n-calib` stratified dates inside the statistics window. For each, keep the mean and
   stdev of every variable over the spec's point subset and over the full domain (in memory).
2. **Temporal error**: simulate `--n-sim` (2000) stratified samples of the size the check uses, and
   take the largest deviation of their mean and stdev from those of all calibration dates.
3. **Point bias**: the subset estimate minus the full-domain estimate, measured directly for the
   spec's seed.

The tolerance is |point bias| + `--margin` (2) x temporal error, per variable, for mean and stdev.
The check then reports an ERROR beyond it and has no WARNING tier. Skewed variables like
insolation get the tolerance they need, because the calibration samples their real distribution.
The stored statistics are not used, so the dataset can calibrate itself, or the previous version
of a dataset calibrates its rebuilds. The tolerance file has a fixed size. It is valid for one
sampling only: if `seed`, `n_dates`, `n_points`, the grid or the number of estimation dates
change, the check reports an ERROR asking to recalibrate, and falls back to the default
tolerances. Calibration reads `--n-calib` dates over the full domain: about 5 min for 300 dates of
ICH1 1 km.

## Limits

- **Single-step faults** are only seen on sampled dates: a random sample hits one given step out
  of N with probability n/N. Put the dates you care about in `include_dates`, or use
  `n_dates: all` when a full pass is affordable.
- **Localized faults** (a single NaN point, a spike on a few points) are found on every sampled
  date, because min/max and the NaN/inf counts cover the full domain. A subtle local change that
  stays inside the physical range only shows in mean/stdev, which see k of G points.
- **A field shifted or rolled in space** keeps its mean/stdev/min/max over the full domain. With a
  point subset it changes the values on the sampled points, so the reference comparison finds it
  (Varda-tiny `d-step-shift`), but not with `n_points: all`. Comparing with the source remains
  spot-bee's job.

## Testing on Varda-tiny

`uv run pytest -m faulty` runs stats-bee on the clean Varda-tiny dataset (expect no ERROR) and
on each of its faulty `d-*` variants, and checks that the expected checks fire. See
`tests/test_faulty_tiny.py` for the mapping from scenario to check.
