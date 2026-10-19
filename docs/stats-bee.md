# stats-bee

stats-bee checks the values of an anemoi dataset on a seeded sample of dates:

| check | what | severity |
|---|---|---|
| `statistics_consistent` | stored mean/stdev/min/max vs the ones recomputed from the sampled dates | ERROR / WARNING, see [tolerances](#how-the-tolerances-work) |
| `physical_ranges` | per-date and stored min/max inside the ranges of `ranges_file` | ERROR |
| `no_nans` | no NaNs when the dataset says `allow_nans: false` | ERROR |
| `constant_fields` | fields in `constant_fields` don't change between dates (ERROR); other fields aren't identical on all dates or uniform in space (WARNING) | ERROR / WARNING |
| `matches_reference_stats` | the sampled dates match `reference_stats`, see [stats files](#stats-files) | ERROR / WARNING |

```bash
uv run stats-bee <dataset.zarr> --spec <spec.yaml>                  # run the checks
uv run stats-bee dump <dataset.zarr> --spec <spec.yaml> -o out.yaml  # write a stats file
uv run stats-bee compare out.yaml reference.yaml                     # compare two stats files
```

`--help` on each command lists the options. This page explains the concepts behind them.

## How stats-bee reads a dataset

stats-bee reads the zarr store directly, not through anemoi-datasets, so the checks see exactly
what is on disk. It reads the `.zattrs` keys that anemoi writes:
- `variables`
- `statistics_start_date` / `statistics_end_date`: the window the stored statistics cover
- `allow_nans`
- `constant_fields`
- `variables_metadata[...].computed_forcing`

It also reads the arrays `data`, `dates`, `mean`, `stdev`, `minimum` and `maximum`. Every sampled
date costs one read of one chunk, which holds all variables of that time step.

## Stats files

A stats file is a small fingerprint of a dataset's values on a seeded sample of dates. Its
format follows the extension: `.yaml`/`.yml` gives YAML (recommended, the same format as the
spec), anything else gives JSON.

| key | content |
|---|---|
| `dataset`, `created`, `generator` | where it comes from |
| `sampling` | seed, `n_dates`, `include_dates`, statistics window, dates of `--dates-from` missing in the dataset |
| `dates`, `time_index`, `random` | the sampled dates; `random` marks the ones used to estimate the statistics |
| `stored` | the dataset's stored mean/stdev/minimum/maximum per variable |
| `per_date` | mean/stdev/minimum/maximum/n_nan per variable, one list entry per date |
| `recomputed` | statistics estimated from the random dates, with their standard errors `se_mean`/`se_stdev` |

Floats are written with `repr`, which round-trips exactly, and NaN is written as `null`.

### Reference workflow in a dataset repo

```bash
# once, after the dataset is built and checked: commit the result
uv run stats-bee dump <dataset.zarr> --spec checks/<dataset>.yaml -o checks/<dataset>.stats.yaml

# in CI, or after rebuilding the dataset
uv run stats-bee dump <dataset.zarr> --spec checks/<dataset>.yaml \
    --dates-from checks/<dataset>.stats.yaml -o new.stats.yaml
uv run stats-bee compare new.stats.yaml checks/<dataset>.stats.yaml   # exit 1 if anything differs
```

`--dates-from` samples exactly the reference's dates. Dates are matched by value, so a dataset
with extra or missing time steps still lines up. The comparison is then cell by cell and needs
no statistics. Each value is:
- *identical*: bitwise equal
- *close*: within `compare_atol_stdev` x the variable's stdev (default 0, i.e. off). The tolerance
  is scaled by stdev rather than by the value, because relative tolerances break for fields near
  zero, like wind components or precipitation.
- *differs*: anything else

With `reference_stats` in the spec, `stats-bee` (and `hive`) run the same comparison as the
`matches_reference_stats` check, on the reference's dates.

## How the tolerances work

`statistics_consistent` compares the stored statistics with statistics recomputed from a
sample, so the tolerance has to cover the sampling error. The dates are the sampling unit,
because grid points within one time step are strongly correlated. They are drawn stratified
(one per equal block) over the statistics window.

- **ERROR**: the mean is off by more than `z_threshold` x a distribution-free bound on its
  sampling error: stdev x sqrt(1 - n/N) / sqrt(n). The per-date means can't spread more than the
  variable itself, so this bound always holds, and the check is very unlikely to fire by chance.
- **WARNING**: the mean or stdev is off by more than `z_threshold` x the standard error estimated
  from the sample itself. That estimate is analytic for the mean and a bootstrap over the dates
  for the stdev. It is much tighter, but a skewed sample can fool it. On Varda-tiny, 16 of 20
  random dates fell at night, and the mean of insolation looked inconsistent.
- **Minimum and maximum** can only be bounded by a sample: a sampled value below the stored
  minimum or above the stored maximum is an ERROR. Exact checks without a tolerance also apply:
  min <= mean <= max, and stdev is 0 exactly when min == max.
- **Full pass**: with `n_dates: all` the recomputation is exact, and only floating-point error
  remains: `statistics_rtol` x (|value| + stdev). On Varda-tiny this takes about 40 s and turns a
  0.1 % change of the T_2M mean into an ERROR.

The INFO finding of the check lists the smallest detectable difference of each mean, both as an
ERROR and as a WARNING. That shows when a fault is below what the sample can resolve. In that
case a reference stats file still catches it, because that comparison is exact.

## Limits

- **Single-step faults** are only seen on sampled dates: a random sample hits one given step out
  of N with probability n/N. Put the dates you care about in `include_dates`, or use
  `n_dates: all` when a full pass is affordable.
- **A field shifted or rolled in space** keeps its mean/stdev/min/max: that is spot-bee territory.
- Dates listed in the dataset's `missing_dates` are not excluded from the sample yet.

## Testing on Varda-tiny

`uv run pytest -m faulty` runs stats-bee on the clean Varda-tiny dataset (expect no ERROR) and
on each of its faulty `d-*` variants, and checks that the expected checks fire. See
`tests/test_faulty_tiny.py` for the mapping from scenario to check.
