# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The sample stats-bee checks, and stats files: a small, fixed-size YAML or JSON fingerprint of a
dataset (the format follows the file extension), stats-bee's dump: the `stats` section of
`ref.yaml` (`--out`), compared by the `matches_reference_stats` check and `hive compare`.

The sample is `n_dates` seeded dates, stratified over the spec's period (plus `include_dates`), see
varda_hive/honeycomb/sampling.py: the same spec samples the same dates on every rebuild.
Every sampled date is read whole (one chunk), so per date and variable:
- minimum, maximum and the NaN and inf counts cover the **full domain**: cheap, and a single bad
  point is never missed;
- mean and stdev of the finite values cover `n_points` seeded grid points, the same on every date
  (the costly part).
A variable may have `flag_values` in the spec (a fill value outside the physical range, matched
exactly in the stored dtype). minimum, maximum, mean and stdev still include the flags (they are
compared with the stored statistics); `valid_minimum`/`valid_maximum` (min/max of the finite
non-flag values) and `n_flag` (number of flags) are the flag-aware per-date summaries, and the
stats file gets a `flag_share` for flagged variables only.
These per-date summaries live in memory only (`Sample`): the checks use them, the stats file does not.

A stats file holds, per variable:
- `stored`: the statistics stored in the dataset (minimum, maximum, mean, stdev)
- `sample`: over all sampled dates, minimum, maximum, NaN and inf counts of the full domain, and
  mean and stdev of the finite values on the sampled points; for a variable with `flag_values`
  also `flag_share`, the share of flag values over the sampled dates (full domain)

Its size depends only on the number of variables (and the list of sampled dates in `sampling`),
not on the size of the dataset. Two stats files of the same sample (same spec sampling and
period) compare cell by cell, without any statistics: a value is identical, close
(within `compare_atol_stdev` * stdev) or differs. Floats are written with `repr`, which
round-trips exactly, and NaN/inf as null. Missing dates are never sampled.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings

import numpy as np

from ... import __version__
from ...honeycomb import Dataset, Finding, Spec, error, info, warning
from ...honeycomb.dataset import STATISTICS, parse_date
from ...honeycomb.files import read_file, save_file
from ...honeycomb.sampling import sample_points, sampled_dates

FORMAT = "varda-hive-stats/5"
# Per-date summary of one variable: mean/stdev (and n_finite, their weight) on the sampled points,
# minimum/maximum and the NaN/inf counts on the full domain.
SUMMARY = ("mean", "stdev", "minimum", "maximum", "n_nan", "n_inf", "n_finite", "valid_minimum", "valid_maximum",
           "n_flag")
COUNTS = ("n_nan", "n_inf", "n_finite", "n_flag")
SAMPLE_STATS = ("minimum", "maximum", "mean", "stdev", "n_nan", "n_inf")
N_BOOTSTRAP = 200


def flag_mask(values: np.ndarray, flags: Sequence[float]) -> np.ndarray:
    """True where `values` equal a flag value, exactly in the stored dtype (a spec value -999.9
    matches the float32 -999.9000244)."""
    return np.isin(values, np.asarray(flags, dtype=values.dtype))


def statistics_window(ds: Dataset) -> tuple[int, int]:
    """Time indices [lo, hi) the stored statistics were computed over."""
    lo, hi = 0, len(ds.dates)
    if start := ds.attrs.get("statistics_start_date"):
        lo = int(np.searchsorted(ds.dates, parse_date(start), side="left"))
    if end := ds.attrs.get("statistics_end_date"):
        hi = int(np.searchsorted(ds.dates, parse_date(end), side="right"))
    return lo, hi


# --- the sample ------------------------------------------------------------------------------


@dataclass
class Sample:  # pylint: disable=too-many-instance-attributes
    """Per-date summaries of the sampled dates x points, in memory."""
    variables: list[str]
    dates: list[str]
    time_index: list[int]
    # True for the random dates inside the statistics window: only those estimate the statistics,
    # `include_dates` would bias the estimate.
    random: np.ndarray
    summary: np.ndarray  # (date, variable, SUMMARY), float64
    points_per_date: int  # values per variable and date for mean/stdev (sampled points x ensemble members)
    values_per_date: int  # the same for the full domain
    n_window: int  # dates in the statistics window, without the missing dates
    missing: list[str]  # sampled dates (the spec's) that are not in the dataset or missing in it
    stored: dict[str, dict[str, Any]]  # None for NaN
    seed: int
    flags: dict[str, list[float]] = field(default_factory=dict)  # flag values of the variables that have some

    @property
    def full_domain(self) -> bool:
        return self.points_per_date == self.values_per_date

    def per_date(self, name: str) -> dict[str, list]:
        """{summary: [value per date]} of one variable, None for NaN."""
        i = self.variables.index(name)
        return {
            s: [_num(x, int if s in COUNTS else float) for x in self.summary[:, i, j]]
            for j, s in enumerate(SUMMARY)
        }

    def aggregate(self) -> dict[str, dict[str, Any]]:  # pylint: disable=too-many-locals
        """Over all sampled dates: minimum, maximum and NaN/inf counts of the full domain, and mean and
        stdev of the finite values on the sampled points. Exact from the per-date summaries: the mean
        and the pooled variance are weighted by the number of finite sampled values of each date."""
        m, s, lo, hi, n_nan, n_inf, weight, n_flag = (self.summary[:, :, SUMMARY.index(c)]
                                                      for c in ("mean", "stdev", "minimum", "maximum", "n_nan", "n_inf",
                                                                "n_finite", "n_flag"))
        total = weight.sum(axis=0)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN variables give NaN
            mean = np.nansum(weight * m, axis=0) / total
            var = np.nansum(weight * (s**2 + (m - mean)**2), axis=0) / total
            lo, hi = np.nanmin(lo, axis=0), np.nanmax(hi, axis=0)
        n_values = len(self.dates) * self.values_per_date
        out = {
            name: {
                "minimum": _num(lo[i]),
                "maximum": _num(hi[i]),
                "mean": _num(mean[i]),
                "stdev": _num(math.sqrt(var[i]) if var[i] >= 0 else math.nan),
                "n_nan": int(n_nan[:, i].sum()),
                "n_inf": int(n_inf[:, i].sum()),
            }
            for i, name in enumerate(self.variables)
        }
        for i, name in enumerate(self.variables):
            if name in self.flags:
                out[name]["flag_share"] = _num(n_flag[:, i].sum() / n_values) if n_values else None
        return out

    def estimate(self) -> dict[str, dict[str, Any]]:
        """The window statistics estimated from the random dates inside it, see `estimate`, with the
        `spatial` term of the point subset."""
        out = estimate(self.summary[self.random], self.n_window, self.variables, self.seed, self.full_domain)
        return {name: rec | {"spatial": self.spatial_fraction()} for name, rec in out.items()}

    def spatial_fraction(self) -> float:
        """sqrt(1 - k/G) / sqrt(k) for k sampled of G values per date: times stdev, it bounds the error
        of a mean over the fixed point subset (0 for the full domain)."""
        k, g = self.points_per_date, self.values_per_date
        return math.sqrt(max(0.0, 1 - k / g)) / math.sqrt(k) if k else 0.0


def sample_indices(ds: Dataset, spec: Spec) -> tuple[dict[int, bool], list[str]]:
    """{time index: is random} of the spec's sample, and the sampled dates the dataset lacks.

    The random dates are stratified over the whole period, so the value checks also cover the dates
    after the statistics window; `is random` is True only for those inside the window.
    """
    window = statistics_window(ds)
    picked, missing = sampled_dates(spec, ds.dates, ds.missing_indices)
    return {t: r and window[0] <= t < window[1] for t, r in picked.items()}, missing


def compute_sample(ds: Dataset, spec: Spec) -> Sample:  # pylint: disable=too-many-locals
    """Read and summarise the spec's sample. One read per sampled date."""
    picked, missing = sample_indices(ds, spec)
    indices = sorted(picked)
    n_grid = ds.data.shape[-1]
    points = sample_points(n_grid, spec.sampling)
    summary = np.full((len(indices), len(ds.variables), len(SUMMARY)), np.nan)
    flags = {n: list(r.flag_values) for n, r in spec.ranges().items() if r.flag_values}
    var_flags = [flags.get(n) for n in ds.variables]
    for k, t in enumerate(indices):
        step = ds.step(t)
        summary[k] = summarise(step, None if len(points) == n_grid else points, var_flags)
    window = statistics_window(ds)
    n_members = int(np.prod(ds.data.shape[2:-1]))
    stored = {s: ds.statistic(s) for s in STATISTICS if s in ds.arrays}  # one read per array
    return Sample(
        variables=list(ds.variables),
        dates=[ds.date_str(t) for t in indices],
        time_index=indices,
        random=np.array([picked[t] for t in indices], dtype=bool),
        summary=summary,
        points_per_date=len(points) * n_members,
        values_per_date=n_grid * n_members,
        n_window=sum(1 for t in range(*window) if t not in ds.missing_indices),
        missing=missing,
        stored={
            name: {
                s: _num(values[i])
                for s, values in stored.items()
            }
            for i, name in enumerate(ds.variables)
        },
        seed=spec.sampling.seed,
        flags=flags,
    )


def summarise(step: np.ndarray,
              points: np.ndarray | None = None,
              flags: Sequence[Sequence[float] | None] | None = None) -> np.ndarray:
    """(variable, ..., grid) -> (variable, SUMMARY) in float64.

    Minimum, maximum and the NaN and inf counts over the full domain; mean and stdev of the finite
    values (and their number, `n_finite`) on the grid `points`, all points if None. An inf must not
    turn the summaries into inf/NaN. No float64 copy of the field: reductions on the stored dtype,
    with a float64 accumulator for mean and stdev.

    `flags` has one entry per variable (None or empty: no flag values). Flags stay in all the
    summaries above; `valid_minimum`/`valid_maximum` are the min/max of the finite non-flag values
    of the full domain (NaN if none) and `n_flag` counts the flags (without flags: the minimum,
    maximum and 0)."""
    x = step.reshape(step.shape[0], -1)
    sub = x if points is None else step[..., points].reshape(step.shape[0], -1)
    out = np.full((x.shape[0], len(SUMMARY)), np.nan)
    for i in range(x.shape[0]):
        finite = np.isfinite(x[i])
        n_finite = int(np.count_nonzero(finite))
        n_nan = int(np.count_nonzero(np.isnan(x[i])))
        out[i, 4:6] = n_nan, x[i].size - n_finite - n_nan
        values = x[i] if n_finite == x[i].size else x[i][finite]
        if values.size:
            out[i, 2:4] = values.min(), values.max()
        var_flags = flags[i] if flags is not None else None
        if var_flags:
            is_flag = flag_mask(x[i], var_flags)
            out[i, 9] = np.count_nonzero(is_flag)
            valid = x[i][finite & ~is_flag]
            if valid.size:
                out[i, 7:9] = valid.min(), valid.max()
        else:
            out[i, 7:9] = out[i, 2:4]
            out[i, 9] = 0
        if points is not None:
            finite = np.isfinite(sub[i])
            n_finite = int(np.count_nonzero(finite))
            values = sub[i] if n_finite == sub[i].size else sub[i][finite]
        out[i, 6] = n_finite
        if values.size:
            out[i, :2] = values.mean(dtype="f8"), values.std(dtype="f8")
    return out


def estimate(  # pylint: disable=too-many-locals
        summary: np.ndarray,
        n_window: int,
        variables: list[str],
        seed: int,
        full_domain: bool = True) -> dict[str, dict[str, Any]]:
    """Estimate the window statistics from the per-date summaries of a random sample of dates.

    The mean and the (population) variance pool the per-date means and variances, weighted by the
    number of finite values of each date (equal weights without NaNs), as anemoi computes the stored
    statistics of a variable with NaNs; both are exact when every date of the window and every point
    was read (`exact`). The standard error of the
    mean is analytic (dates are the sampling unit, with finite population correction `fpc`); the
    one of the stdev comes from a seeded bootstrap over the dates. Both cover the sampling of
    dates only, and are estimated from the sample, so they are unreliable for small samples of
    skewed variables. Minimum and maximum can only be bounded by a sample.
    """
    n = len(summary)
    m, s, lo, hi = (summary[:, :, SUMMARY.index(c)] for c in ("mean", "stdev", "minimum", "maximum"))
    w = summary[:, :, SUMMARY.index("n_finite")]
    fpc = math.sqrt(max(0.0, 1 - n / n_window)) if n_window else 0.0
    exact = n == n_window and full_domain
    out: dict[str, dict[str, Any]] = {
        # minimum/maximum are summarised over the full domain: exact as soon as every date was read
        name: {
            "n_dates": n,
            "exact": exact,
            "exact_extremes": n == n_window,
            "fpc": fpc
        }
        for name in variables
    }
    if n == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, stdev = _pooled(m, s, w, axis=0)
        se_mean = np.nanstd(m, axis=0, ddof=1) / math.sqrt(n) * fpc if n > 1 else np.full(len(variables), np.nan)
        if n > 1 and fpc > 0:
            idx = np.random.default_rng([seed, 2]).integers(0, n, size=(N_BOOTSTRAP, n))
            boot = _pooled(m[idx], s[idx], w[idx], axis=1)[1]
            se_stdev = boot.std(axis=0, ddof=1) * fpc
        else:
            se_stdev = np.zeros(len(variables)) if fpc == 0 else np.full(len(variables), np.nan)
    for i, name in enumerate(variables):
        out[name] |= {
            "mean": _num(mean[i]),
            "stdev": _num(stdev[i]),
            "minimum": _num(lo[:, i].min()),
            "maximum": _num(hi[:, i].max()),
            "se_mean": _num(se_mean[i]),
            "se_stdev": _num(se_stdev[i])
        }
    return out


def _pooled(m: np.ndarray, s: np.ndarray, w: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Pooled mean and stdev of groups with means `m`, stdevs `s` and `w` values each, along `axis`.
    Groups without a finite value (NaN mean) have no weight; no group at all gives NaN."""
    w = np.where(np.isfinite(m) & np.isfinite(s), w, 0.0)
    m, s = np.nan_to_num(m), np.nan_to_num(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = w.sum(axis=axis)
        mean = (w * m).sum(axis=axis) / total
        var = (w * (s**2 + (m - np.expand_dims(mean, axis))**2)).sum(axis=axis) / total
    return mean, np.sqrt(var)


def _num(x: Any, kind: Callable[[float], float] = float) -> float | None:
    """A JSON/YAML-safe number: NaN and inf become None."""
    x = float(x)
    return kind(x) if math.isfinite(x) else None


# --- stats files -----------------------------------------------------------------------------


def provenance(ds: Dataset, fmt: str) -> dict[str, Any]:
    """The header of a stats or tolerance file: format, dataset, creation time and generator."""
    return {
        "format": fmt,
        "dataset": {
            "name": ds.name,
            "uuid": ds.attrs.get("uuid")
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": f"varda-dataset-checker-hive {__version__}",
    }


def stats_file(ds: Dataset, spec: Spec, sample: Sample) -> dict[str, Any]:
    """The stats file content for `sample` of `ds`."""
    window = statistics_window(ds)
    return provenance(ds, FORMAT) | {
        "sampling": {
            "seed":
            spec.sampling.seed,
            "n_dates":
            spec.sampling.n_dates,
            "n_points":
            spec.sampling.n_points,
            "include_dates": [str(d) for d in spec.sampling.include_dates],
            "n_grid":
            ds.data.shape[-1],
            "full_domain":
            sample.full_domain,
            "statistics_window": [
                ds.date_str(window[0]) if window[0] < len(ds.dates) else None,
                ds.date_str(window[1] - 1) if window[1] > 0 else None,
            ],
            "n_window":
            sample.n_window,
            "missing_dates":
            sample.missing,
            # The sampled dates, and which of them are random dates inside the statistics window.
            "dates":
            sample.dates,
            "random":
            sample.random.tolist(),
        },
        "variables": sample.variables,
        "stored": sample.stored,
        "sample": sample.aggregate(),
    }


def compute_stats(ds: Dataset, spec: Spec) -> dict[str, Any]:
    """Read the sample of `ds` and build its stats file content."""
    return stats_file(ds, spec, compute_sample(ds, spec))


def save_stats(stats: dict, path: str | Path | None) -> None:
    """Write a stats (or tolerance) file, see `honeycomb.files.save_file`."""
    save_file(stats, path)


def load_stats(path: str | Path) -> dict:
    return read_file(path, FORMAT, "stats file", "hive ... --out", section="stats")


# --- comparing two stats files ---------------------------------------------------------------

IDENTICAL, CLOSE, DIFFERS = "identical", "close", "differs"


@dataclass
class Comparison:
    findings: list[Finding] = field(default_factory=list)
    # variable -> {"identical": n, "close": n, "differs": n} over all compared cells
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def compare_stats(stats: dict, reference: dict, atol_stdev: float = 0.0) -> Comparison:
    """Compare `stats` with `reference` cell by cell: the stored statistics and the sample
    statistics of every variable. The sample statistics are only comparable for the same sample:
    same seed, points, grid and dates."""
    result = Comparison()
    variables = [v for v in reference["variables"] if v in stats["variables"]]
    for name in sorted(set(stats["variables"]) - set(reference["variables"])):
        result.findings.append(error("variable not in reference stats", variable=name))
    for name in sorted(set(reference["variables"]) - set(stats["variables"])):
        result.findings.append(error("variable missing (present in reference stats)", variable=name))

    sampling_findings, comparable = _compare_sampling(stats["sampling"], reference["sampling"])
    result.findings.extend(sampling_findings)
    sections = ["stored", "sample"] if comparable else ["stored"]

    for name in variables:
        scale = reference["stored"][name].get("stdev") or reference["sample"][name].get("stdev") or 0.0
        pairs = [(stats[section].get(name, {}), reference[section][name], section) for section in sections]
        result.counts[name] = _compare_variable(result.findings, name, pairs, atol_stdev * abs(scale))

    identical = [v for v, c in result.counts.items() if c[CLOSE] == c[DIFFERS] == 0]
    result.findings.append(
        info(f"{len(identical)}/{len(variables)} variables identical ({' and '.join(sections)} statistics)",
             details={"variables": identical}))
    return result


def _compare_variable(findings: list[Finding], name: str, sections: list[tuple[dict, dict, str]],
                      atol: float) -> dict[str, int]:
    """Classify every statistic of one variable in (new, reference, section name) pairs; append a
    finding per non-identical one and return the counts per verdict."""
    counts = {IDENTICAL: 0, CLOSE: 0, DIFFERS: 0}
    for new, ref, section in sections:
        for stat, ref_value in ref.items():
            value = new.get(stat)
            verdict = _classify(value, ref_value, 0.0 if stat in COUNTS else atol)
            counts[verdict] += 1
            if verdict != IDENTICAL:
                # a changed flag share is a processing change, never an error
                findings.append(_finding(verdict, f"{section} {stat}", name, value, ref_value, stat == "flag_share"))
    return counts


def _compare_sampling(new: dict, ref: dict) -> tuple[list[Finding], bool]:
    """Findings on the samples of two stats files, and whether their sample statistics compare:
    the same points (seed, n_points, grid) are needed; differing dates are reported."""
    findings = []
    # The seed and n_points pick the points; with the full domain they don't matter.
    keys = ["n_grid"] if new.get("full_domain") and ref.get("full_domain") else ["n_grid", "seed", "n_points"]
    mismatch = {k: [new.get(k), ref.get(k)] for k in keys if new.get(k) != ref.get(k)}
    if mismatch:
        findings.append(
            error("sample differs from the reference's (seed, points or grid): sample statistics not compared",
                  details=mismatch))
    only_ref = [d for d in ref["dates"] if d not in set(new["dates"])]
    only_new = [d for d in new["dates"] if d not in set(ref["dates"])]
    if only_ref or only_new:
        findings.append(
            error(
                f"sampled dates differ from the reference's: {len(only_ref)} missing, {len(only_new)} extra; "
                "sample statistics not compared",
                details={
                    "missing": only_ref[:20],
                    "extra": only_new[:20]
                }))
    # Sample statistics of different samples differ anyway: comparing them would only repeat this.
    return findings, not (mismatch or only_ref or only_new)


def _classify(value: float | None, reference: float | None, atol: float) -> str:
    if value == reference:  # also both None (NaN)
        return IDENTICAL
    if value is None or reference is None:
        return DIFFERS
    return CLOSE if abs(value - reference) <= atol else DIFFERS


def _finding(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        verdict: str,
        what: str,
        variable: str,
        value: float | None,
        reference: float | None,
        warn_only: bool = False) -> Finding:
    make = error if verdict == DIFFERS and not warn_only else warning
    diff = abs(value - reference) if value is not None and reference is not None else None
    return make(f"{what} {verdict}: {value} vs {reference}",
                variable=variable,
                details={
                    "value": value,
                    "reference": reference,
                    "abs_diff": diff
                })
