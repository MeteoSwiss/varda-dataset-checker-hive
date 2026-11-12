# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The sample stats-bee checks, and stats files: a small, fixed-size YAML or JSON fingerprint of a
dataset (the format follows the file extension), written by `stats-bee dump` and compared by
`stats-bee compare`.

The sample is `n_dates` seeded dates, stratified over the whole time range (plus `include_dates`).
Every sampled date is read whole (one chunk), so per date and variable:
- minimum, maximum and the NaN and inf counts cover the **full domain**: cheap, and a single bad
  point is never missed;
- mean and stdev of the finite values cover `n_points` seeded grid points, the same on every date
  (the costly part).
These per-date summaries live in memory only (`Sample`): the checks use them, the stats file does not.

A stats file holds, per variable:
- `stored`: the statistics stored in the dataset (minimum, maximum, mean, stdev)
- `sample`: over all sampled dates, minimum, maximum, NaN and inf counts of the full domain, and
  mean and stdev of the finite values on the sampled points

Its size depends only on the number of variables (and the list of sampled dates in `sampling`),
not on the size of the dataset. Two stats files of the same sample (same seed, points and dates,
see `--dates-from`) compare cell by cell, without any statistics: a value is identical, close
(within `compare_atol_stdev` * stdev) or differs. Floats are written with `repr`, which
round-trips exactly, and NaN/inf as null. Dates in the dataset's `missing_dates` are never sampled.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import yaml

from ... import __version__
from ...honeycomb import Dataset, Finding, Spec, error, info, warning
from ...honeycomb.dataset import STATISTICS
from ...honeycomb.sampling import random_time_indices, sample_points

FORMAT = "varda-hive-stats/4"
# Per-date summary of one variable: mean/stdev (and n_finite, their weight) on the sampled points,
# minimum/maximum and the NaN/inf counts on the full domain.
SUMMARY = ("mean", "stdev", "minimum", "maximum", "n_nan", "n_inf", "n_finite")
COUNTS = ("n_nan", "n_inf", "n_finite")
SAMPLE_STATS = ("minimum", "maximum", "mean", "stdev", "n_nan", "n_inf")
N_BOOTSTRAP = 200


def statistics_window(ds: Dataset) -> tuple[int, int]:
    """Time indices [lo, hi) the stored statistics were computed over."""
    lo, hi = 0, len(ds.dates)
    if start := ds.attrs.get("statistics_start_date"):
        lo = int(np.searchsorted(ds.dates, np.datetime64(start, "s"), side="left"))
    if end := ds.attrs.get("statistics_end_date"):
        hi = int(np.searchsorted(ds.dates, np.datetime64(end, "s"), side="right"))
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
    missing: list[str]  # requested dates (`--dates-from`, `include_dates`) that could not be sampled
    stored: dict[str, dict[str, Any]]  # None for NaN
    seed: int

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

    def aggregate(self) -> dict[str, dict[str, Any]]:
        """Over all sampled dates: minimum, maximum and NaN/inf counts of the full domain, and mean and
        stdev of the finite values on the sampled points. Exact from the per-date summaries: the mean
        and the pooled variance are weighted by the number of finite sampled values of each date."""
        m, s, lo, hi, n_nan, n_inf, weight = (self.summary[:, :, j] for j in range(len(SUMMARY)))
        total = weight.sum(axis=0)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN variables give NaN
            mean = np.nansum(weight * m, axis=0) / total
            var = np.nansum(weight * (s**2 + (m - mean)**2), axis=0) / total
            lo, hi = np.nanmin(lo, axis=0), np.nanmax(hi, axis=0)
        return {
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


def sample_indices(ds: Dataset, spec: Spec, dates_from: dict | None = None) -> tuple[dict[int, bool], list[str]]:
    """{time index: is random} of the sample, and the requested dates that could not be sampled.

    The random dates are stratified over the whole time range, so the value checks also cover the
    dates after the statistics window; `is random` is True only for those inside the window.
    """
    window = statistics_window(ds)
    date_strs = [str(d) for d in ds.dates]
    excluded = ds.missing_indices
    if dates_from is not None:
        position = {d: t for t, d in enumerate(date_strs) if t not in excluded}
        wanted = list(zip(dates_from["sampling"]["dates"], dates_from["sampling"]["random"]))
        missing = [d for d, _ in wanted if d not in position]
        picked = {position[d]: r for d, r in wanted if d in position}
    else:
        n_times = len(ds.dates)
        missing = [date_strs[t] for t in spec.sampling.include_dates if t in excluded]
        picked = {t: False for t in spec.sampling.include_dates if 0 <= t < n_times and t not in excluded}
        picked |= {t: True for t in random_time_indices(n_times, spec.sampling, stratified=True, exclude=excluded)}
    return {t: r and window[0] <= t < window[1] for t, r in picked.items()}, missing


def compute_sample(ds: Dataset, spec: Spec, dates_from: dict | None = None) -> Sample:
    """Read and summarise the sample: the spec's, or the dates of `dates_from` (another stats file)
    so that the two compare. One read per sampled date."""
    picked, missing = sample_indices(ds, spec, dates_from)
    indices = sorted(picked)
    n_grid = ds.data.shape[-1]
    points = sample_points(n_grid, spec.sampling)
    summary = np.full((len(indices), len(ds.variables), len(SUMMARY)), np.nan)
    for k, t in enumerate(indices):
        step = ds.step(t)
        summary[k] = summarise(step, None if len(points) == n_grid else points)
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
    )


def summarise(step: np.ndarray, points: np.ndarray | None = None) -> np.ndarray:
    """(variable, ..., grid) -> (variable, SUMMARY) in float64.

    Minimum, maximum and the NaN and inf counts over the full domain; mean and stdev of the finite
    values (and their number, `n_finite`) on the grid `points`, all points if None. An inf must not
    turn the summaries into inf/NaN. No float64 copy of the field: reductions on the stored dtype,
    with a float64 accumulator for mean and stdev."""
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

    Every date has the same number of values, so the mean is the mean of the per-date means and
    the (population) variance is mean(per-date variance) + var(per-date means); both are exact
    when every date of the window and every point was read (`exact`). The standard error of the
    mean is analytic (dates are the sampling unit, with finite population correction `fpc`); the
    one of the stdev comes from a seeded bootstrap over the dates. Both cover the sampling of
    dates only, and are estimated from the sample, so they are unreliable for small samples of
    skewed variables. Minimum and maximum can only be bounded by a sample.
    """
    n = len(summary)
    m, s, lo, hi = (summary[:, :, j] for j in range(4))
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
        mean = m.mean(axis=0)
        stdev = np.sqrt((s**2).mean(axis=0) + m.var(axis=0))
        se_mean = m.std(axis=0, ddof=1) / math.sqrt(n) * fpc if n > 1 else np.full(len(variables), np.nan)
        if n > 1 and fpc > 0:
            idx = np.random.default_rng([seed, 2]).integers(0, n, size=(N_BOOTSTRAP, n))
            boot = np.sqrt((s[idx]**2).mean(axis=1) + m[idx].var(axis=1))
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


def stats_file(ds: Dataset, spec: Spec, sample: Sample, dates_from: dict | None = None) -> dict[str, Any]:
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
            "include_dates":
            spec.sampling.include_dates,
            "n_grid":
            ds.data.shape[-1],
            "full_domain":
            sample.full_domain,
            "dates_from":
            dates_from["dataset"]["name"] if dates_from else None,
            "statistics_window": [
                ds.date_str(window[0]) if window[0] < len(ds.dates) else None,
                ds.date_str(window[1] - 1) if window[1] > 0 else None,
            ],
            "n_window":
            sample.n_window,
            "missing_dates":
            sample.missing,
            # The sampled dates, and which of them are random dates inside the statistics window,
            # so `--dates-from` reproduces the sample on a dataset with another time axis.
            "dates":
            sample.dates,
            "random":
            sample.random.tolist(),
        },
        "variables": sample.variables,
        "stored": sample.stored,
        "sample": sample.aggregate(),
    }


def compute_stats(ds: Dataset, spec: Spec, dates_from: dict | None = None) -> dict[str, Any]:
    """Read the sample of `ds` and build its stats file content."""
    return stats_file(ds, spec, compute_sample(ds, spec, dates_from), dates_from)


YAML_SUFFIXES = (".yaml", ".yml")
_Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def save_stats(stats: dict, path: str | Path | None) -> None:
    """Write YAML to a `.yaml`/`.yml` path, JSON to any other path, JSON to stdout if it is None or "-".

    YAML: dicts in block style, every list on one line, so a diff shows which variable and
    statistic changed. Also used for tolerance files."""
    if path not in (None, "-") and Path(path).suffix in YAML_SUFFIXES:
        text = yaml.dump(stats, Dumper=_Dumper, sort_keys=False, default_flow_style=None, width=120)
    else:
        text = json.dumps(stats, indent=1, allow_nan=False) + "\n"
    if path in (None, "-"):
        print(text, end="")
    else:
        Path(path).write_text(text, encoding="utf-8")


def read_file(path: str | Path, fmt: str, what: str) -> dict:
    """Load a YAML/JSON file written by `save_stats` and check its `format`."""
    text = Path(path).read_text(encoding="utf-8")
    content = yaml.load(text, Loader=_Loader) if Path(path).suffix in YAML_SUFFIXES else json.loads(text)
    found = content.get("format") if isinstance(content, dict) else None
    if found != fmt:
        hint = " (an older stats file: regenerate it with `stats-bee dump`)" if str(found).startswith(
            "varda-hive-stats/") else ""
        raise ValueError(f"{path}: not a {what} of format {fmt!r}, found {found!r}{hint}")
    return content


def load_stats(path: str | Path) -> dict:
    return read_file(path, FORMAT, "stats file")


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
                findings.append(_finding(verdict, f"{section} {stat}", name, value, ref_value))
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
                f"sampled dates differ from the reference's: {len(only_ref)} missing, {len(only_new)} extra "
                "(use --dates-from); sample statistics not compared",
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


def _finding(verdict: str, what: str, variable: str, value: float | None, reference: float | None) -> Finding:
    make = error if verdict == DIFFERS else warning
    diff = abs(value - reference) if value is not None and reference is not None else None
    return make(f"{what} {verdict}: {value} vs {reference}",
                variable=variable,
                details={
                    "value": value,
                    "reference": reference,
                    "abs_diff": diff
                })
