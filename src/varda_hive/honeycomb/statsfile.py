"""Stats files: a small JSON fingerprint of a dataset's values, written by
`stats-bee dump` and compared by `stats-bee compare`.

A stats file holds, for a seeded sample of dates:
- `stored`: the statistics stored in the dataset (mean, stdev, minimum, maximum)
- `per_date`: mean, stdev, minimum, maximum and NaN count of every variable at every sampled date
- `recomputed`: the statistics estimated from the random dates inside the statistics window,
  with their sampling errors (`se_*`, 0 when every date of the window was read)

Two stats files built on the same dates (see `--dates-from`) compare cell by cell, without
any statistics: a value is identical, close (within `compare_atol_stdev` * stdev) or differs.
Floats are written with `repr`, which round-trips exactly, and NaN as null.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .. import __version__
from .dataset import STATISTICS, Dataset
from .findings import Finding, error, info, warning
from .sampling import random_time_indices
from .spec import Spec

FORMAT = "varda-hive-stats/1"
PER_DATE = ("mean", "stdev", "minimum", "maximum", "n_nan")
N_BOOTSTRAP = 200


def statistics_window(ds: Dataset) -> tuple[int, int]:
    """Time indices [lo, hi) the stored statistics were computed over."""
    lo, hi = 0, len(ds.dates)
    if start := ds.attrs.get("statistics_start_date"):
        lo = int(np.searchsorted(ds.dates, np.datetime64(start, "s"), side="left"))
    if end := ds.attrs.get("statistics_end_date"):
        hi = int(np.searchsorted(ds.dates, np.datetime64(end, "s"), side="right"))
    return lo, hi


def compute_stats(ds: Dataset, spec: Spec, dates_from: dict | None = None) -> dict[str, Any]:
    """Build the stats file content for `ds`.

    The dates are the spec's sample (stratified random dates in the statistics window plus
    `include_dates`), or exactly those of `dates_from`, another stats file, so that the two
    can be compared date by date.
    """
    window = statistics_window(ds)
    date_strs = [str(d) for d in ds.dates]
    if dates_from is not None:
        position = {d: t for t, d in enumerate(date_strs)}
        wanted = list(zip(dates_from["dates"], dates_from["random"]))
        missing = [d for d, _ in wanted if d not in position]
        picked = {position[d]: r for d, r in wanted if d in position}
    else:
        missing = []
        picked = {t: False for t in spec.sampling.include_dates if 0 <= t < len(ds.dates)}
        picked |= {t: True for t in random_time_indices(len(ds.dates), spec.sampling, window, stratified=True)}
    indices = sorted(picked)

    per_date = np.full((len(indices), len(ds.variables), len(PER_DATE)), np.nan)
    for k, t in enumerate(indices):
        per_date[k] = _summarise(ds.step(t))

    random = np.array([picked[t] and window[0] <= t < window[1] for t in indices], dtype=bool)
    return {
        "format": FORMAT,
        "dataset": {"name": ds.name, "path": str(ds.path.resolve()), "uuid": ds.attrs.get("uuid")},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": f"varda-dataset-checker-hive {__version__}",
        "sampling": {
            "seed": spec.sampling.seed,
            "n_dates": spec.sampling.n_dates,
            "include_dates": spec.sampling.include_dates,
            "dates_from": dates_from["dataset"]["name"] if dates_from else None,
            "statistics_window": [date_strs[window[0]] if window[0] < len(date_strs) else None,
                                  date_strs[window[1] - 1] if window[1] > 0 else None],
            "n_window": window[1] - window[0],
            "missing_dates": missing,
        },
        "variables": list(ds.variables),
        "dates": [date_strs[t] for t in indices],
        "time_index": indices,
        # True for dates of the random sample inside the statistics window: only those are used
        # to estimate the statistics, the `include_dates` would bias the estimate.
        "random": random.tolist(),
        "stored": {
            name: {s: _num(ds.statistic(s)[i]) for s in STATISTICS if s in ds.arrays}
            for i, name in enumerate(ds.variables)
        },
        "per_date": {
            name: {s: [_num(x, int if s == "n_nan" else float) for x in per_date[:, i, j]]
                   for j, s in enumerate(PER_DATE)}
            for i, name in enumerate(ds.variables)
        },
        "recomputed": _recompute(per_date[random], window[1] - window[0], ds.variables, spec.sampling.seed),
    }


def _summarise(step: np.ndarray) -> np.ndarray:
    """(variable, ...) -> (variable, PER_DATE) in float64, ignoring NaNs."""
    x = step.reshape(step.shape[0], -1).astype("f8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN fields give NaN, which is what we want
        return np.stack([np.nanmean(x, axis=1), np.nanstd(x, axis=1), np.nanmin(x, axis=1),
                         np.nanmax(x, axis=1), np.isnan(x).sum(axis=1)], axis=1)


def _recompute(per_date: np.ndarray, n_window: int, variables: list[str], seed: int) -> dict[str, dict]:
    """Estimate the window statistics from the per-date summaries of a random sample of dates.

    Every date has the same number of grid points, so the mean is the mean of the per-date
    means and the (population) variance is mean(per-date variance) + var(per-date means); both
    are exact when every date of the window was read. The standard error of the mean is
    analytic (dates are the sampling unit, with finite population correction `fpc`); the one of
    the stdev comes from a seeded bootstrap over the dates. Both are estimated from the sample,
    so they are unreliable for small samples of skewed variables. Minimum and maximum can only
    be bounded by a sample, so they carry no standard error.
    """
    n = len(per_date)
    m, s, lo, hi = (per_date[:, :, j] for j in range(4))
    fpc = math.sqrt(max(0.0, 1 - n / n_window)) if n_window else 0.0
    out = {name: {"n_dates": n, "exact": n == n_window, "fpc": fpc} for name in variables}
    if n == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = m.mean(axis=0)
        stdev = np.sqrt((s**2).mean(axis=0) + m.var(axis=0))
        se_mean = m.std(axis=0, ddof=1) / math.sqrt(n) * fpc if n > 1 else np.full(len(variables), np.nan)
        if n > 1 and fpc > 0:
            idx = np.random.default_rng([seed, 2]).integers(0, n, size=(N_BOOTSTRAP, n))
            boot = np.sqrt((s[idx] ** 2).mean(axis=1) + m[idx].var(axis=1))
            se_stdev = boot.std(axis=0, ddof=1) * fpc
        else:
            se_stdev = np.zeros(len(variables)) if fpc == 0 else np.full(len(variables), np.nan)
    for i, name in enumerate(variables):
        out[name] |= {"mean": _num(mean[i]), "stdev": _num(stdev[i]), "minimum": _num(lo[:, i].min()),
                      "maximum": _num(hi[:, i].max()), "se_mean": _num(se_mean[i]), "se_stdev": _num(se_stdev[i])}
    return out


def _num(x, kind=float):
    x = float(x)
    return None if math.isnan(x) else kind(x)


def save_stats(stats: dict, path: str | Path | None) -> None:
    """Write to `path`, or to stdout if it is None or "-"."""
    text = json.dumps(stats, indent=1) + "\n"
    if path in (None, "-"):
        print(text, end="")
    else:
        Path(path).write_text(text)


def load_stats(path: str | Path) -> dict:
    stats = json.loads(Path(path).read_text())
    if stats.get("format") != FORMAT:
        raise ValueError(f"{path}: not a stats file of format {FORMAT!r}")
    return stats


# --- comparing two stats files ---------------------------------------------------------------

IDENTICAL, CLOSE, DIFFERS = "identical", "close", "differs"


@dataclass
class Comparison:
    findings: list[Finding] = field(default_factory=list)
    # variable -> {"identical": n, "close": n, "differs": n} over all compared cells
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def compare_stats(stats: dict, reference: dict, atol_stdev: float = 0.0) -> Comparison:
    """Compare `stats` with `reference` cell by cell: the stored statistics, and every per-date
    summary on the dates both files have. Dates are matched by value, not by position."""
    result = Comparison()
    variables = [v for v in reference["variables"] if v in stats["variables"]]
    for name in sorted(set(stats["variables"]) - set(reference["variables"])):
        result.findings.append(error("variable not in reference stats", variable=name))
    for name in sorted(set(reference["variables"]) - set(stats["variables"])):
        result.findings.append(error("variable missing (present in reference stats)", variable=name))

    position = {d: k for k, d in enumerate(stats["dates"])}
    common = [(position[d], k, d) for k, d in enumerate(reference["dates"]) if d in position]
    only_ref = [d for d in reference["dates"] if d not in position]
    only_new = sorted(set(stats["dates"]) - set(reference["dates"]))
    if only_ref:
        result.findings.append(error(f"{len(only_ref)} reference dates not in stats (use --dates-from)",
                                     details={"dates": only_ref[:20]}))
    if only_new:
        result.findings.append(info(f"{len(only_new)} dates not in reference stats, not compared",
                                    details={"dates": only_new[:20]}))

    for name in variables:
        counts = result.counts.setdefault(name, {IDENTICAL: 0, CLOSE: 0, DIFFERS: 0})
        scale = reference["stored"][name].get("stdev") or reference["recomputed"][name].get("stdev") or 0.0
        atol = atol_stdev * abs(scale)
        for stat, ref_value in reference["stored"][name].items():
            value = stats["stored"].get(name, {}).get(stat)
            verdict = _classify(value, ref_value, atol)
            counts[verdict] += 1
            if verdict != IDENTICAL:
                result.findings.append(_finding(verdict, f"stored {stat}", name, value, ref_value))
        for stat in PER_DATE:
            new, ref = stats["per_date"][name][stat], reference["per_date"][name][stat]
            cells = [(d, new[a], ref[b], _classify(new[a], ref[b], 0.0 if stat == "n_nan" else atol))
                     for a, b, d in common]
            for verdict in (IDENTICAL, CLOSE, DIFFERS):
                counts[verdict] += sum(c[3] == verdict for c in cells)
            for verdict, make in ((DIFFERS, error), (CLOSE, warning)):
                bad = [c for c in cells if c[3] == verdict]
                if bad:
                    diffs = [abs(x - y) for _, x, y, _ in bad if x is not None and y is not None]
                    result.findings.append(make(
                        f"{stat} {verdict} at {len(bad)}/{len(cells)} dates", variable=name,
                        time=bad[0][0] if len(bad) == 1 else None,
                        details={"dates": [c[0] for c in bad[:10]], "values": [c[1] for c in bad[:10]],
                                 "reference": [c[2] for c in bad[:10]],
                                 "max_abs_diff": max(diffs) if diffs else None}))

    identical = [v for v, c in result.counts.items() if c[CLOSE] == c[DIFFERS] == 0]
    result.findings.append(info(f"{len(identical)}/{len(variables)} variables identical on "
                                f"{len(common)} common dates", details={"variables": identical}))
    return result


def _classify(value, reference, atol: float) -> str:
    if value == reference:  # also both None (NaN)
        return IDENTICAL
    if value is None or reference is None:
        return DIFFERS
    return CLOSE if abs(value - reference) <= atol else DIFFERS


def _finding(verdict: str, what: str, variable: str, value, reference) -> Finding:
    make = error if verdict == DIFFERS else warning
    diff = abs(value - reference) if value is not None and reference is not None else None
    return make(f"{what} {verdict}: {value} vs {reference}", variable=variable,
                details={"value": value, "reference": reference, "abs_diff": diff})
