# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""stats-bee: checks on the values and stored statistics.

All checks work on one sample of dates x points (see statsfile.py), read once per run and shared
via `ctx.cache`.
"""
import math
from typing import Any, Generator, Iterator

import numpy as np

from ...honeycomb import CheckContext, Finding, bee_dump, check, dumps, error, info, skipped, warning
from ...honeycomb.dataset import is_zarr
from ...honeycomb.files import provides
from .calibrate import load_tolerances, mismatch
from .statsfile import FORMAT, Sample, compare_stats, compute_sample, load_stats, stats_file


def _reference(ctx: CheckContext) -> tuple[dict | None, str]:
    """The reference stats (a ref or stats file), or None and why there is none."""
    if (reason := ctx.reference_missing()) is not None:
        return None, reason
    path = ctx.reference_path
    assert path is not None
    if is_zarr(path) or not provides(path, "stats", FORMAT):
        return None, f"the reference {path} has no stats section (a zarr or a meta file?)"
    return load_stats(path), ""


def _sample(ctx: CheckContext) -> Sample:
    if "sample" not in ctx.cache:
        ctx.cache["sample"] = compute_sample(ctx.dataset, ctx.spec)
    return ctx.cache["sample"]


@check("stats")
def statistics_consistent(ctx: CheckContext) -> Iterator[Finding]:
    """Stored mean/stdev/min/max are self-consistent and match the sample within the sampling error.

    Self-consistent: no NaN, min <= mean <= max, stdev 0 exactly when min == max. A sample can
    only bound min/max, exact with `n_dates: all`; mean and stdev are exact with `n_dates: all` and
    `n_points: all`.
    With `calibrated_tolerances` in the spec, mean and stdev use the calibrated tolerances.
    """
    sample = _sample(ctx)
    calibrated = yield from _calibrated(ctx)
    estimates = sample.estimate()
    for name in sample.variables:
        stored, rec = sample.stored[name], estimates[name]
        if not stored:
            yield skipped("no stored statistics", variable=name)
            continue
        values = [stored.get(s) for s in ("minimum", "mean", "stdev", "maximum")]
        if None in values:
            yield error("stored statistics contain NaN or inf", variable=name, details=stored)
            continue
        lo, mean, sd, hi = (float(v) for v in values if v is not None)
        # Internal consistency of the stored values: exact, no tolerance needed.
        if not lo <= mean <= hi:
            yield error(f"stored statistics out of order: minimum {lo}, mean {mean}, maximum {hi}",
                        variable=name,
                        details=stored)
        if sd < 0 or (sd == 0) != (lo == hi):
            yield error(f"stored stdev {sd} inconsistent with minimum {lo} and maximum {hi}",
                        variable=name,
                        details=stored)
        if any(rec.get(s) is None for s in ("mean", "stdev", "minimum", "maximum")):
            continue  # no finite value on a sampled date: reported by finite_values
        moments = {"mean": mean, "stdev": sd}
        yield from _compare_moments(ctx, name, moments, rec, calibrated.get(name, {}))
        yield from _compare_extremes(ctx, name, lo, hi, rec)
    n = next((rec["n_dates"] for rec in estimates.values()), 0)
    yield info(f"statistics recomputed from {n} dates x {sample.points_per_date} values",
               details=_resolution(sample, estimates, calibrated, ctx.spec.tolerances.z_threshold))


def _compare_moments(ctx: CheckContext, name: str, stored: dict[str, float], rec: dict,
                     calibrated: dict) -> Iterator[Finding]:
    """Stored vs recomputed mean and stdev: calibrated tolerance if there is one, otherwise ERROR
    beyond the distribution-free bound and WARNING beyond the sample's own standard error."""
    tol, sd = ctx.spec.tolerances, stored["stdev"]
    bound, spatial = _bounds(rec, sd), rec["spatial"]
    for stat in ("mean", "stdev"):
        diff = abs(rec[stat] - stored[stat])
        rounding = tol.statistics_rtol * (abs(stored[stat]) + sd)  # + sd: means close to 0
        details = {
            "stored": stored[stat],
            "recomputed": rec[stat],
            "abs_diff": diff,
            "n_dates": rec["n_dates"],
            "exact": rec["exact"]
        }
        message = f"stored {stat} {stored[stat]:.6g} differs from recomputed {rec[stat]:.6g}"
        if calibrated.get(stat) is not None:
            if diff > calibrated[stat] + rounding:
                yield error(message + " (beyond the calibrated tolerance)",
                            variable=name,
                            details=details | {"tolerance": calibrated[stat]})
            continue
        se = rec[f"se_{stat}"]
        if diff > tol.z_threshold * bound[stat] + rounding:
            yield error(message, variable=name, details=details | {"se_bound": bound[stat]})
        elif se is not None and diff > tol.z_threshold * (se + spatial * sd) + rounding:
            # se can be 0, e.g. for a field that is the same on every sampled date: z is then unbounded.
            yield warning(message + " (beyond the sampling error estimated from the sample)",
                          variable=name,
                          details=details | {"z": diff / se if se > 0 else None})


def _compare_extremes(ctx: CheckContext, name: str, lo: float, hi: float, rec: dict) -> Iterator[Finding]:
    """A sample of dates can only bound the minimum and maximum, unless every date was read: they are
    summarised over the full domain of each date."""
    slack = ctx.spec.tolerances.statistics_rtol * max(abs(lo), abs(hi))
    exact = rec["exact_extremes"]
    if rec["minimum"] < lo - slack or (exact and rec["minimum"] > lo + slack):
        yield error(f"stored minimum {lo:.6g} but the sample reaches {rec['minimum']:.6g}",
                    variable=name,
                    details={
                        "stored": lo,
                        "recomputed": rec["minimum"],
                        "exact": exact
                    })
    if rec["maximum"] > hi + slack or (exact and rec["maximum"] < hi - slack):
        yield error(f"stored maximum {hi:.6g} but the sample reaches {rec['maximum']:.6g}",
                    variable=name,
                    details={
                        "stored": hi,
                        "recomputed": rec["maximum"],
                        "exact": exact
                    })


def _bounds(rec: dict, sd: float) -> dict[str, float]:
    """Distribution-free bounds on the sampling error of mean and stdev, in units of the error.

    Mean: the per-date means can't spread more than the variable itself, so sd * fpc / sqrt(n)
    bounds the error from sampling dates; the fixed point subset adds sd * rec["spatial"]. Beyond
    z times that it's an ERROR. Stdev: only bounded (0) when everything was read.
    """
    return {
        "mean": sd * (rec["fpc"] / math.sqrt(rec["n_dates"]) + rec["spatial"]),
        "stdev": 0.0 if rec["exact"] else math.inf,
    }


def _calibrated(ctx: CheckContext) -> Generator[Finding, None, dict[str, Any]]:
    """Yields an ERROR if the spec's tolerance file doesn't fit this sample; returns its tolerances."""
    if ctx.spec.calibrated_tolerances is None:
        return {}
    tolerances = load_tolerances(ctx.spec.calibrated_tolerances)
    if differs := mismatch(tolerances, ctx.dataset, ctx.spec):
        yield error(
            "calibrated tolerances were made for another sampling, recalibrate with `stats-bee calibrate`; "
            "using the default tolerances",
            details={"file vs now": differs})
        return {}
    return tolerances["variables"]


def _resolution(sample: Sample, estimates: dict, calibrated: dict, z: float) -> dict:
    """Smallest difference to the stored mean the test can detect, per variable: as an ERROR
    (calibrated, or distribution-free bound) and as a WARNING (sample standard error)."""
    out = {}
    for name, rec in estimates.items():
        sd = sample.stored[name].get("stdev")
        if rec.get("mean") is None or sd is None:
            continue
        if calibrated.get(name, {}).get("mean") is not None:
            out[name] = {"error_mean": calibrated[name]["mean"], "calibrated": True}
            continue
        out[name] = {
            "error_mean": z * _bounds(rec, sd)["mean"],
            "warning_mean": z * (rec["se_mean"] + rec["spatial"] * sd) if rec["se_mean"] is not None else None
        }
    return {"min_detectable": out}


@check("stats")
def physical_ranges(ctx: CheckContext) -> Iterator[Finding]:
    """Sampled values and stored min/max lie inside the `min`/`max` of the spec's `variables`.

    Skipped when no variable has a `min` or `max`; variables without one are listed as INFO.
    """
    ranges = ctx.spec.ranges()
    if not ranges:
        yield skipped("no `min`/`max` for any variable in the spec")
        return
    sample = _sample(ctx)
    unlisted = [v for v in sample.variables if v not in ranges]
    if unlisted:
        yield info(f"{len(unlisted)} variables without a physical range", details={"variables": unlisted})
    for name, bounds in ranges.items():
        if name not in sample.variables:
            continue
        lo, hi, units = bounds.min, bounds.max, bounds.units
        per_date = sample.per_date(name)
        for date, vmin, vmax in zip(sample.dates, per_date["minimum"], per_date["maximum"]):
            if vmin is not None and (vmin < lo or vmax > hi):
                yield error(f"values [{vmin:.6g}, {vmax:.6g}] outside physical range [{lo}, {hi}] {units}",
                            variable=name,
                            time=date,
                            details={
                                "minimum": vmin,
                                "maximum": vmax
                            })
        stored = sample.stored[name]
        if stored.get("minimum") is not None and (stored["minimum"] < lo or stored["maximum"] > hi):
            yield error(
                f"stored minimum/maximum [{stored['minimum']:.6g}, {stored['maximum']:.6g}] "
                f"outside physical range [{lo}, {hi}] {units}",
                variable=name)


@check("stats")
def finite_values(ctx: CheckContext) -> Iterator[Finding]:
    """No inf on the sampled dates and points, and NaN only in the variables allowed to have them.

    The variables come from the dataset's `allow_nans` if it is a list, else from the spec; with
    `allow_nans: true` (and no list in the spec) NaNs are not checked.
    """
    allowed = _nans_allowed(ctx)
    sample = _sample(ctx)
    for name in sample.variables:
        per_date = sample.per_date(name)
        for t, date, n_nan, n_inf in zip(sample.time_index, sample.dates, per_date["n_nan"], per_date["n_inf"]):
            if n_nan and allowed is not None and name not in allowed:
                yield error(f"{n_nan} NaN values", variable=name, time=date, details={"time_index": t, "n_nan": n_nan})
            if n_inf:
                yield error(f"{n_inf} infinite values",
                            variable=name,
                            time=date,
                            details={
                                "time_index": t,
                                "n_inf": n_inf
                            })
    if allowed is None:
        yield info("NaNs not checked: dataset allows NaNs")
    elif allowed:
        yield info(f"NaNs allowed in {len(allowed)} variables", details={"variables": sorted(allowed)})


def _nans_allowed(ctx: CheckContext) -> set[str] | None:
    """The variables that may contain NaNs, None for all."""
    attr = ctx.dataset.attrs.get("allow_nans", False)
    if isinstance(attr, list):
        return {str(v) for v in attr}
    spec = ctx.spec.nan_variables()
    if isinstance(spec, set):
        return spec
    return None if attr is True or spec is True else set()


@check("stats")
def statistics_window_valid(ctx: CheckContext) -> Iterator[Finding]:
    """`statistics_start_date` <= `statistics_end_date`, both within the dataset's dates."""
    ds = ctx.dataset
    if len(ds.dates) == 0:
        yield error("dataset has no dates")
        return
    first, last = ds.dates[0], ds.dates[-1]
    bounds = {}
    for key in ("statistics_start_date", "statistics_end_date"):
        if (value := ds.attrs.get(key)) is None:
            yield warning(f"no `{key}` attribute: assuming the statistics cover all dates")
            continue
        bounds[key] = date = np.datetime64(value, "s")
        if not first <= date <= last:
            yield error(f"{key} {date} outside the dataset's dates [{first}, {last}]",
                        details={
                            key: str(date),
                            "first_date": str(first),
                            "last_date": str(last)
                        })
    start, end = bounds.get("statistics_start_date"), bounds.get("statistics_end_date")
    if start is not None and end is not None and start > end:
        yield error(f"statistics_start_date {start} after statistics_end_date {end}")


@check("stats")
def constant_fields(ctx: CheckContext) -> Iterator[Finding]:
    """Fields in `constant_fields` never change; no other field is constant in time or uniform in space.

    `constant_fields` comes from the dataset, or from the spec's variables marked `constant` if the
    dataset has none. A field in
    it that changes is an ERROR; any other field identical on all sampled dates, or uniform in
    space on one date, is a WARNING. Computed forcings are exempt from the uniform-in-space test.
    """
    sample = _sample(ctx)
    listed = ctx.dataset.attrs.get("constant_fields")
    if listed is None:
        listed = ctx.spec.constant_field_names() or []
    # Forcings like cos_julian_day, or insolation at night, are legitimately uniform in space.
    forcings = {v for v, m in ctx.dataset.attrs.get("variables_metadata", {}).items() if m.get("computed_forcing")}
    for name in sample.variables:
        per_date = sample.per_date(name)
        # Dates without any finite value are reported by finite_values, not as a change.
        dated = [(d, s) for d, s in zip(
            sample.dates, zip(per_date["mean"], per_date["stdev"], per_date["minimum"], per_date["maximum"]))
                 if s[0] is not None]
        summaries = [s for _, s in dated]
        if len(summaries) < 2:
            continue
        changed = [d for d, s in dated if s != summaries[0]]
        if name in listed and changed:
            yield error(f"listed as constant field but changes at {len(changed)}/{len(summaries) - 1} dates",
                        variable=name,
                        details={
                            "dates": changed[:10],
                            "first_date": sample.dates[0]
                        })
        elif name not in listed and not changed:
            yield warning(f"identical on all {len(summaries)} sampled dates but not in constant_fields", variable=name)
        # A whole field set to one value (e.g. zeroed), unless the variable is constant anyway.
        stored = sample.stored[name]
        if name not in forcings and stored.get("minimum") != stored.get("maximum"):
            for date, vmin, vmax in zip(sample.dates, per_date["minimum"], per_date["maximum"]):
                if vmin is not None and vmin == vmax:  # over the full domain
                    yield warning(f"spatially constant field (all values {vmin})", variable=name, time=date)


@dumps("stats")
def stats_dump(ctx: CheckContext) -> dict[str, Any]:
    """stats-bee's section of `ref.yaml`: the stats file of the spec's sample."""
    return stats_file(ctx.dataset, ctx.spec, _sample(ctx))


@check("stats")
def matches_reference_stats(ctx: CheckContext) -> Iterator[Finding]:
    """Stored and sample statistics match the reference (`--reference` or the spec's `reference`, a
    ref file), on the same sample: the spec's.

    Skipped without a reference. Compares the dump with the reference's stats section, as `hive compare`.
    """
    reference, why = _reference(ctx)
    if reference is None:
        yield skipped(why)
        return
    yield from compare_stats(bee_dump(ctx, "stats"), reference, ctx.spec.tolerances.compare_atol_stdev).findings
