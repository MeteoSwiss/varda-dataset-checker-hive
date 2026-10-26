"""stats-bee: checks on the values and stored statistics.

All checks work on the stats file content (see honeycomb/statsfile.py),
computed once per run from the sampled dates and shared via `ctx.cache`.
"""
import math

import yaml

from ...honeycomb import CheckContext, check, error, info, warning
from ...honeycomb.statsfile import compare_stats, compute_stats, load_stats


def _stats(ctx: CheckContext) -> dict:
    if "stats" not in ctx.cache:
        reference = load_stats(ctx.spec.reference_stats) if ctx.spec.reference_stats else None
        ctx.cache["reference_stats"] = reference
        # Sample the reference's dates, so matches_reference_stats can compare date by date.
        ctx.cache["stats"] = compute_stats(ctx.dataset, ctx.spec, dates_from=reference)
    return ctx.cache["stats"]


@check("stats")
def statistics_consistent(ctx: CheckContext):
    """Stored mean/stdev/min/max are self-consistent and match the sampled dates within the sampling error.

    Self-consistent: no NaN, min <= mean <= max, stdev 0 exactly when min == max. A sample can
    only bound min/max; with `n_dates: all` everything is compared exactly.
    """
    stats, tol = _stats(ctx), ctx.spec.tolerances
    for name in stats["variables"]:
        stored, rec = stats["stored"][name], stats["recomputed"][name]
        if not stored:
            yield info("skipped: no stored statistics", variable=name)
            continue
        lo, mean, sd, hi = (stored.get(s) for s in ("minimum", "mean", "stdev", "maximum"))
        # Internal consistency of the stored values: exact, no tolerance needed.
        if None in (lo, mean, sd, hi):
            yield error("stored statistics contain NaN", variable=name, details=stored)
            continue
        if not lo <= mean <= hi:
            yield error(f"stored statistics out of order: minimum {lo}, mean {mean}, maximum {hi}",
                        variable=name, details=stored)
        if sd < 0 or (sd == 0) != (lo == hi):
            yield error(f"stored stdev {sd} inconsistent with minimum {lo} and maximum {hi}",
                        variable=name, details=stored)
        if rec.get("mean") is None:
            continue
        # Mean and stdev: the sampling error sets the tolerance (exact for a full pass, se = 0).
        # The per-date means can't spread more than the variable itself, so stdev * fpc / sqrt(n)
        # bounds the standard error of the mean whatever the distribution: beyond that it's an
        # ERROR. The sample's own standard errors are tighter but can be underestimated (e.g. a
        # sample of mostly night-time dates for insolation): beyond those it's a WARNING.
        bound = {"mean": sd * rec["fpc"] / math.sqrt(rec["n_dates"]), "stdev": 0.0 if rec["exact"] else math.inf}
        for stat in ("mean", "stdev"):
            diff = abs(rec[stat] - stored[stat])
            rounding = tol.statistics_rtol * (abs(stored[stat]) + sd)  # + sd: means close to 0
            details = {"stored": stored[stat], "recomputed": rec[stat], "abs_diff": diff,
                       "n_dates": rec["n_dates"], "exact": rec["exact"]}
            message = f"stored {stat} {stored[stat]:.6g} differs from recomputed {rec[stat]:.6g}"
            if diff > tol.z_threshold * bound[stat] + rounding:
                yield error(message, variable=name, details=details | {"se_bound": bound[stat]})
            elif rec[f"se_{stat}"] is not None and diff > tol.z_threshold * rec[f"se_{stat}"] + rounding:
                yield warning(message + " (beyond the sampling error estimated from the sampled dates)",
                              variable=name, details=details | {"z": diff / rec[f"se_{stat}"]})
        # Minimum and maximum: a sample can only bound them, unless every date was read.
        slack = tol.statistics_rtol * max(abs(lo), abs(hi))
        if rec["minimum"] < lo - slack or (rec["exact"] and rec["minimum"] > lo + slack):
            yield error(f"stored minimum {lo:.6g} but the sampled dates reach {rec['minimum']:.6g}",
                        variable=name, details={"stored": lo, "recomputed": rec["minimum"], "exact": rec["exact"]})
        if rec["maximum"] > hi + slack or (rec["exact"] and rec["maximum"] < hi - slack):
            yield error(f"stored maximum {hi:.6g} but the sampled dates reach {rec['maximum']:.6g}",
                        variable=name, details={"stored": hi, "recomputed": rec["maximum"], "exact": rec["exact"]})
    n = next(iter(stats["recomputed"].values()), {}).get("n_dates", 0)
    yield info(f"statistics recomputed from {n} dates", details=_resolution(stats, tol.z_threshold))


def _resolution(stats: dict, z: float) -> dict:
    """Smallest difference to the stored mean the test can detect, per variable: as an ERROR
    (distribution-free bound) and as a WARNING (sample standard error)."""
    out = {}
    for name, rec in stats["recomputed"].items():
        sd = stats["stored"][name].get("stdev")
        if rec.get("mean") is None or sd is None:
            continue
        out[name] = {"error_mean": z * sd * rec["fpc"] / math.sqrt(rec["n_dates"]),
                     "warning_mean": z * rec["se_mean"] if rec["se_mean"] is not None else None}
    return {"min_detectable": out}


@check("stats")
def physical_ranges(ctx: CheckContext):
    """Sampled values and stored min/max lie inside the ranges of the spec's `ranges_file`.

    Skipped without a `ranges_file`; variables not in it are listed as INFO.
    """
    if ctx.spec.ranges_file is None:
        yield info("skipped: no `ranges_file` in spec")
        return
    ranges = yaml.safe_load(ctx.spec.ranges_file.read_text()) or {}
    stats = _stats(ctx)
    unlisted = [v for v in stats["variables"] if v not in ranges]
    if unlisted:
        yield info(f"{len(unlisted)} variables without a physical range", details={"variables": unlisted})
    for name, bounds in ranges.items():
        if name not in stats["variables"]:
            continue
        lo, hi, units = bounds.get("min", -math.inf), bounds.get("max", math.inf), bounds.get("units", "")
        per_date = stats["per_date"][name]
        for date, vmin, vmax in zip(stats["dates"], per_date["minimum"], per_date["maximum"]):
            if vmin is not None and (vmin < lo or vmax > hi):
                yield error(f"values [{vmin:.6g}, {vmax:.6g}] outside physical range [{lo}, {hi}] {units}",
                            variable=name, time=date, details={"minimum": vmin, "maximum": vmax})
        stored = stats["stored"][name]
        if stored.get("minimum") is not None and (stored["minimum"] < lo or stored["maximum"] > hi):
            yield error(f"stored minimum/maximum [{stored['minimum']:.6g}, {stored['maximum']:.6g}] "
                        f"outside physical range [{lo}, {hi}] {units}", variable=name)


@check("stats")
def no_nans(ctx: CheckContext):
    """No NaNs on the sampled dates, unless the dataset sets `allow_nans`."""
    if ctx.dataset.attrs.get("allow_nans", False):
        yield info("skipped: dataset allows NaNs")
        return
    stats = _stats(ctx)
    for name in stats["variables"]:
        for t, date, n in zip(stats["time_index"], stats["dates"], stats["per_date"][name]["n_nan"]):
            if n:
                yield error(f"{n} NaN values", variable=name, time=date, details={"time_index": t, "n_nan": n})


@check("stats")
def constant_fields(ctx: CheckContext):
    """Fields in `constant_fields` never change; no other field is constant in time or uniform in space.

    `constant_fields` comes from the dataset, or from the spec if the dataset has none. A field in
    it that changes is an ERROR; any other field identical on all sampled dates, or uniform in
    space on one date, is a WARNING. Computed forcings are exempt from the uniform-in-space test.
    """
    stats = _stats(ctx)
    listed = ctx.dataset.attrs.get("constant_fields")
    if listed is None:
        listed = ctx.spec.constant_fields or []
    # Forcings like cos_julian_day, or insolation at night, are legitimately uniform in space.
    forcings = {v for v, m in ctx.dataset.attrs.get("variables_metadata", {}).items() if m.get("computed_forcing")}
    for name in stats["variables"]:
        per_date = stats["per_date"][name]
        summaries = list(zip(per_date["mean"], per_date["stdev"], per_date["minimum"], per_date["maximum"]))
        if len(summaries) < 2:
            continue
        changed = [d for d, s in zip(stats["dates"], summaries) if s != summaries[0]]
        if name in listed and changed:
            yield error(f"listed as constant field but changes at {len(changed)}/{len(summaries) - 1} dates",
                        variable=name, details={"dates": changed[:10], "first_date": stats["dates"][0]})
        elif name not in listed and not changed:
            yield warning(f"identical on all {len(summaries)} sampled dates but not in constant_fields",
                          variable=name)
        # A whole field set to one value (e.g. zeroed), unless the variable is constant anyway.
        stored = stats["stored"][name]
        if name not in forcings and stored.get("minimum") != stored.get("maximum"):
            for date, sd, value in zip(stats["dates"], per_date["stdev"], per_date["mean"]):
                if sd == 0:
                    yield warning(f"spatially constant field (all values {value})", variable=name, time=date)


@check("stats")
def matches_reference_stats(ctx: CheckContext):
    """Stored statistics and per-date values match the spec's `reference_stats`, on its dates.

    Skipped without `reference_stats`. Same comparison as `stats-bee compare`.
    """
    if ctx.spec.reference_stats is None:
        yield info("skipped: no `reference_stats` in spec")
        return
    stats = _stats(ctx)
    yield from compare_stats(stats, ctx.cache["reference_stats"], ctx.spec.tolerances.compare_atol_stdev).findings
