# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""meta-bee: metadata and structure checks. Cheap: reads the group attributes (`.zattrs` or
`zarr.json`) and the small arrays (dates, coordinates, statistics), never `data`.

Three groups:
- internal consistency (no spec needed): what `anemoi-datasets validate` checks, natively and with
  precise findings, plus what it misses (frequency, date attributes, missing_dates, ...)
- against the spec: expected variables, dates, grid, flags and per-variable metadata
- against a reference (`matches_reference`): the same comparison as `hive compare`
"""
from dataclasses import replace
import math
from typing import Any, Iterator

import numpy as np

from ...honeycomb import CheckContext, Finding, bee_dump, check, dumps, error, info, skipped, warning
from ...honeycomb.dataset import STATISTICS, Dataset, format_frequency, parse_date, parse_frequency
from ...honeycomb.files import provides
from .metafile import FORMAT, compare_meta, fingerprint, load_reference

MAX_LISTED = 10  # findings of one kind before the rest are summarised in one

# name, valid range, and the bound under which all values suggest radians
_COORDINATES = (("latitudes", -90, 90, math.pi), ("longitudes", -180, 360, 2 * math.pi))


def _step(ds: Dataset) -> np.timedelta64 | None:
    """The most common positive step between dates, None for fewer than 2 dates."""
    diffs = np.diff(ds.dates)
    diffs = diffs[diffs > np.timedelta64(0, "s")]
    if diffs.size == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    return values[np.argmax(counts)]


def _capped(findings: list[Finding], what: str) -> Iterator[Finding]:
    yield from findings[:MAX_LISTED]
    if len(findings) > MAX_LISTED:
        rest = len(findings) - MAX_LISTED
        yield replace(findings[0], message=f"... and {rest} more {what}", variable=None, time=None, details={})


# --- internal consistency --------------------------------------------------------------------


@check("meta")
def dates_valid(ctx: CheckContext) -> Iterator[Finding]:
    """Dates are set, strictly increasing and evenly spaced."""
    dates = ctx.dataset.dates
    if dates.size == 0:
        yield error("the dataset has no dates")
        return
    if (nat := np.flatnonzero(np.isnat(dates))).size:
        yield error(f"{nat.size} dates are not set (NaT)", details={"indices": nat[:MAX_LISTED].tolist()})
        return
    diffs = np.diff(dates)
    bad = []
    for i in np.flatnonzero(diffs <= np.timedelta64(0, "s")):
        kind = "duplicate date" if diffs[i] == np.timedelta64(0, "s") else "dates not increasing"
        bad.append(
            error(f"{kind}: index {i + 1} ({dates[i + 1]}) after index {i} ({dates[i]})",
                  time=str(dates[i + 1]),
                  details={"index": int(i + 1)}))
    if bad:
        yield from _capped(bad, "unordered dates")
        return
    steps, counts = np.unique(diffs, return_counts=True)
    if len(steps) > 1:
        main = steps[np.argmax(counts)]
        gaps = [
            error(f"step of {format_frequency(diffs[i])} instead of {format_frequency(main)} after {dates[i]}",
                  time=str(dates[i + 1]),
                  details={"index": int(i + 1)}) for i in np.flatnonzero(diffs != main)
        ]
        yield from _capped(gaps, "irregular steps")


@check("meta")
def frequency_consistent(ctx: CheckContext) -> Iterator[Finding]:
    """The `frequency` attribute is the step between the dates."""
    value = ctx.dataset.attrs.get("frequency")
    if value is None:
        yield warning("no `frequency` attribute")
        return
    step = _step(ctx.dataset)
    if step is None:
        yield skipped("fewer than 2 dates")
        return
    try:
        frequency = parse_frequency(value)
    except ValueError as e:
        yield error(f"`frequency` attribute: {e}")
        return
    if frequency != step:
        yield error(f"`frequency` attribute is {value} but the dates are {format_frequency(step)} apart",
                    details={
                        "attribute": value,
                        "step": format_frequency(step)
                    })


@check("meta")
def date_attrs_consistent(ctx: CheckContext) -> Iterator[Finding]:
    """The `start_date`/`end_date` attributes are the first and last dates."""
    ds = ctx.dataset
    if ds.dates.size == 0:
        yield skipped("no dates")
        return
    for key, actual in (("start_date", ds.dates[0]), ("end_date", ds.dates[-1])):
        value = ds.attrs.get(key)
        if value is None:
            yield warning(f"no `{key}` attribute")
        elif parse_date(value) != actual:
            yield error(
                f"`{key}` attribute is {value} but the dates {'start' if key == 'start_date' else 'end'} "
                f"at {actual}",
                details={
                    "attribute": value,
                    "dates": str(actual)
                })


@check("meta")
def missing_dates_valid(ctx: CheckContext) -> Iterator[Finding]:
    """Every entry of `missing_dates` is one of the dates; lists them, as their data is excluded."""
    ds = ctx.dataset
    missing = ds.attrs.get("missing_dates") or []
    dates = set(ds.dates.tolist())
    for value in missing:
        date = parse_date(value)
        if date.astype(object) not in dates:
            yield error(f"`missing_dates` lists {value}, which is not one of the dates", time=str(date))
        else:
            yield info("date listed as missing: excluded from the statistics and from training", time=str(date))


@check("meta")
def shape_consistent(ctx: CheckContext) -> Iterator[Finding]:
    """`shape`, `variables`, `field_shape`, dates, coordinates and statistics agree with `data`."""
    ds = ctx.dataset
    shape = tuple(ds.data.shape)
    if len(shape) != 4:
        yield error(f"`data` has {len(shape)} dimensions, expected 4 (time, variable, ensemble, grid)")
        return
    n_time, n_var, _, n_grid = shape
    if (attr := ds.attrs.get("shape")) is not None and tuple(attr) != shape:
        yield error(f"`shape` attribute {list(attr)} differs from the `data` array {list(shape)}")
    if len(ds.variables) != n_var:
        yield error(f"{len(ds.variables)} names in `variables` but {n_var} variables in `data`")
    if len(set(ds.variables)) != len(ds.variables):
        dups = sorted({v for v in ds.variables if ds.variables.count(v) > 1})
        yield error(f"duplicate names in `variables`: {dups}")
    if len(ds.dates) != n_time:
        yield error(f"{len(ds.dates)} dates but {n_time} time steps in `data`")
    if (field_shape := ds.attrs.get("field_shape")) is not None and math.prod(field_shape) != n_grid:
        yield error(f"`field_shape` {list(field_shape)} has {math.prod(field_shape)} points, `data` has {n_grid}")
    for name, coords in (("latitudes", ds.latitudes), ("longitudes", ds.longitudes)):
        if coords is None:
            yield error(f"no `{name}` array")
        elif coords.shape != (n_grid, ):
            yield error(f"`{name}` has shape {list(coords.shape)}, expected [{n_grid}]")
    for name in STATISTICS:
        if name in ds.arrays and ds.array(name).shape != (n_var, ):
            yield error(f"statistics `{name}` has shape {list(ds.array(name).shape)}, expected [{n_var}]")


@check("meta")
def coordinates_valid(ctx: CheckContext) -> Iterator[Finding]:
    """Latitudes and longitudes are finite, in range and in degrees."""
    ds = ctx.dataset
    for name, lo, hi, rad in _COORDINATES:
        coords = getattr(ds, name)
        if coords is None:
            yield skipped(f"no `{name}` array")
            continue
        if not np.all(np.isfinite(coords)):
            yield error(f"{int(np.sum(~np.isfinite(coords)))} `{name}` are not finite")
            continue
        if coords.min() < lo or coords.max() > hi:
            yield error(f"`{name}` outside [{lo}, {hi}]: {coords.min():.6g}..{coords.max():.6g}")
        elif np.all(np.abs(coords) <= rad) and np.ptp(coords) > 0:
            yield warning(f"all `{name}` within [-{rad:.4g}, {rad:.4g}]: are they in radians?")


@check("meta")
def variables_metadata_consistent(ctx: CheckContext) -> Iterator[Finding]:
    """`variables_metadata` and `constant_fields` name only variables of the dataset."""
    ds = ctx.dataset
    names = set(ds.variables)
    metadata = ds.attrs.get("variables_metadata")
    if metadata is None:
        yield info("no `variables_metadata` attribute")
    else:
        for name in sorted(set(metadata) - names):
            yield error("`variables_metadata` has an entry for a variable not in the dataset", variable=name)
        lacking = [warning("no `variables_metadata` entry", variable=v) for v in ds.variables if v not in metadata]
        yield from _capped(lacking, "variables without `variables_metadata`")
    constant = set(ds.attrs.get("constant_fields") or [])
    for name in sorted(constant - names):
        yield error("`constant_fields` lists a variable not in the dataset", variable=name)
    # anemoi flags constant fields in both places; they must agree. Once any variable carries the
    # flag, a variable without it counts as not constant (anemoi sets it only where it applies).
    flags = {
        v: m["constant_in_time"]
        for v, m in (metadata or {}).items() if isinstance(m, dict) and "constant_in_time" in m
    }
    if flags and "constant_fields" in ds.attrs:
        in_time = {v for v, flag in flags.items() if flag}
        for name in sorted((constant & names) - in_time):
            said = "says not" if name in flags else "does not flag it as"
            yield error(f"in `constant_fields`, but `variables_metadata` {said} constant_in_time", variable=name)
        for name in sorted(in_time - constant):
            yield error("constant_in_time in `variables_metadata`, but not in `constant_fields`", variable=name)


# --- against the spec ------------------------------------------------------------------------


@check("meta")
def variables_match_spec(ctx: CheckContext) -> Iterator[Finding]:
    """The dataset has exactly the variables listed in the spec."""
    expected = ctx.spec.variable_names()
    if expected is None:
        yield skipped("no `variables` in spec")
        return
    actual = ctx.dataset.variables
    for name in sorted(set(expected) - set(actual)):
        yield error("variable missing from dataset", variable=name)
    for name in sorted(set(actual) - set(expected)):
        yield error("variable not in spec", variable=name)


@check("meta")
def dates_match_spec(ctx: CheckContext) -> Iterator[Finding]:
    """First date, last date and step are the spec's `start_date`, `end_date` and `frequency`."""
    spec, ds = ctx.spec, ctx.dataset
    if spec.start_date is None and spec.end_date is None and spec.frequency is None:
        yield skipped("no `start_date`, `end_date` or `frequency` in spec")
        return
    if ds.dates.size == 0:
        yield error("the dataset has no dates")
        return
    for key, expected, actual in (("start_date", spec.start_date, ds.dates[0]), ("end_date", spec.end_date,
                                                                                 ds.dates[-1])):
        if expected is not None and parse_date(expected.isoformat()) != actual:
            yield error(f"dates {'start' if key == 'start_date' else 'end'} at {actual}, the spec's `{key}` is "
                        f"{expected.isoformat()}")
    if spec.frequency is not None:
        step = _step(ds)
        if step is not None and step != parse_frequency(spec.frequency):
            yield error(f"dates are {format_frequency(step)} apart, the spec's `frequency` is {spec.frequency}")


@check("meta")
def grid_matches_spec(ctx: CheckContext) -> Iterator[Finding]:
    """Number of grid points, `resolution` and the lat/lon extent match the spec."""
    spec, ds = ctx.spec, ctx.dataset
    if spec.grid_points is None and spec.resolution is None and spec.bbox is None:
        yield skipped("no `grid_points`, `resolution` or `bbox` in spec")
        return
    if spec.grid_points is not None and ds.data.shape[-1] != spec.grid_points:
        yield error(f"{ds.data.shape[-1]} grid points, the spec expects {spec.grid_points}")
    if spec.resolution is not None and str(ds.attrs.get("resolution")) != spec.resolution:
        yield error(f"`resolution` attribute is {ds.attrs.get('resolution')!r}, the spec expects "
                    f"{spec.resolution!r}")
    if (box := spec.bbox) is not None:
        if ds.latitudes is None or ds.longitudes is None:
            yield error("no coordinates to compare with the spec's `bbox`")
            return
        bounds = (("lat_min", ds.latitudes.min(), box.lat_min), ("lat_max", ds.latitudes.max(), box.lat_max),
                  ("lon_min", ds.longitudes.min(), box.lon_min), ("lon_max", ds.longitudes.max(), box.lon_max))
        for name, actual, expected in bounds:
            if not abs(float(actual) - expected) <= box.tolerance:
                yield error(f"{name} is {float(actual):.6g}, the spec's `bbox` expects {expected:.6g} "
                            f"(tolerance {box.tolerance:g})")


@check("meta")
def flags_match_spec(ctx: CheckContext) -> Iterator[Finding]:
    """`allow_nans` and `missing_dates` are the spec's."""
    spec, attrs = ctx.spec, ctx.dataset.attrs
    expected_nans = spec.nan_variables()
    if expected_nans is None and spec.missing_dates is None:
        yield skipped("no `allow_nans` or `missing_dates` in spec")
        return
    if expected_nans is not None:
        yield from _compare_allow_nans(attrs.get("allow_nans", False), expected_nans)
    if spec.missing_dates is not None:
        actual = {parse_date(d) for d in attrs.get("missing_dates") or []}
        expected = {parse_date(d.isoformat()) for d in spec.missing_dates}
        for d in sorted(actual - expected):
            yield error("date listed in `missing_dates` but not in the spec's", time=str(d))
        for d in sorted(expected - actual):
            yield error("date in the spec's `missing_dates` but not listed by the dataset", time=str(d))


def _nan_set(value: Any) -> bool | set[str]:
    """`allow_nans` as True (everywhere) or the set of variables, where false is the empty set."""
    if isinstance(value, (list, set, tuple)):
        return {str(v) for v in value}
    return True if value is True else set()


def _compare_allow_nans(actual: Any, expected: bool | set[str]) -> Iterator[Finding]:
    """The dataset's `allow_nans` (a bool or a list of variables) against the spec's."""
    have, want = _nan_set(actual), _nan_set(expected)
    if isinstance(have, set) and isinstance(want, set):
        for name in sorted(have - want):
            yield error("NaNs allowed by the dataset's `allow_nans` but not by the spec", variable=name)
        for name in sorted(want - have):
            yield error("NaNs allowed by the spec but not listed in the dataset's `allow_nans`", variable=name)
    elif have != want:
        shown = sorted(want) if isinstance(want, set) else want
        hint = (" (list the variables in the spec's `allow_nans`, or set `allow_nans: true` on them)" if isinstance(
            have, set) else "")
        yield error(f"`allow_nans` is {actual!r}, the spec expects {shown!r}{hint}")


def _lookup(entry: dict, key: str) -> tuple[bool, Any]:
    """`key` in a variables_metadata entry, at the top or in its `mars` section."""
    for section in (entry, entry.get("mars") or {}):
        if isinstance(section, dict) and key in section:
            return True, section[key]
    return False, None


@check("meta")
def variables_metadata_match_spec(ctx: CheckContext) -> Iterator[Finding]:
    """Per-variable metadata (mars param, level, units, ...) matches the spec's `variables` entries."""
    spec, ds = ctx.spec, ctx.dataset
    expected, units = spec.expected_metadata(), spec.units()
    if not expected and not units:
        yield skipped("no per-variable metadata or units in the spec's `variables`")
        return
    metadata = ds.attrs.get("variables_metadata") or {}
    for name, keys in expected.items():
        if name not in metadata:
            yield error("no `variables_metadata` entry to compare with the spec", variable=name)
            continue
        for key, value in keys.items():
            found, actual = _lookup(metadata[name], key)
            if not found:
                yield error(f"`{key}` missing from `variables_metadata`, the spec expects {value!r}", variable=name)
            elif str(actual) != str(value):
                yield error(f"`{key}` is {actual!r}, the spec expects {value!r}", variable=name)
    without = []
    for name, unit in units.items():
        if name not in metadata:
            continue
        found, actual = _lookup(metadata[name], "units")
        if not found:
            without.append(name)
        elif str(actual) != unit:
            yield error(f"units are {actual!r}, the spec says {unit!r}", variable=name)
    if without:
        yield warning(f"the dataset stores no units for {len(without)} variables with units in the spec",
                      details={"variables": without})


# --- against a reference ---------------------------------------------------------------------


@dumps("meta")
def meta_dump(ctx: CheckContext) -> dict[str, Any]:
    """meta-bee's section of `ref.yaml`: the meta file (attributes, and shape, dtype and hash of every
    array but `data`)."""
    return fingerprint(ctx.dataset)


@check("meta")
def matches_reference(ctx: CheckContext) -> Iterator[Finding]:
    """Metadata and structure match the reference (`--reference` or the spec's `reference`): a ref
    file, a meta file or a zarr. Compares the dump with the reference's meta section, as `hive compare`."""
    if (reason := ctx.reference_missing()) is not None:
        yield skipped(reason)
        return
    path = ctx.reference_path
    assert path is not None
    if not provides(path, "meta", FORMAT):
        yield skipped(f"the reference {path} has no meta section (a stats file?)")
        return
    comparison = compare_meta(bee_dump(ctx, "meta"), load_reference(path), ctx.spec.reference_policy)
    yield from comparison.findings
