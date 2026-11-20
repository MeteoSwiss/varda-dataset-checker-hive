# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Meta files: a small fingerprint of a dataset's metadata and structure, meta-bee's dump: the
`meta` section of `ref.yaml` (`--out`), compared by the `matches_reference` check and `hive compare`.

A meta file holds:
- `attrs`: the group attributes (`.zattrs` in zarr format 2, `zarr.json` in format 3)
- `arrays`: for every array, its shape and dtype, and, except for `data`, a sha256 of its values
  plus a short human-readable summary (date range, coordinate extent, ...)

Two fingerprints compare item by item, on dotted paths (`variables_metadata.T_850.mars.levelist`,
`arrays.dates.values`). Each item is identical, differs, added (only in the dataset), missing (only in
the reference) or ignored. The severity of a difference comes from the policy: the spec's
`reference_policy` first, then `DEFAULT_POLICY`. Run-to-run churn (uuid, timestamps, provenance,
recipe) is ignored by default.

The fingerprint does not depend on the zarr format: the same content written as format 2 or 3 gives
the same meta file (attrs, shapes, numpy dtypes and value hashes), so a dataset rewritten in format 3
still compares clean against a reference taken from format 2.

A reference can be a meta file or a zarr: `load_reference` fingerprints a zarr on the fly, so both
compare the same way.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ... import __version__
from ...honeycomb import Dataset, Finding, ReferencePolicy, Severity, info
from ...honeycomb.dataset import STATISTICS, format_frequency, is_zarr
from ...honeycomb.files import read_file, save_file

FORMAT = "varda-hive-meta/1"
IDENTICAL, DIFFERS, ADDED, MISSING, IGNORED = "identical", "differs", "added", "missing", "ignored"
STATUSES = (IDENTICAL, DIFFERS, ADDED, MISSING, IGNORED)

# Arrays holding statistics (`statistics_*` too): their values are not bit-reproducible between builds.
STATISTICS_ARRAYS = (*STATISTICS, "sums", "squares", "count", "has_nans")

# Severity of a difference, None to ignore it. First match wins; paths not matched are WARNING. Taken
# from the IGNORE lists of `anemoi-datasets compare`, plus what a rebuild or a library update changes
# without changing the data.
DEFAULT_POLICY: list[tuple[str, Severity | None]] = [
    # churn: differs on every build
    *((p, None) for p in (
        "uuid",
        "latest_write_timestamp",
        "provenance_load",
        "provenance_load.*",
        "total_size",
        "total_number_of_files",
        "version",
        "history",
        "history.*",
        "recipe",
        "recipe.*",
        "order_by",
        "order_by.*",
        "chunks",
        "data_request",
        "data_request.*",
    )),
    # informative text
    *((p, Severity.INFO) for p in ("description", "attribution", "licence", "license", "title")),
    # structure
    *((p, Severity.ERROR) for p in (
        "variables",
        "shape",
        "dtype",
        "field_shape",
        "resolution",
        "frequency",
        "start_date",
        "end_date",
        "last_date",
        "ensemble_dimension",
        "flatten_grid",
        "arrays.*.shape",
        "arrays.*.dtype",
        "arrays.dates.values",
        "arrays.latitudes.values",
        "arrays.longitudes.values",
    )),
    # statistics values: rebuilds are not bit-reproducible (stats-bee compares them with tolerances)
    *((f"arrays.{p}.values", Severity.INFO) for p in STATISTICS_ARRAYS),
    ("arrays.statistics_*.values", Severity.INFO),
]
_SPEC_ORDER = (("ignore", None), ("error", Severity.ERROR), ("warning", Severity.WARNING), ("info", Severity.INFO))

# --- the fingerprint -------------------------------------------------------------------------


def _summary(values: np.ndarray) -> str:
    """A short description of an array's values, shown when they differ."""
    if values.size == 0:
        return "empty"
    if np.issubdtype(values.dtype, np.datetime64):
        dates = values.astype("datetime64[s]")
        steps = np.unique(np.diff(dates))
        step = f"step {format_frequency(steps[0])}" if len(steps) == 1 else f"{len(steps)} different steps"
        return f"{len(dates)} dates {dates[0]}..{dates[-1]}, {step}"
    if np.issubdtype(values.dtype, np.number) or values.dtype == bool:
        v = values.astype("f8")
        finite = v[np.isfinite(v)]
        extent = f"{finite.min():.6g}..{finite.max():.6g}" if finite.size else "no finite values"
        return f"{values.size} values {extent}"
    return f"{values.size} values"


def fingerprint(ds: Dataset) -> dict[str, Any]:
    """The meta file content of a dataset. Reads the group attributes and every array except `data`."""
    arrays: dict[str, dict[str, Any]] = {}
    for name in ds.arrays:
        array = ds.array(name)
        entry: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if name != "data":
            values = np.ascontiguousarray(ds.values(name))
            entry["values"] = hashlib.sha256(values.tobytes()).hexdigest()
            entry["summary"] = _summary(values)
        arrays[name] = entry
    return {
        "format": FORMAT,
        "dataset": {
            "name": ds.name,
            "path": str(ds.path.resolve()),
        },
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "varda_hive": __version__,
        },
        "attrs": ds.attrs,
        "arrays": arrays,
    }


def save_meta(meta: dict, path: str | Path | None) -> None:
    save_file(meta, path)


def load_meta(path: str | Path) -> dict:
    return read_file(path, FORMAT, "meta file", "hive ... --out", section="meta")


def load_reference(path: str | Path) -> dict:
    """A meta file, or the fingerprint of a zarr dataset."""
    if not Path(path).exists():
        raise FileNotFoundError(f"reference not found: {path}")
    return fingerprint(Dataset(path)) if is_zarr(path) else load_meta(path)


# --- comparing two fingerprints --------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    path: str
    status: str
    reference: Any = None
    actual: Any = None
    severity: Severity | None = None  # None for identical and ignored items
    note: str = ""

    @property
    def group(self) -> str:
        return group_of(self.path)

    @property
    def variable(self) -> str | None:
        parts = self.path.split(".")
        return parts[1] if parts[0] == "variables_metadata" and len(parts) > 1 else None


@dataclass
class MetaComparison:
    items: list[Item] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(i.status == s for i in self.items) for s in STATUSES}

    @property
    def findings(self) -> list[Finding]:
        out = [_finding(i) for i in self.items if i.severity is not None]
        counts = ", ".join(f"{n} {s}" for s, n in self.counts.items() if n)
        out.append(info(f"compared with the reference: {counts}", details=self.counts))
        return out


def group_of(path: str) -> str:
    """The section of the compare table an item belongs to."""
    head = path.split(".")[0]
    array = path.split(".")[1] if head == "arrays" and "." in path else ""
    if head in ("variables", "variables_metadata", "constant_fields") or array == "data":
        return "variables"
    if head in ("start_date", "end_date", "frequency", "missing_dates", "last_date") or array == "dates":
        return "dates"
    if head in ("field_shape", "resolution", "proj_string", "flatten_grid", "remapping") or array in ("latitudes",
                                                                                                      "longitudes"):
        return "grid"
    if head.startswith("statistics_") or array in STATISTICS_ARRAYS or array.startswith("statistics_"):
        return "statistics"
    return "arrays" if head == "arrays" else "attrs"


def severity_of(path: str, policy: ReferencePolicy | None = None) -> Severity | None:
    """Severity of a difference at `path`, None if it is ignored."""
    for key, severity in _SPEC_ORDER:
        if policy is not None and any(fnmatch(path, p) for p in getattr(policy, key)):
            return severity
    for pattern, severity in DEFAULT_POLICY:
        if fnmatch(path, pattern):
            return severity
    return Severity.WARNING


def compare_meta(actual: dict, reference: dict, policy: ReferencePolicy | None = None) -> MetaComparison:
    comparison = MetaComparison()
    raw: list[tuple[str, str, Any, Any]] = []
    raw.extend(_walk(actual.get("attrs", {}), reference.get("attrs", {}), ()))
    raw.extend(_compare_arrays(actual.get("arrays", {}), reference.get("arrays", {})))
    counts = (_counts(reference), _counts(actual))
    for path, status, ref, act in raw:
        if status == IDENTICAL:
            comparison.items.append(Item(path, status, ref, act))
            continue
        severity = severity_of(path, policy)
        if severity is None:
            comparison.items.append(Item(path, IGNORED, ref, act))
            continue
        note = _note(ref, act, status)
        if follows := _follows(path, ref, act, status, counts):
            severity, note = Severity.INFO, f"{shorten(ref)} -> {shorten(act)}, follows from {follows}"
        comparison.items.append(Item(path, status, ref, act, severity, note))
    return comparison


Counts = tuple[int | None, int | None]


def _counts(meta: dict) -> Counts:
    """Number of variables and of dates, the two sizes a shape change usually comes from."""
    variables = meta.get("attrs", {}).get("variables")
    dates = meta.get("arrays", {}).get("dates", {}).get("shape")
    return (len(variables) if isinstance(variables, list) else None, dates[0] if dates else None)


def _follows(path: str, reference: Any, actual: Any, status: str, counts: tuple[Counts, Counts]) -> str:
    """For a shape that differs only by a change of the number of variables or dates, which are
    reported themselves (`variables`, `arrays.dates.shape`): what it follows from, else ""."""
    if status != DIFFERS or path in ("variables", "arrays.dates.shape"):
        return ""
    if not (path == "shape" or (path.startswith("arrays.") and path.endswith(".shape"))):
        return ""
    if not isinstance(reference, list) or not isinstance(actual, list) or len(reference) != len(actual):
        return ""
    (ref_vars, ref_dates), (act_vars, act_dates) = counts
    causes = {(ref_vars, act_vars): "the number of variables", (ref_dates, act_dates): "the number of dates"}
    found = set()
    for pair in zip(reference, actual):
        if pair[0] == pair[1]:
            continue
        if pair not in causes:
            return ""
        found.add(causes[pair])
    return " and ".join(sorted(found))


def _walk(actual: Any, reference: Any, path: tuple[str, ...]) -> Iterator[tuple[str, str, Any, Any]]:
    """Leaves of two attribute trees: dicts recurse, everything else (lists too) is one item."""
    name = ".".join(path)
    if isinstance(actual, dict) and isinstance(reference, dict):
        for key in sorted(set(actual) | set(reference), key=str):
            sub = (*path, str(key))
            if key not in reference:
                yield ".".join(sub), ADDED, None, actual[key]
            elif key not in actual:
                yield ".".join(sub), MISSING, reference[key], None
            else:
                yield from _walk(actual[key], reference[key], sub)
        return
    yield name, IDENTICAL if _equal(actual, reference) else DIFFERS, reference, actual


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return type(a) is type(b) and a == b  # noqa: E721  # 1 != 1.0 != True in metadata


def _compare_arrays(actual: dict, reference: dict) -> Iterator[tuple[str, str, Any, Any]]:
    for name in sorted(set(actual) | set(reference)):
        path = f"arrays.{name}"
        if name not in reference:
            yield path, ADDED, None, _describe_array(actual[name])
            continue
        if name not in actual:
            yield path, MISSING, _describe_array(reference[name]), None
            continue
        a, r = actual[name], reference[name]
        for key in ("shape", "dtype"):
            yield f"{path}.{key}", IDENTICAL if a.get(key) == r.get(key) else DIFFERS, r.get(key), a.get(key)
        if "values" in a and "values" in r and a["shape"] == r["shape"]:
            same = a["values"] == r["values"]
            ref_v, act_v = r.get("summary", r["values"][:12]), a.get("summary", a["values"][:12])
            if not same and ref_v == act_v:  # same summary, other values: show the hashes
                ref_v, act_v = f"{ref_v} (sha256 {r['values'][:8]})", f"{act_v} (sha256 {a['values'][:8]})"
            yield f"{path}.values", IDENTICAL if same else DIFFERS, ref_v, act_v


def _describe_array(entry: dict) -> str:
    return f"{entry.get('dtype')} {tuple(entry.get('shape', ()))}"


def _note(reference: Any, actual: Any, status: str) -> str:
    """What changed in a list, e.g. the variables: added and missing elements, or only the order."""
    if status != DIFFERS or not _str_list(reference) or not _str_list(actual):
        return ""
    ref_set, act_set = set(reference), set(actual)
    parts = []
    if added := [x for x in actual if x not in ref_set]:
        parts.append(f"added {shorten(added)}")
    if missing := [x for x in reference if x not in act_set]:
        parts.append(f"missing {shorten(missing)}")
    if not parts:
        if len(reference) != len(actual):
            parts.append(f"same elements, length {len(reference)} -> {len(actual)} (duplicates)")
        else:
            moved = [r for r, a in zip(reference, actual) if r != a]
            parts.append(f"same elements, order differs: {shorten(moved)}")
    return "; ".join(parts)


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def shorten(value: Any, width: int = 60) -> str:
    """`value` as text, cut to `width` characters."""
    text = str(value)
    return text if len(text) <= width else text[:width - 3] + "..."


def _finding(item: Item) -> Finding:
    assert item.severity is not None
    if item.status == ADDED:
        message = f"{item.path}: not in the reference ({shorten(item.actual)})"
    elif item.status == MISSING:
        message = f"{item.path}: missing, the reference has {shorten(item.reference)}"
    else:
        change = item.note or f"{shorten(item.reference, 100)} -> {shorten(item.actual, 100)}"
        message = f"{item.path} differs from the reference: {change}"
    details = {"path": item.path, "status": item.status, "reference": item.reference, "actual": item.actual}
    return Finding(item.severity, message, variable=item.variable, details=details)
