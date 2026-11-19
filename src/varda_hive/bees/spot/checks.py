# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee checks: sampled dataset values against the original source, see docs/spot-bee.md."""
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from ...honeycomb import (CheckContext, Dataset, Finding, NotComparable, Source, check, error, info, load_source,
                          skipped, warning)
from ...honeycomb.sampling import sample_points, sample_time_indices
from ...honeycomb.spec import Tolerances


@dataclass
class _Tally:  # pylint: disable=too-many-instance-attributes
    """Comparison counts of one variable over all sampled dates."""
    compared: int = 0
    identical: int = 0
    differ: int = 0
    max_abs_diff: float = 0.0
    dates_differ: list[str] = field(default_factory=list)
    # Of the differing dates: one entry per date, and the totals reported in the variable's ERROR.
    per_date: list[dict[str, Any]] = field(default_factory=list)
    n_nan_mismatch: int = 0
    max_abs_diff_differ: float | None = None
    first_points: list[int] = field(default_factory=list)  # of the first differing date
    members: set[int] = field(default_factory=set)

    @property
    def status(self) -> str:
        if self.differ:
            return "differ"
        return "identical" if self.identical == self.compared else "within_tolerance"

    def error(self, name: str, n_dates: int, n_members: int) -> Finding:
        """The one ERROR of a differing variable: totals, with the dates in `details`."""
        details: dict[str, Any] = {
            "n_differ": self.differ,
            "n_compared": self.compared,
            "n_nan_mismatch": self.n_nan_mismatch,
            "max_abs_diff": self.max_abs_diff_differ,
            "first_points": self.first_points,
            "dates": self.per_date,
        }
        if n_members > 1:
            details["members"] = sorted(self.members)
        return error(
            f"{self.differ}/{self.compared} sampled values differ from source on "
            f"{len(self.dates_differ)}/{n_dates} dates",
            variable=name,
            time=self.dates_differ[0],  # the first; all of them in details["dates"]
            details=details)


@dataclass
class _SpotRun:  # pylint: disable=too-many-instance-attributes
    """State of one comparison of the sample with the source."""
    source: Source
    tolerances: Tolerances
    points: np.ndarray
    shape: tuple[int, int]  # (ensemble members, grid points)
    tallies: dict[str, _Tally] = field(default_factory=dict)
    not_comparable: dict[str, str] = field(default_factory=dict)  # variable -> reason
    unavailable: dict[str, list[str]] = field(default_factory=dict)  # variable -> dates
    failed: dict[str, str] = field(default_factory=dict)  # variable -> first failure, reported once
    not_compared: dict[str, list[str]] = field(default_factory=dict)  # failed variable -> dates skipped after

    def compare_step(self, ds: Dataset, t: int) -> Iterator[Finding]:
        step = ds.step(t)  # (variable, ensemble, grid)
        when = ds.date_str(t)
        for i, name in enumerate(ds.variables):
            if name in self.failed:
                self.not_compared.setdefault(name, []).append(when)
                continue
            if name in self.not_comparable:
                continue
            expected = self._fetch(name, ds.dates[t], when)
            if isinstance(expected, Finding):
                yield expected
            elif expected is not None:
                actual = step[i][:, self.points]
                # Compare in the dataset's precision: a float64 source equal to the stored float32 is identical.
                expected = expected[:, self.points].astype(actual.dtype, copy=False)
                self._compare(name, actual, expected, when, t)

    def _fetch(self, name: str, date: np.datetime64, when: str) -> np.ndarray | Finding | None:
        """The source field as (ensemble, grid), None if the source has none, or the ERROR of a failure."""
        try:
            fetched = self.source.fetch(name, date)  # pylint: disable=assignment-from-no-return  # Protocol stub
            expected = None if fetched is None else _as_field(fetched)
        except NotComparable as exc:
            self.not_comparable[name] = str(exc) or "not comparable"
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed[name] = f"source failed at {when}: {type(exc).__name__}: {exc}"
            return error(self.failed[name], variable=name, time=when)
        if expected is None:
            self.unavailable.setdefault(name, []).append(when)
            return None
        expected = expected.reshape(1, -1) if expected.ndim == 1 else expected
        if expected.shape != self.shape:
            self.failed[name] = f"source field has shape {expected.shape}, dataset has {self.shape}"
            return error(self.failed[name], variable=name, time=when)
        return expected

    def _compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
            self, name: str, actual: np.ndarray, expected: np.ndarray, when: str, t: int) -> None:
        """Add one (variable, date) to its tally, with the details of the date if any value differs."""
        rtol, atol = self.tolerances.source_tolerance(name)
        tally = self.tallies.setdefault(name, _Tally())
        a_nan, e_nan = np.isnan(actual), np.isnan(expected)
        max_diff = _max_abs_diff(actual, expected)
        tally.compared += actual.size
        tally.identical += int(((actual == expected) | (a_nan & e_nan)).sum())
        tally.max_abs_diff = max(tally.max_abs_diff, max_diff or 0.0)
        bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if not bad.any():
            return
        n_bad = int(bad.sum())
        members, cols = np.nonzero(bad)
        tally.differ += n_bad
        tally.dates_differ.append(when)
        tally.n_nan_mismatch += int((a_nan != e_nan).sum())
        if max_diff is not None:
            tally.max_abs_diff_differ = max(tally.max_abs_diff_differ or 0.0, max_diff)
        if not tally.first_points:
            tally.first_points = self.points[np.unique(cols)][:10].tolist()
        tally.members |= set(members.tolist())
        tally.per_date.append({"time": when, "time_index": int(t), "n_differ": n_bad, "max_abs_diff": max_diff})

    def summary(self, n_dates: int) -> Finding:
        """One INFO finding: which variables are identical to the source, within tolerance, differ, or were
        not compared."""
        groups: dict[str, dict[str, dict[str, Any]]] = {"identical": {}, "within_tolerance": {}, "differ": {}}
        for name, tally in self.tallies.items():
            if name not in self.failed:
                entry: dict[str, Any] = {"max_abs_diff": tally.max_abs_diff, "compared": tally.compared}
                if tally.differ:
                    entry |= {"differ": tally.differ, "dates": tally.dates_differ}
                groups[tally.status][name] = entry
        unavailable = {n: d for n, d in self.unavailable.items() if n not in self.tallies and n not in self.failed}
        n_members = self.shape[0]
        message = (f"spot summary: {len(groups['identical'])} identical, "
                   f"{len(groups['within_tolerance'])} within tolerance, {len(groups['differ'])} differ, "
                   f"{len(self.not_comparable)} not comparable, {len(unavailable)} unavailable" +
                   (f", {len(self.failed)} failed" if self.failed else "") +
                   f" ({n_dates} dates × {len(self.points)} points" +
                   (f" × {n_members} members)" if n_members > 1 else ")"))
        return info(message,
                    details={
                        "identical": sorted(groups["identical"]),
                        "within_tolerance": groups["within_tolerance"],
                        "differ": groups["differ"],
                        "not_comparable": self.not_comparable,
                        "unavailable": unavailable,
                        "failed": self.failed,
                        "not_compared_after_failure": self.not_compared,
                        "n_dates": n_dates,
                        "n_points": len(self.points),
                        "n_members": n_members,
                    })


def _as_field(fetched: Any) -> np.ndarray:
    """A plug-in's field as float64, masked values as NaN; raises for anything not numeric."""
    if isinstance(fetched, np.ma.MaskedArray):
        return np.ma.filled(fetched.astype(np.float64), np.nan)
    return np.asarray(fetched, dtype=np.float64)


def _max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float | None:
    """Largest |actual - expected|, ignoring NaN (and equal infinities); inf if only one side is infinite;
    None if no pair is comparable."""
    with np.errstate(invalid="ignore"):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    comparable = diff[~np.isnan(diff)]
    return float(comparable.max()) if comparable.size else None


@check("spot")
def matches_source(ctx: CheckContext) -> Iterator[Finding]:
    """Sampled values agree with the source; summarises identical / within tolerance / differing variables."""
    if "source" not in ctx.cache:
        ctx.cache["source"] = load_source(ctx.spec)
    source = ctx.cache["source"]
    if source is None:
        yield skipped("no `source` in spec")
        return
    ds = ctx.dataset
    bind = getattr(source, "bind", None)
    if callable(bind):
        bind(ds)
    spot = _SpotRun(source=source,
                    tolerances=ctx.spec.tolerances,
                    points=sample_points(ds.data.shape[-1], ctx.spec.sampling),
                    shape=(ds.data.shape[2], ds.data.shape[-1]))
    times = sample_time_indices(len(ds.dates), ctx.spec.sampling, exclude=ds.missing_indices)
    for t in times:
        yield from spot.compare_step(ds, t)
    for name, tally in spot.tallies.items():
        if tally.differ:
            yield tally.error(name, len(times), spot.shape[0])
    for name, dates in spot.unavailable.items():
        if name not in spot.tallies and name not in spot.failed:
            # Never found: a wrong mapping, root or date format looks exactly like this.
            yield error(f"not available from source on any of {len(dates)} sampled dates",
                        variable=name,
                        details={"dates": dates})
        else:
            yield warning(f"not available from source on {len(dates)}/{len(times)} sampled dates",
                          variable=name,
                          details={"dates": dates})
    for name, reason in spot.not_comparable.items():
        yield info(f"not comparable with source: {reason}", variable=name)
    if not any(tally.compared for tally in spot.tallies.values()):
        # Nothing was checked against the source, so the check must not count as passed under --strict.
        yield skipped("no sampled value was compared with the source",
                      details={
                          "n_dates": len(times),
                          "n_points": len(spot.points)
                      })
    yield spot.summary(len(times))
