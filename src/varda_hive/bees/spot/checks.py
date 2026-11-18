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
class _Tally:
    """Comparison counts of one variable over all sampled dates."""
    compared: int = 0
    identical: int = 0
    differ: int = 0
    max_abs_diff: float = 0.0
    dates_differ: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.differ:
            return "differ"
        return "identical" if self.identical == self.compared else "within_tolerance"


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

    def compare_step(self, ds: Dataset, t: int) -> Iterator[Finding]:
        step = ds.step(t)  # (variable, ensemble, grid)
        when = ds.date_str(t)
        for i, name in enumerate(ds.variables):
            if name in self.not_comparable or name in self.failed:
                continue
            expected = self._fetch(name, ds.dates[t], when)
            if isinstance(expected, Finding):
                yield expected
            elif expected is not None:
                actual = step[i][:, self.points]
                # Compare in the dataset's precision: a float64 source equal to the stored float32 is identical.
                expected = expected[:, self.points].astype(actual.dtype, copy=False)
                finding = self._compare(name, actual, expected, when)
                if finding is not None:
                    yield error(finding[0], variable=name, time=when, details=finding[1] | {"time_index": int(t)})

    def _fetch(self, name: str, date: np.datetime64, when: str) -> np.ndarray | Finding | None:
        """The source field as (ensemble, grid), None if the source has none, or the ERROR of a failure."""
        try:
            fetched = self.source.fetch(name, date)  # pylint: disable=assignment-from-no-return  # Protocol stub
        except NotComparable as exc:
            self.not_comparable[name] = str(exc) or "not comparable"
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed[name] = f"source failed at {when}: {type(exc).__name__}: {exc}"
            return error(self.failed[name], variable=name, time=when)
        if fetched is None:
            self.unavailable.setdefault(name, []).append(when)
            return None
        expected = np.asarray(fetched)
        expected = expected.reshape(1, -1) if expected.ndim == 1 else expected
        if expected.shape != self.shape:
            self.failed[name] = f"source field has shape {expected.shape}, dataset has {self.shape}"
            return error(self.failed[name], variable=name, time=when)
        return expected

    def _compare(self, name: str, actual: np.ndarray, expected: np.ndarray, when: str) -> tuple[str, dict] | None:
        """Add one (variable, date) to its tally; the message and details of an ERROR if any value differs."""
        rtol, atol = self.tolerances.source_tolerance(name)
        tally = self.tallies.setdefault(name, _Tally())
        a_nan, e_nan = np.isnan(actual), np.isnan(expected)
        max_diff = _max_abs_diff(actual, expected)
        tally.compared += actual.size
        tally.identical += int(((actual == expected) | (a_nan & e_nan)).sum())
        tally.max_abs_diff = max(tally.max_abs_diff, max_diff or 0.0)
        bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if not bad.any():
            return None
        tally.differ += int(bad.sum())
        tally.dates_differ.append(when)
        members, cols = np.nonzero(bad)
        details: dict[str, Any] = {
            "n_differ": int(bad.sum()),
            "n_nan_mismatch": int((a_nan != e_nan).sum()),
            "max_abs_diff": max_diff,
            "first_points": self.points[cols][:10].tolist(),
        }
        if actual.shape[0] > 1:
            details["members"] = sorted(set(members.tolist()))
        return f"{int(bad.sum())}/{actual.size} sampled values differ from source", details

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
                        "n_dates": n_dates,
                        "n_points": len(self.points),
                        "n_members": n_members,
                    })


def _max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float | None:
    """Largest |actual - expected| over the values where both are finite; None if there are none."""
    with np.errstate(invalid="ignore"):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = diff[np.isfinite(diff)]
    return float(finite.max()) if finite.size else None


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
    for name, dates in spot.unavailable.items():
        yield warning(f"not available from source on {len(dates)}/{len(times)} sampled dates",
                      variable=name,
                      details={"dates": dates})
    for name, reason in spot.not_comparable.items():
        yield info(f"not comparable with source: {reason}", variable=name)
    yield spot.summary(len(times))
