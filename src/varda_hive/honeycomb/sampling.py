# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Seeded sampling, so every run with the same spec checks the same cells.

The sample is defined by the spec alone: the dates are drawn on the spec's period (`start_date` to
`end_date` every `frequency`), not on the dataset's time axis, so every rebuild that meets the spec
samples the same dates, and a reference written from one compares with the next. Changing the
sampling or the period in the spec changes the sample: an intended change, accepted with a new
reference. Only a spec without a period falls back to the dataset's dates."""
from typing import Collection

import numpy as np

from .dataset import parse_frequency, spec_date
from .spec import Sampling, Spec


def random_time_indices(n_times: int,
                        sampling: Sampling,
                        window: tuple[int, int] | None = None,
                        stratified: bool = False,
                        exclude: Collection[int] = ()) -> list[int]:
    """The random part of the sample, within `window` = [lo, hi) (default: all time steps).

    `stratified` draws one index per equal block of the window, which spreads the sample over the
    diurnal and synoptic cycles and so lowers the variance of estimates made from it. Indices in
    `exclude` (e.g. missing dates) are never drawn; with none excluded the draw is unchanged.
    """
    lo, hi = window or (0, n_times)
    if hi <= lo:
        return []
    if sampling.n_dates == "all":
        return [t for t in range(lo, hi) if t not in exclude]
    rng = np.random.default_rng([sampling.seed, 0])
    n = min(sampling.n_dates, hi - lo)
    if stratified:
        edges = np.linspace(lo, hi, n + 1).astype(int)
        picked = []
        for a, b in zip(edges[:-1], edges[1:]):
            if not any(a <= t < b for t in exclude):
                picked.append(int(rng.integers(a, b)))
            elif block := [t for t in range(a, b) if t not in exclude]:
                picked.append(int(rng.choice(block)))
        return picked
    if not exclude:
        return sorted((lo + rng.choice(hi - lo, size=n, replace=False)).tolist())
    candidates = np.array([t for t in range(lo, hi) if t not in exclude], dtype=int)
    return sorted(rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist())


def sample_axis(spec: Spec, dates: np.ndarray) -> np.ndarray:
    """The time axis the sample is drawn on: the spec's period when it has `start_date`, `end_date`
    and `frequency`, else the dataset's `dates`."""
    if spec.start_date is None or spec.end_date is None or spec.frequency is None:
        return np.asarray(dates).astype("datetime64[s]")
    start, end = spec_date(spec.start_date), spec_date(spec.end_date)
    axis = np.arange(start, end + np.timedelta64(1, "s"), parse_frequency(spec.frequency))
    return axis.astype("datetime64[s]")


def sampled_dates(spec: Spec, dates: np.ndarray, missing: Collection[int] = ()) -> tuple[dict[int, bool], list[str]]:
    """The sample in `dates`, as {time index: drawn at random}, and the sampled dates not in `dates` (or
    at one of its `missing` indices). The stratified random draw on `sample_axis`, without the missing
    dates (the spec's `missing_dates`, else the dataset's), plus `include_dates`. The one definition of
    the sample, so spot-bee and stats-bee check the same cells."""
    dates = np.asarray(dates).astype("datetime64[s]")
    axis = sample_axis(spec, dates)
    if spec.missing_dates is not None:
        skip = np.array([spec_date(d) for d in spec.missing_dates], dtype="datetime64[s]")
    else:
        skip = dates[sorted(missing)]
    exclude = set(np.flatnonzero(np.isin(axis, skip)).tolist())
    wanted = {str(spec_date(d)): False for d in spec.sampling.include_dates}
    wanted |= {
        str(axis[i]): True
        for i in random_time_indices(len(axis), spec.sampling, stratified=True, exclude=exclude)
    }
    position = {str(d): t for t, d in enumerate(dates) if t not in missing}
    picked = {position[d]: random for d, random in wanted.items() if d in position}
    return picked, sorted(d for d in wanted if d not in position)


def sample_time_indices(spec: Spec, dates: np.ndarray, missing: Collection[int] = ()) -> list[int]:
    """The sampled time indices in `dates`, sorted, see `sampled_dates`."""
    return sorted(sampled_dates(spec, dates, missing)[0])


def sample_points(n_grid: int, sampling: Sampling, salt: int = 0) -> np.ndarray:
    """Sorted seeded grid point indices, the same for every date; all points for `n_points: all`."""
    if sampling.n_points == "all" or sampling.n_points >= n_grid:
        return np.arange(n_grid)
    rng = np.random.default_rng([sampling.seed, 1, salt])
    return np.sort(rng.choice(n_grid, size=sampling.n_points, replace=False))
