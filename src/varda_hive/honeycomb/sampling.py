# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Seeded sampling, so every run with the same spec checks the same cells."""
from typing import Collection

import numpy as np

from .spec import Sampling


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


def sample_time_indices(n_times: int, sampling: Sampling, exclude: Collection[int] = ()) -> list[int]:
    picked = set(random_time_indices(n_times, sampling, exclude=exclude))
    picked |= {t for t in sampling.include_dates if 0 <= t < n_times and t not in exclude}
    return sorted(picked)


def sample_points(n_grid: int, sampling: Sampling, salt: int = 0) -> np.ndarray:
    """Sorted seeded grid point indices, the same for every date; all points for `n_points: all`."""
    if sampling.n_points == "all" or sampling.n_points >= n_grid:
        return np.arange(n_grid)
    rng = np.random.default_rng([sampling.seed, 1, salt])
    return np.sort(rng.choice(n_grid, size=sampling.n_points, replace=False))
