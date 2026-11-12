# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Calibrated tolerances for `statistics_consistent`, written by `stats-bee calibrate`.

The default tolerance of the sampled mean is a distribution-free bound, z * stdev / sqrt(n): safe,
but loose, because stdev includes the spatial variance while the error of a sample of dates
depends only on how much the domain mean varies in time. Calibration measures the error instead:

1. Read a large calibration sample of M stratified dates inside the statistics window, and keep the
   per-date mean and stdev of every variable over the spec's point subset and over the full domain.
2. Temporal error: simulate many stratified subsamples of the size the check uses, estimate mean
   and stdev from each, and take the largest deviation from the estimate of all M dates.
3. Spatial bias: the point subset is the same on every date, so its error does not average out over
   dates. It is measured directly: the estimate from the subset minus the one from the full domain.

tolerance = |spatial bias| + margin * temporal error, per variable and statistic. The margin covers
the finite calibration sample. The stored statistics are not used, so the dataset under test (or a
previous version of it) can calibrate itself. The file has a fixed size: a few numbers per variable.
"""
from pathlib import Path
from typing import Any

import numpy as np

from ...honeycomb import Dataset, Spec
from ...honeycomb.sampling import random_time_indices, sample_points
from .statsfile import _num, provenance, read_file, sample_indices, statistics_window, summarise

FORMAT = "varda-hive-tolerances/1"
# The sampling settings a tolerance file is only valid for.
MATCH = ("seed", "n_dates", "n_points", "n_grid", "n_estimate")


def n_estimate(ds: Dataset, spec: Spec) -> int:
    """Number of random dates inside the statistics window the check estimates the statistics from."""
    return sum(sample_indices(ds, spec)[0].values())


def sampling_key(ds: Dataset, spec: Spec) -> dict[str, Any]:
    return {
        "seed": spec.sampling.seed,
        "n_dates": spec.sampling.n_dates,
        "n_points": spec.sampling.n_points,
        "n_grid": ds.data.shape[-1],
        "n_estimate": n_estimate(ds, spec),
    }


def calibrate(  # pylint: disable=too-many-locals
        ds: Dataset, spec: Spec, n_calib: int, n_sim: int = 2000, margin: float = 2.0) -> dict[str, Any]:
    """The tolerance file content for `ds` sampled as in `spec`, from `n_calib` calibration dates."""
    if spec.sampling.n_dates == "all":
        raise ValueError("nothing to calibrate: with `n_dates: all` the dates are not sampled")
    key = sampling_key(ds, spec)
    n = key["n_estimate"]
    window = statistics_window(ds)
    dates = random_time_indices(len(ds.dates),
                                spec.sampling.model_copy(update={"n_dates": n_calib}),
                                window,
                                stratified=True,
                                exclude=ds.missing_indices)
    if len(dates) < 2 * n:
        raise ValueError(f"calibration needs at least 2 x {n} dates in the statistics window, got {len(dates)}")
    points = sample_points(ds.data.shape[-1], spec.sampling)
    full_domain = len(points) == ds.data.shape[-1]

    # (date, variable, [mean, stdev]) over the point subset and over the full domain
    sub = np.empty((len(dates), len(ds.variables), 2))
    full = np.empty_like(sub)
    for k, t in enumerate(dates):
        step = ds.step(t)
        full[k] = summarise(step)[:, :2]
        sub[k] = full[k] if full_domain else summarise(step, points)[:, :2]

    def estimates(m: np.ndarray, s: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        return m.mean(axis=axis), np.sqrt((s**2).mean(axis=axis) + m.var(axis=axis))

    m, s = sub[..., 0], sub[..., 1]
    all_mean, all_stdev = estimates(m, s, 0)
    full_mean, full_stdev = estimates(full[..., 0], full[..., 1], 0)
    # Stratified subsamples of n of the M dates: one date per equal block, as the check draws them.
    rng = np.random.default_rng([spec.sampling.seed, 3])
    edges = np.linspace(0, len(dates), n + 1).astype(int)
    idx = np.stack([rng.integers(a, b, size=n_sim) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    sim_mean, sim_stdev = estimates(m[idx], s[idx], 1)  # (n_sim, variable)
    temporal = {"mean": np.abs(sim_mean - all_mean).max(axis=0), "stdev": np.abs(sim_stdev - all_stdev).max(axis=0)}
    bias = {"mean": all_mean - full_mean, "stdev": all_stdev - full_stdev}

    variables = {}
    for i, name in enumerate(ds.variables):
        entry = {stat: _num(abs(bias[stat][i]) + margin * temporal[stat][i]) for stat in ("mean", "stdev")}
        entry |= {f"temporal_{stat}": _num(temporal[stat][i]) for stat in ("mean", "stdev")}
        entry |= {f"spatial_bias_{stat}": _num(bias[stat][i]) for stat in ("mean", "stdev")}
        variables[name] = entry
    return provenance(ds, FORMAT) | {
        "sampling": key,
        "calibration": {
            "n_calib": len(dates),
            "n_sim": n_sim,
            "margin": margin
        },
        "variables": variables,
    }


def load_tolerances(path: str | Path) -> dict:
    return read_file(path, FORMAT, "tolerance file")


def mismatch(tolerances: dict, ds: Dataset, spec: Spec) -> dict[str, list]:
    """The sampling settings that differ between the tolerance file and this check: {key: [file, now]}."""
    now = sampling_key(ds, spec)
    return {k: [tolerances["sampling"].get(k), now[k]] for k in MATCH if tolerances["sampling"].get(k) != now[k]}
