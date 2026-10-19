"""Seeded sampling, so every run with the same spec checks the same cells."""
import numpy as np

from .spec import Sampling


def random_time_indices(n_times: int, sampling: Sampling, window: tuple[int, int] | None = None,
                        stratified: bool = False) -> list[int]:
    """The random part of the sample, within `window` = [lo, hi) (default: all time steps).

    `stratified` draws one index per equal block of the window, which spreads the sample over the
    diurnal and synoptic cycles and so lowers the variance of estimates made from it.
    """
    lo, hi = window or (0, n_times)
    if hi <= lo:
        return []
    if sampling.n_dates == "all":
        return list(range(lo, hi))
    rng = np.random.default_rng([sampling.seed, 0])
    n = min(sampling.n_dates, hi - lo)
    if stratified:
        edges = np.linspace(lo, hi, n + 1).astype(int)
        return [int(rng.integers(a, b)) for a, b in zip(edges[:-1], edges[1:])]
    return sorted((lo + rng.choice(hi - lo, size=n, replace=False)).tolist())


def sample_time_indices(n_times: int, sampling: Sampling) -> list[int]:
    picked = set(random_time_indices(n_times, sampling))
    picked |= {t for t in sampling.include_dates if 0 <= t < n_times}
    return sorted(picked)


def sample_points(n_points: int, sampling: Sampling, salt: int = 0) -> np.ndarray:
    rng = np.random.default_rng([sampling.seed, 1, salt])
    n = min(sampling.n_points, n_points)
    return np.sort(rng.choice(n_points, size=n, replace=False))
