"""Seeded sampling, so every run with the same spec checks the same cells."""
import numpy as np

from .spec import Sampling


def sample_time_indices(n_times: int, sampling: Sampling) -> list[int]:
    rng = np.random.default_rng([sampling.seed, 0])
    n = min(sampling.n_dates, n_times)
    picked = set(rng.choice(n_times, size=n, replace=False).tolist())
    picked |= {t for t in sampling.include_dates if 0 <= t < n_times}
    return sorted(picked)


def sample_points(n_points: int, sampling: Sampling, salt: int = 0) -> np.ndarray:
    rng = np.random.default_rng([sampling.seed, 1, salt])
    n = min(sampling.n_points, n_points)
    return np.sort(rng.choice(n_points, size=n, replace=False))
