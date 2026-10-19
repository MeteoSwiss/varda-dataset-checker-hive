"""stats-bee: checks on the values and stored statistics.

Planned: recompute mean/stdev/min/max on sampled dates vs stored
statistics, physical ranges per variable, constant fields, and a cheap
per-step summary scan to catch single-time-step faults.
"""
import numpy as np

from ...honeycomb import CheckContext, check, error, info
from ...honeycomb.sampling import sample_time_indices


@check("stats")
def no_nans(ctx: CheckContext):
    """No NaNs on the sampled dates when the dataset says allow_nans: false."""
    ds = ctx.dataset
    if ds.attrs.get("allow_nans", False):
        yield info("skipped: dataset allows NaNs")
        return
    for t in sample_time_indices(len(ds.dates), ctx.spec.sampling):
        counts = np.isnan(ds.step(t)).reshape(len(ds.variables), -1).sum(axis=1)
        for i in np.flatnonzero(counts):
            yield error(f"{counts[i]} NaN values", variable=ds.variables[i], time=ds.date_str(t),
                        details={"time_index": int(t), "n_nan": int(counts[i])})
