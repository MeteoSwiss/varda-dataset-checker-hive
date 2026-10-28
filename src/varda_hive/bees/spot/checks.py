"""spot-bee: compare a seeded sample of (time, variable, grid points)
against the original source, provided by the dataset's source plug-in."""
from typing import Iterator

import numpy as np

from ...honeycomb import CheckContext, Finding, check, error, info, load_source, warning
from ...honeycomb.sampling import sample_points, sample_time_indices


@check("spot")
def matches_source(ctx: CheckContext) -> Iterator[Finding]:
    """Sampled values agree with the source within the spec tolerances."""
    if "source" not in ctx.cache:
        ctx.cache["source"] = load_source(ctx.spec)
    source = ctx.cache["source"]
    if source is None:
        yield info("skipped: no `source` in spec")
        return
    ds, tol = ctx.dataset, ctx.spec.tolerances
    points = sample_points(ds.data.shape[-1], ctx.spec.sampling)
    for t in sample_time_indices(len(ds.dates), ctx.spec.sampling):
        step = ds.step(t)
        for i, name in enumerate(ds.variables):
            expected = source.fetch(name, ds.dates[t])
            if expected is None:
                yield warning("not available from source", variable=name, time=ds.date_str(t))
                continue
            actual = step[i].reshape(-1)[points]
            expected = np.asarray(expected).reshape(-1)[points]
            bad = ~np.isclose(actual, expected, rtol=tol.source_rtol, atol=tol.source_atol, equal_nan=True)
            if bad.any():
                diff = np.abs(actual - expected)
                finite = diff[np.isfinite(diff)]
                yield error(f"{bad.sum()}/{len(points)} sampled points differ from source",
                            variable=name, time=ds.date_str(t),
                            details={"time_index": int(t), "n_nan_mismatch": int((~np.isfinite(diff)).sum()),
                                     "max_abs_diff": float(finite.max()) if finite.size else None,
                                     "first_points": points[bad][:10].tolist()})
