"""meta-bee: metadata and structure checks. Cheap - never reads `data`.

Planned: anemoi-datasets validate, structure diff vs a reference dataset,
date/frequency/missing_dates consistency, grid and resolution vs spec.
"""
from ...honeycomb import CheckContext, check, error, info


@check("meta")
def variables_match_spec(ctx: CheckContext):
    """The dataset has exactly the variables listed in the spec."""
    expected = ctx.spec.variables
    if expected is None:
        yield info("skipped: no `variables` in spec")
        return
    actual = ctx.dataset.variables
    for name in sorted(set(expected) - set(actual)):
        yield error("variable missing from dataset", variable=name)
    for name in sorted(set(actual) - set(expected)):
        yield error("variable not in spec", variable=name)
