# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example spot-bee source plug-in: reads the "true" values from another
anemoi zarr dataset (e.g. a trusted copy).

For GRIB sources indexed with `anemoi-datasets grib-index` (e.g. ICH1) use
varda_hive.sources.grib_index:GribIndexSource instead; a dataset repo with
another kind of source writes its own plug-in like this one. The only required
member is `fetch(variable, date)`, returning the field in dataset grid order,
shape (grid,) or (ensemble, grid), or None if the source has no data for it.
`fetch` may raise varda_hive.honeycomb.NotComparable for a variable that
cannot be compared (e.g. a derived field), and an optional `bind(dataset)` is
called once before the first fetch. See docs/spot-bee.md.

Reference it from the spec as:

    source:
      plugin: source_plugin_example.py:ZarrSource
      args: {path: /path/to/reference.zarr}   # or relative to the spec file
"""
from pathlib import Path

import numpy as np

from varda_hive.honeycomb import Dataset, NotComparable


class ZarrSource:
    def __init__(self, path: str, base_dir: str | Path = "."):
        self.ds = Dataset(Path(base_dir) / path)  # a relative path is relative to the spec file
        self._time_index = {d: t for t, d in enumerate(self.ds.dates)}

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        if variable not in self.ds.variables:
            raise NotComparable("not in the reference dataset")
        t = self._time_index.get(np.datetime64(date, "s"))
        if t is None:
            return None
        return self.ds.data[t, self.ds.index(variable)]  # (ensemble, grid)
