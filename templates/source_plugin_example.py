"""Example spot-bee source plug-in: reads the "true" values from another
anemoi zarr dataset (e.g. a trusted copy).

A real plug-in lives in the dataset's repo and reads the dataset's original
source instead, e.g. GRIB files via a grib-index DB for ICH1. The only
contract is a `fetch(variable, date)` method returning the flattened field
in dataset grid order, or None if the source has no data for it.

Reference it from the spec as:

    source:
      plugin: source_plugin_example.py:ZarrSource
      args: {path: /path/to/reference.zarr}
"""
import numpy as np

from varda_hive.honeycomb import Dataset


class ZarrSource:
    def __init__(self, path: str):
        self.ds = Dataset(path)
        self._time_index = {d: t for t, d in enumerate(self.ds.dates)}

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        t = self._time_index.get(np.datetime64(date, "s"))
        if t is None or variable not in self.ds.variables:
            return None
        return self.ds.data[t, self.ds.index(variable)].reshape(-1)
