"""Read-only access to an anemoi dataset (zarr v2) without going through
anemoi-datasets, so checks see exactly what is stored on disk."""
from functools import cached_property
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

STATISTICS = ("mean", "stdev", "minimum", "maximum")


class Dataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / ".zattrs").exists():
            raise FileNotFoundError(f"not a zarr dataset (no .zattrs): {self.path}")
        self.root = zarr.open_group(str(self.path), mode="r")

    @cached_property
    def attrs(self) -> dict[str, Any]:
        return json.loads((self.path / ".zattrs").read_text())

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def variables(self) -> list[str]:
        return list(self.attrs["variables"])

    def index(self, variable: str) -> int:
        return self.variables.index(variable)

    @cached_property
    def dates(self) -> np.ndarray:
        return self.root["dates"][:].astype("datetime64[s]")

    def date_str(self, t: int) -> str:
        return str(self.dates[t])

    @property
    def data(self) -> zarr.Array:
        """(time, variable, ensemble, grid); index lazily, one time step is one chunk."""
        return self.root["data"]

    def step(self, t: int) -> np.ndarray:
        """All variables at time index `t`, shape (variable, ensemble, grid)."""
        return self.data[t]

    def statistic(self, name: str) -> np.ndarray:
        return self.root[name][:]

    @property
    def arrays(self) -> list[str]:
        return sorted(self.root.array_keys())
