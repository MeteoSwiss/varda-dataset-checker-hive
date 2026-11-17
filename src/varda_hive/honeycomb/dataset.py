# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Read-only access to an anemoi dataset (zarr v2) without going through
anemoi-datasets, so checks see exactly what is stored on disk."""
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import zarr

STATISTICS = ("mean", "stdev", "minimum", "maximum")

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_ISO = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def parse_frequency(value: str | int) -> np.timedelta64:
    """An anemoi frequency as a timedelta: `1h`, `30m`, `1d`, an ISO 8601 duration (`PT6H`), or an int
    number of hours. ValueError for anything else."""
    if isinstance(value, int) and not isinstance(value, bool):
        return np.timedelta64(value * 3600, "s")
    text = str(value).strip()
    if m := re.fullmatch(r"(\d+)\s*([smhd])", text):
        return np.timedelta64(int(m[1]) * _UNITS[m[2]], "s")
    if (m := _ISO.match(text.upper())) and any(m.groups()):
        days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
        return np.timedelta64(((days * 24 + hours) * 60 + minutes) * 60 + seconds, "s")
    raise ValueError(f"not a frequency: {value!r}")


def format_frequency(step: np.timedelta64) -> str:
    """The shortest anemoi spelling of a step: 3600 s -> `1h`."""
    seconds = int(step / np.timedelta64(1, "s"))
    for unit in ("d", "h", "m"):
        if seconds and seconds % _UNITS[unit] == 0:
            return f"{seconds // _UNITS[unit]}{unit}"
    return f"{seconds}s"


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

    @cached_property
    def missing_indices(self) -> frozenset[int]:
        """Time indices listed in the `missing_dates` attribute: anemoi leaves their data unset and
        excludes them from the statistics, so checks must not sample them."""
        missing = np.array(self.attrs.get("missing_dates") or [], dtype="datetime64[s]")
        return frozenset(np.flatnonzero(np.isin(self.dates, missing)).tolist())

    @cached_property
    def latitudes(self) -> np.ndarray | None:
        return self.root["latitudes"][:] if "latitudes" in self.arrays else None

    @cached_property
    def longitudes(self) -> np.ndarray | None:
        return self.root["longitudes"][:] if "longitudes" in self.arrays else None

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

    @cached_property
    def arrays(self) -> list[str]:
        return sorted(self.root.array_keys())
