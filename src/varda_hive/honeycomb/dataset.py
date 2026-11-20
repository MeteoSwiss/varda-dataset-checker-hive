# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Read-only access to an anemoi dataset (zarr format 2 or 3) without going through
anemoi-datasets, so checks see exactly what is stored on disk.

The group attributes are read straight from the json file (`.zattrs` for format 2, the `attributes` of
the root `zarr.json` for format 3), so meta-bee stays cheap; arrays are read through zarr."""
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Any
import warnings

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


def parse_date(value: Any) -> np.datetime64:
    """A date attribute (`2020-03-01T00:00:00`, `2020-03-01 00:00:00`, trailing `Z`) to the second."""
    return np.datetime64(str(value).replace(" ", "T").removesuffix("Z"), "s")


def format_frequency(step: np.timedelta64) -> str:
    """The shortest anemoi spelling of a step: 3600 s -> `1h`."""
    seconds = int(step / np.timedelta64(1, "s"))
    for unit in ("d", "h", "m"):
        if seconds and seconds % _UNITS[unit] == 0:
            return f"{seconds // _UNITS[unit]}{unit}"
    return f"{seconds}s"


def _zarr_json(path: Path) -> dict[str, Any] | None:
    """The root `zarr.json` of a format 3 group, None if there is none or it is not a group."""
    try:
        meta = json.loads((path / "zarr.json").read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("node_type") == "group" else None


def is_zarr(path: str | Path) -> bool:
    """A zarr group: format 2 (`.zattrs`) or format 3 (`zarr.json` with `node_type: group`)."""
    path = Path(path)
    return (path / ".zattrs").is_file() or _zarr_json(path) is not None


class Dataset:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not is_zarr(self.path):
            raise FileNotFoundError(f"not a zarr dataset (no .zattrs or zarr.json group): {self.path}")
        # not the consolidated metadata (zarr 3's default): checks must see the arrays on disk
        self.root = zarr.open_group(str(self.path), mode="r", use_consolidated=False)

    @cached_property
    def attrs(self) -> dict[str, Any]:
        """The group attributes, from the same format zarr opened: `zarr.json` wins over a `.zattrs`
        left behind, as in zarr 3."""
        if (meta := _zarr_json(self.path)) is not None:
            return dict(meta.get("attributes") or {})
        return json.loads((self.path / ".zattrs").read_text())

    def array(self, name: str) -> zarr.Array:
        """The array `name`; TypeError if it is a group."""
        node = self.root[name]
        if not isinstance(node, zarr.Array):
            raise TypeError(f"`{name}` is not an array")
        return node

    def values(self, name: str) -> np.ndarray:
        """All values of the array `name`."""
        return np.asarray(self.array(name)[...])

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
        return self.values("dates").astype("datetime64[s]")

    @cached_property
    def missing_indices(self) -> frozenset[int]:
        """Time indices listed in the `missing_dates` attribute: anemoi leaves their data unset and
        excludes them from the statistics, so checks must not sample them."""
        missing = np.array([parse_date(d) for d in self.attrs.get("missing_dates") or []], dtype="datetime64[s]")
        return frozenset(np.flatnonzero(np.isin(self.dates, missing)).tolist())

    @cached_property
    def latitudes(self) -> np.ndarray | None:
        return self.values("latitudes") if "latitudes" in self.arrays else None

    @cached_property
    def longitudes(self) -> np.ndarray | None:
        return self.values("longitudes") if "longitudes" in self.arrays else None

    def date_str(self, t: int) -> str:
        return str(self.dates[t])

    @property
    def data(self) -> zarr.Array:
        """(time, variable, ensemble, grid); index lazily, one time step is one or a few chunks."""
        return self.array("data")

    def step(self, t: int) -> np.ndarray:
        """All variables at time index `t`, shape (variable, ensemble, grid)."""
        return np.asarray(self.data[t])

    def statistic(self, name: str) -> np.ndarray:
        return self.values(name)

    @cached_property
    def arrays(self) -> list[str]:
        # zarr 3 warns about files that are not zarr nodes (e.g. a `.snakemake_timestamp` of the build)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not recognized as a component of a Zarr hierarchy")
            return sorted(self.root.array_keys())
