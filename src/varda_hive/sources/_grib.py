# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Decoding and validating GRIB messages for the GRIB-based sources (grib_index, fdb).

A decoded message must have the dataset's number of grid points, and, for grids whose coordinates
eccodes can compute, the dataset's coordinates; missing points (bitmap) become NaN. Also which
variables have no one-to-one GRIB message, and the GRIB `number` of each dataset member.
"""
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from ..honeycomb import NotComparable

T = TypeVar("T")

# gridType values whose coordinates eccodes computes as geographic latitudes/longitudes.
CHECKED_GRIDS = {"regular_ll", "reduced_ll", "regular_gg", "reduced_gg"}
GRID_ATOL_DEG = 1e-3  # GRIB stores coordinates to 1e-6 degrees, anemoi as float64


class ConfigError(str):
    """A variable's configuration error, raised by `fetch` so that only this variable fails."""


def map_targets(dataset: Any, target: Callable[[str, dict[str, Any]], T | str]) -> dict[str, T | str]:
    """`target(name, variables_metadata entry)` for every dataset variable: its target, the reason it is
    not comparable, or a ConfigError for an exception, so that only this variable fails."""
    metadata = dataset.attrs.get("variables_metadata", {})
    out: dict[str, T | str] = {}
    for name in dataset.variables:
        try:
            out[name] = target(name, metadata.get(name, {}))
        except (ValueError, KeyError, TypeError) as exc:
            out[name] = ConfigError(f"source variable {name}: {exc}")
    return out


def resolve(targets: Mapping[str, T | str], variable: str) -> T:
    """The target of `variable` from a source's `bind`: raises ValueError for a ConfigError, and
    NotComparable for a reason (a plain str) or an unknown variable."""
    target = targets.get(variable)
    if target is None:
        raise NotComparable("unknown to the source (bind was not called or variable not in dataset)")
    if isinstance(target, ConfigError):
        raise ValueError(str(target))
    if isinstance(target, str):
        raise NotComparable(target)
    return target


def load_eccodes(source: str, extra: str) -> Any:
    """The eccodes module, or an ImportError naming the extra to install."""
    try:
        import eccodes  # pylint: disable=import-outside-toplevel
    except (ImportError, RuntimeError) as exc:
        raise ImportError(f"{source} needs eccodes: install varda-dataset-checker-hive[{extra}]") from exc
    return eccodes


class Decoder:
    """Decodes GRIB messages into dataset-ordered values, checking them against the dataset grid."""

    def __init__(self, eccodes: Any, n_grid: int | None, coords: tuple[np.ndarray, np.ndarray] | None):
        self.eccodes = eccodes
        self.n_grid = n_grid
        self.coords = coords  # dataset (latitudes, longitudes), None to skip the coordinate check
        self._grids_checked: set[str] = set()  # md5GridSection of the grids that match the dataset

    def decode(self, message: bytes, where: str) -> np.ndarray:
        """The values of one GRIB message; `where` names it in errors (a path, a request)."""
        eccodes = self.eccodes
        handle = eccodes.codes_new_from_message(message)
        try:
            n_points = eccodes.codes_get(handle, "numberOfDataPoints")
            if self.n_grid is not None and n_points != self.n_grid:
                raise ValueError(f"GRIB message in {where} has {n_points} points, dataset grid has {self.n_grid}")
            self._check_grid(handle, where)
            values = eccodes.codes_get_values(handle).astype(np.float64)
            if eccodes.codes_get(handle, "bitmapPresent"):
                # The bitmap, not the missingValue, marks missing points: a real value may equal missingValue.
                # The bitmap's native type is bytes: codes_get_array returns a bool for it (eccodes 2.39).
                values[eccodes.codes_get_long_array(handle, "bitmap") == 0] = np.nan
        finally:
            eccodes.codes_release(handle)
        return values

    def count(self, data: bytes) -> int:
        """Number of GRIB messages in `data`, read from the `totalLength` of each."""
        eccodes, n, offset = self.eccodes, 0, 0
        while offset < len(data):
            handle = eccodes.codes_new_from_message(data[offset:])
            try:
                offset += int(eccodes.codes_get(handle, "totalLength"))
            finally:
                eccodes.codes_release(handle)
            n += 1
        return n

    def _check_grid(self, handle: Any, where: str) -> None:
        """Raise if the message's grid coordinates are not the dataset's, once per distinct grid."""
        eccodes = self.eccodes
        if self.coords is None or eccodes.codes_get(handle, "gridType") not in CHECKED_GRIDS:
            return
        grid = eccodes.codes_get(handle, "md5GridSection")
        if grid in self._grids_checked:
            return
        lat = eccodes.codes_get_array(handle, "latitudes")
        lon = eccodes.codes_get_array(handle, "longitudes")
        lat_ds, lon_ds = self.coords
        deviation = np.maximum(np.abs(lat - lat_ds), np.abs((lon - lon_ds + 180) % 360 - 180))
        worst = int(np.argmax(deviation))
        if deviation[worst] > GRID_ATOL_DEG:
            raise ValueError(f"GRIB grid in {where} is not the dataset's: point {worst} is at "
                             f"({lat[worst]:.4f}, {lon[worst]:.4f}) in GRIB, ({lat_ds[worst]:.4f}, "
                             f"{lon_ds[worst]:.4f}) in the dataset (cropped, thinned or reordered grid?)")
        self._grids_checked.add(grid)


def dataset_coords(dataset: Any, check_grid: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """The dataset's (latitudes, longitudes) for the grid check, None if not checked or not known."""
    if not check_grid or dataset.latitudes is None or dataset.longitudes is None:
        return None
    return np.asarray(dataset.latitudes, np.float64), np.asarray(dataset.longitudes, np.float64)


def member_numbers(members: list[int] | None, n_members: int) -> list[str]:
    """The GRIB `number` of each dataset member, empty for a single member the spec doesn't name.

    Matched when the spec names the members (also one member, e.g. a dataset of EPS member 5) or
    the dataset has several."""
    if members is not None:
        if len(members) != n_members:
            raise ValueError(f"`members` lists {len(members)} GRIB numbers, dataset has {n_members} members")
        return [str(m) for m in members]
    return [str(m) for m in range(n_members)] if n_members > 1 else []


def not_comparable(meta: dict[str, Any]) -> str | None:
    """Why a variable has no one-to-one GRIB message, from its variables_metadata entry."""
    if meta.get("computed_forcing"):
        return "computed forcing"
    if meta.get("process"):
        period = meta.get("period")
        return f"derived field ({meta['process']}" + (f" over {period[0]}..{period[1]})" if period else ")")
    if meta.get("constant_in_time"):
        return "constant field, not in the time-indexed source"
    if not isinstance(meta.get("mars"), dict) or "param" not in meta["mars"]:
        return "no mars metadata"
    return None


def text(value: Any) -> str:
    """Numbers in the spec or metadata (850, 0.01) as GRIB keys print them: 850.0 -> "850"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if value is None else str(value)
