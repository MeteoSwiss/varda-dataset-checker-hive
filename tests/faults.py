# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fault primitives: small edits of an anemoi-like zarr store (format 2) in place.

The names and parameters follow the primitives of the Varda-tiny faulty-dataset generator (the
`faults[].primitive` entries of its manifest), so a scenario there can be written here as a list of
`(primitive, params)` pairs, see `tests/test_detection.py`. Every primitive takes the store's path first.
"""
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr

STATISTICS = ("mean", "stdev", "minimum", "maximum")

# --- helpers ----------------------------------------------------------------------------------


def get_attrs(path: Path) -> dict:
    return json.loads((path / ".zattrs").read_text())


def set_attrs(path: Path, **changes) -> None:
    """Set top-level attributes; None removes one."""
    attrs = get_attrs(path)
    for key, value in changes.items():
        if value is None:
            attrs.pop(key)
        else:
            attrs[key] = value
    (path / ".zattrs").write_text(json.dumps(attrs))


def get_array(path: Path, name: str) -> np.ndarray:
    values = zarr.open_group(str(path), mode="r")[name][:]
    return values.astype("datetime64[s]") if name == "dates" else values


def set_array(path: Path, name: str, values: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="r+")
    if name in root and root[name].shape == values.shape:
        root[name][:] = values
    else:
        root.create_array(name, data=values, overwrite=True)


def _index(path: Path, variable: str) -> int:
    return get_attrs(path)["variables"].index(variable)


def recompute_statistics(path: Path) -> None:
    """mean, stdev, minimum and maximum over the statistics window, as anemoi-datasets stores them."""
    attrs = get_attrs(path)
    dates = get_array(path, "dates")
    start = np.datetime64(attrs.get("statistics_start_date", dates[0]), "s")
    end = np.datetime64(attrs.get("statistics_end_date", dates[-1]), "s")
    window = get_array(path, "data")[(dates >= start) & (dates <= end)].astype("f8")
    for name, values in zip(STATISTICS, (np.nanmean(window, axis=(0, 2, 3)), np.nanstd(
            window, axis=(0, 2, 3)), np.nanmin(window, axis=(0, 2, 3)), np.nanmax(window, axis=(0, 2, 3)))):
        set_array(path, name, values)


# --- metadata ---------------------------------------------------------------------------------


def set_attr(path: Path, path_in_attrs: list[str], value: Any) -> None:
    """Set a nested attribute, e.g. `["variables_metadata", "T_850", "mars", "levelist"]`."""
    attrs = get_attrs(path)
    node = attrs
    for key in path_in_attrs[:-1]:
        node = node[key]
    node[path_in_attrs[-1]] = value
    set_attrs(path, **{path_in_attrs[0]: attrs[path_in_attrs[0]]})


def swap_variable_names(path: Path, a: str, b: str) -> None:
    """Swap two names in `variables`; data and statistics unchanged."""
    variables = get_attrs(path)["variables"]
    i, j = variables.index(a), variables.index(b)
    variables[i], variables[j] = b, a
    set_attrs(path, variables=variables)


def edit_constant_fields(path: Path, remove: list[str], add: list[str]) -> None:
    constant = [v for v in get_attrs(path)["constant_fields"] if v not in remove]
    set_attrs(path, constant_fields=constant + add)


def shift_date_attrs(path: Path, keys: list[str], hours: int) -> None:
    attrs = get_attrs(path)
    set_attrs(path, **{k: str(np.datetime64(attrs[k], "s") + np.timedelta64(hours, "h")) for k in keys})


def add_missing_date(path: Path, index: int) -> None:
    """List a date that is present in the data in `missing_dates`."""
    set_attrs(path, missing_dates=get_attrs(path)["missing_dates"] + [str(get_array(path, "dates")[index])])


def corrupt_dates(path: Path, mode: str, index: int) -> None:
    """`swap`: swap dates `index` and `index + 1`; `duplicate`: `index + 1` repeats `index`."""
    dates = get_array(path, "dates")
    if mode == "swap":
        dates[[index, index + 1]] = dates[[index + 1, index]]
    elif mode == "duplicate":
        dates[index + 1] = dates[index]
    else:
        raise ValueError(mode)
    set_array(path, "dates", dates)


def perturb_coordinates(path: Path, offset: dict[str, float] | None = None, swap: bool = False) -> None:
    lat, lon = get_array(path, "latitudes"), get_array(path, "longitudes")
    if swap:
        lat, lon = lon, lat
    offset = offset or {}
    set_array(path, "latitudes", lat + offset.get("latitudes", 0.0))
    set_array(path, "longitudes", lon + offset.get("longitudes", 0.0))


# --- additional variables and dates -----------------------------------------------------------


def _insert_variable(path: Path, name: str, values: np.ndarray, metadata: dict) -> None:
    """Insert a variable at its sorted position (anemoi sorts them), with its statistics."""
    attrs = get_attrs(path)
    variables = attrs["variables"]
    i = int(np.searchsorted(variables, name))
    data = np.insert(get_array(path, "data"), i, values, axis=1)
    set_array(path, "data", data)
    for stat in STATISTICS:
        set_array(path, stat, np.insert(get_array(path, stat), i, 0.0))
    set_attrs(path,
              variables=variables[:i] + [name] + variables[i:],
              variables_metadata={
                  **attrs["variables_metadata"], name: metadata
              },
              shape=list(data.shape),
              chunks=[1, data.shape[1], 1, data.shape[3]])
    recompute_statistics(path)


def add_variable_copy(path: Path, name: str, copy_of: str) -> None:
    attrs = get_attrs(path)
    values = get_array(path, "data")[:, attrs["variables"].index(copy_of)]
    _insert_variable(path, name, values, attrs["variables_metadata"][copy_of])


def add_pressure_level(path: Path, param: str, level: int, between: list[int]) -> None:
    """A level interpolated linearly in log(p) between two existing ones."""
    attrs = get_attrs(path)
    lower, upper = between
    data = get_array(path, "data")
    w = np.log(level / lower) / np.log(upper / lower)
    values = (1 - w) * data[:, _index(path, f"{param}_{lower}")] + w * data[:, _index(path, f"{param}_{upper}")]
    metadata = json.loads(json.dumps(attrs["variables_metadata"][f"{param}_{upper}"]))
    metadata["mars"]["levelist"] = level
    _insert_variable(path, f"{param}_{level}", values.astype(data.dtype), metadata)


def append_timesteps(path: Path, count: int) -> None:
    """`count` more steps after the end, with the data of the last `count` steps."""
    data, dates = get_array(path, "data"), get_array(path, "dates")
    step = dates[1] - dates[0]
    dates = np.concatenate([dates, dates[-1] + step * np.arange(1, count + 1)])
    data = np.concatenate([data, data[-count:]])
    set_array(path, "data", data)
    set_array(path, "dates", dates)
    set_attrs(path, end_date=str(dates[-1]), last_date=str(dates[-1]).replace("T", " "), shape=list(data.shape))


# --- statistics and data ----------------------------------------------------------------------


def scale_statistic(path: Path, variable: str, stat: str, factor: float) -> None:
    values = get_array(path, stat)
    values[_index(path, variable)] *= factor
    set_array(path, stat, values)


def set_statistic(path: Path, variable: str, stat: str, value: float) -> None:
    values = get_array(path, stat)
    values[_index(path, variable)] = value
    set_array(path, stat, values)


def swap_statistics(path: Path, variable: str, a: str, b: str) -> None:
    i = _index(path, variable)
    va, vb = get_array(path, a), get_array(path, b)
    va[i], vb[i] = vb[i], va[i]
    set_array(path, a, va)
    set_array(path, b, vb)


def offset_variable(path: Path, variable: str, offset: float) -> None:
    """Add `offset` at every step and to mean, minimum and maximum: self-consistent."""
    i = _index(path, variable)
    data = get_array(path, "data")
    data[:, i] += offset
    set_array(path, "data", data)
    for stat in ("mean", "minimum", "maximum"):
        values = get_array(path, stat)
        values[i] += offset
        set_array(path, stat, values)


def corrupt_timestep(  # pylint: disable=too-many-arguments
        path: Path,
        variables: list[str] | str,
        index: int,
        mode: str,
        *,
        other: str | None = None,
        fraction: float = 0.01,
        amplitude: float = 0.0,
        factor: float = 1.0,
        offset: float = 0.0,
        shift: int = 0,
        update_statistics: bool = False,
        seed: int = 0) -> None:
    """Corrupt the data of `variables` ("all" for every one) at time `index`.

    modes: `nan`, `zero`, `spike` (+`amplitude` on a seeded `fraction` of the points), `scale`
    (x * factor + offset), `swap` (with `other`), `stale` (copy of `index - 1`), `roll` (by `shift`
    points). The statistics stay as they were unless `update_statistics`."""
    data = get_array(path, "data")
    names = get_attrs(path)["variables"] if variables == "all" else variables
    modes: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "nan": lambda x, i: np.full_like(x, np.nan),
        "zero": lambda x, i: np.zeros_like(x),
        "scale": lambda x, i: x * factor + offset,
        "stale": lambda x, i: data[index - 1, i],
        "roll": lambda x, i: np.roll(x, shift, axis=-1),
        "swap": lambda x, i: data[index, _index(path, other)].copy(),
    }
    if mode == "spike":
        points = np.random.default_rng(seed).choice(data.shape[-1],
                                                    max(1, round(fraction * data.shape[-1])),
                                                    replace=False)
        modes["spike"] = lambda x, i: np.where(np.isin(np.arange(x.shape[-1]), points), x + amplitude, x)
    new = {i: modes[mode](data[index, i], i) for i in map(lambda n: _index(path, n), names)}
    if mode == "swap":
        new[_index(path, other)] = data[index, _index(path, names[0])].copy()
    for i, values in new.items():
        data[index, i] = values
    set_array(path, "data", data)
    if update_statistics:
        recompute_statistics(path)


PRIMITIVES: dict[str, Callable[..., None]] = {
    f.__name__: f
    for f in (set_attr, swap_variable_names, edit_constant_fields, shift_date_attrs, add_missing_date, corrupt_dates,
              perturb_coordinates, add_variable_copy, add_pressure_level, append_timesteps, scale_statistic,
              set_statistic, swap_statistics, offset_variable, corrupt_timestep)
}


def apply(path: Path, faults: list[tuple[str, dict]]) -> Path:
    """Apply `(primitive, params)` pairs in order. `set_attr` takes the manifest's `path` key."""
    for primitive, params in faults:
        if primitive == "set_attr":
            set_attr(path, params["path"], params["value"])
        else:
            PRIMITIVES[primitive](path, **params)
    return path
