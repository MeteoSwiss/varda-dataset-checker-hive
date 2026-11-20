# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
import zarr

VARIABLES = ["FIS", "T_2M", "T_850", "U_10M", "V_10M"]
NT, NG = 8, 50


def _mars(variable: str) -> dict:
    """mars keys like anemoi writes them: T_850 is T on pressure level 850."""
    name, _, level = variable.rpartition("_")
    if level.isdigit():
        return {"param": name, "levtype": "pl", "levelist": int(level)}
    return {"param": variable, "levtype": "sfc"}


def build_dataset(path: Path, variables=VARIABLES, seed: int = 42, nt: int = NT, zarr_format: int = 2) -> Path:
    """Small anemoi-like zarr dataset (format 2 or 3): data (time, variable, 1, grid) + metadata.
    Fields drift in time; the first variable (FIS) is a constant field."""
    rng = np.random.default_rng(seed)
    data = rng.normal(280, 5, size=(nt, len(variables), 1, NG)) + 0.5 * np.arange(nt)[:, None, None, None]
    data[:, 0] = data[0, 0]
    data = data.astype("f4")
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    root.create_array("data", data=data, chunks=(1, len(variables), 1, NG))
    dates = np.datetime64("2020-03-01T00:00:00") + np.arange(nt) * np.timedelta64(1, "h")
    root.create_array("dates", data=dates.astype("datetime64[s]"))
    root.create_array("latitudes", data=np.linspace(45, 48, NG))
    root.create_array("longitudes", data=np.linspace(5, 11, NG))
    v = data.astype("f8")
    root.create_array("mean", data=v.mean(axis=(0, 2, 3)))
    root.create_array("stdev", data=v.std(axis=(0, 2, 3)))
    root.create_array("minimum", data=v.min(axis=(0, 2, 3)))
    root.create_array("maximum", data=v.max(axis=(0, 2, 3)))
    attrs = {
        "variables": list(variables),
        "allow_nans": False,
        "frequency": "1h",
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "statistics_start_date": str(dates[0]),
        "statistics_end_date": str(dates[-1]),
        "constant_fields": list(variables[:1]),
        "field_shape": [NG],
        "shape": list(data.shape),
        "missing_dates": [],
        "resolution": "2km",
        "variables_metadata": {
            v: {
                "mars": _mars(v),
                **({
                    "constant_in_time": True
                } if v == variables[0] else {})
            }
            for v in variables
        },
    }
    root.attrs.update(attrs)
    return path


@pytest.fixture
def dataset_path(tmp_path):
    return build_dataset(tmp_path / "clean.zarr")


@pytest.fixture
def dataset_path_v3(tmp_path):
    return build_dataset(tmp_path / "clean-v3.zarr", zarr_format=3)


@pytest.fixture
def write_spec(tmp_path):

    def write(text: str, name: str = "spec.yaml") -> Path:
        path = tmp_path / name
        path.write_text(text)
        return path

    return write
