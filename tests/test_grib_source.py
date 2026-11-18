# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee with the grib-index source, on GRIB files and an anemoi-style index written here."""
from pathlib import Path
import sqlite3

from conftest import NG, VARIABLES
import numpy as np
import pytest
import zarr

from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run

eccodes = pytest.importorskip("eccodes")

COLUMNS = ["valid_datetime", "shortName", "level", "typeOfLevel", "number", "step"]


def _message(values: np.ndarray, packing: str = "grid_ieee") -> bytes:
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(handle, "Ni", len(values) // 5)
        eccodes.codes_set(handle, "Nj", 5)
        eccodes.codes_set(handle, "packingType", packing)
        if packing == "grid_simple":
            eccodes.codes_set(handle, "bitsPerValue", 12)
        eccodes.codes_set_values(handle, values.astype(np.float64))
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


class GribArchive:
    """GRIB messages of the clean dataset, appended to one file and indexed like `anemoi-datasets grib-index`."""

    def __init__(self, root: Path, dataset: Path):
        self.root = root
        self.db = root / "index.db"
        z = zarr.open_group(str(dataset), mode="r")
        self.data, self.dates = z["data"][:], [str(d) for d in z["dates"][:].astype("datetime64[s]")]
        (root / "data").mkdir()
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE paths (id INTEGER PRIMARY KEY, path TEXT)")
            con.execute("CREATE TABLE grib_index (_id INTEGER PRIMARY KEY, _path_id INTEGER, _offset INTEGER, "
                        "_length INTEGER, " + ", ".join(f'"{c}" TEXT' for c in COLUMNS) + ")")
            con.execute("INSERT INTO paths VALUES (1, 'data/fields.grib')")

    def add(self, t: int, values: np.ndarray, packing: str = "grid_ieee", **keys: str) -> None:
        message = _message(values, packing)
        path = self.root / "data" / "fields.grib"
        offset = path.stat().st_size if path.exists() else 0
        with open(path, "ab") as f:
            f.write(message)
        row = {"valid_datetime": self.dates[t], "number": "0", "step": "0"} | keys
        with sqlite3.connect(self.db) as con:
            con.execute(
                "INSERT INTO grib_index (_path_id, _offset, _length, " + ", ".join(f'"{c}"' for c in row) +
                ") VALUES (1, ?, ?, " + ", ".join("?" * len(row)) + ")", (offset, len(message), *row.values()))

    def add_clean(self, skip=()) -> None:
        """T_2M, T_850 as stored; U_10M in mm-like units (x1000); V_10M; FIS (constant) is not indexed."""
        for t in range(len(self.dates)):
            for name in VARIABLES[1:]:
                if (name, t) in skip:
                    continue
                values = self.data[t, VARIABLES.index(name), 0]
                if name == "T_850":
                    self.add(t, values, shortName="T", level="850", typeOfLevel="isobaricInhPa")
                elif name == "U_10M":
                    self.add(t, values.astype(np.float64) * 1000, shortName="U_KM")
                else:
                    self.add(t, values, shortName=name)

    def spec(self, variables=None, skip=None, **sampling) -> Spec:
        args = {
            "root": str(self.root),
            "indexes": {
                "main": "index.db"
            },
            "variables": {
                "U_10M": {
                    "match": {
                        "shortName": "U_KM"
                    },
                    "scale": 0.001
                }
            } | (variables or {}),
            "skip": skip or {},
        }
        return Spec(source={
            "plugin": "varda_hive.sources.grib_index:GribIndexSource",
            "args": args
        },
                    sampling={"n_dates": "all"} | sampling)


@pytest.fixture
def archive(tmp_path, dataset_path):
    (tmp_path / "grib").mkdir()
    return GribArchive(tmp_path / "grib", dataset_path)


def _findings(path, spec):
    return run(CheckContext(Dataset(path), spec), bees=["spot"]).findings


def _summary(findings):
    [summary] = [f for f in findings if f.message.startswith("spot summary")]
    return summary


def test_grib_clean(archive, dataset_path):
    archive.add_clean()
    findings = _findings(dataset_path, archive.spec())
    assert [f for f in findings if f.severity != Severity.INFO] == []
    details = _summary(findings).details
    assert details["identical"] == ["T_2M", "T_850", "V_10M"]
    assert list(details["within_tolerance"]) == ["U_10M"]  # x1000 then x0.001 is not bit-exact
    assert details["not_comparable"] == {"FIS": "constant field, not in the time-indexed source"}
    assert details["n_dates"] == 8


def test_grib_detects_changed_value(archive, dataset_path):
    archive.add_clean()
    zarr.open_group(str(dataset_path), mode="r+")["data"][4, VARIABLES.index("T_850"), 0, 7] += 0.5
    errors = [f for f in _findings(dataset_path, archive.spec()) if f.severity == Severity.ERROR]
    assert [(f.variable, f.time) for f in errors] == [("T_850", "2020-03-01T04:00:00")]
    assert errors[0].details["first_points"] == [7]
    assert errors[0].details["max_abs_diff"] == pytest.approx(0.5, abs=1e-4)


def test_grib_missing_and_skipped(archive, dataset_path):
    archive.add_clean(skip={("V_10M", 3)})
    findings = _findings(dataset_path, archive.spec(skip={"T_2M": "checked elsewhere"}))
    [warn] = [f for f in findings if f.severity == Severity.WARNING]
    assert (warn.variable, warn.details["dates"]) == ("V_10M", ["2020-03-01T03:00:00"])
    assert _summary(findings).details["not_comparable"]["T_2M"] == "checked elsewhere"


def test_grib_lossy_packing_within_tolerance(archive, dataset_path):
    for t in range(len(archive.dates)):
        archive.add(t, archive.data[t, VARIABLES.index("T_2M"), 0], packing="grid_simple", shortName="T_2M")
    spec = archive.spec(skip={v: "-" for v in ("T_850", "U_10M", "V_10M")})
    findings = _findings(dataset_path, spec)
    assert list(_summary(findings).details["within_tolerance"]) == ["T_2M"]
    tight = spec.model_copy(update={"tolerances": Spec(tolerances={"source_atol": 0, "source_rtol": 1e-9}).tolerances})
    assert {f.variable for f in _findings(dataset_path, tight) if f.severity == Severity.ERROR} == {"T_2M"}


def test_grib_ambiguous_message(archive, dataset_path):
    archive.add_clean()
    archive.add(2, archive.data[2, VARIABLES.index("T_2M"), 0] + 1, shortName="T_2M")
    [err] = [f for f in _findings(dataset_path, archive.spec()) if f.severity == Severity.ERROR]
    assert err.variable == "T_2M" and "2 GRIB messages match" in err.message


def test_grib_wrong_grid(archive, dataset_path):
    archive.add_clean(skip={("V_10M", 0)})
    archive.add(0, np.zeros(NG + 5), shortName="V_10M")
    [err] = [f for f in _findings(dataset_path, archive.spec()) if f.severity == Severity.ERROR]
    assert err.variable == "V_10M" and "55 points" in err.message


def test_grib_config_errors(archive, dataset_path, tmp_path):
    archive.add_clean()
    [err] = _findings(dataset_path, archive.spec(variables={"V_10M": {"match": {"paramName": "x"}}}))
    assert err.severity == Severity.ERROR and "no column ['paramName']" in err.message
    [err] = _findings(dataset_path, archive.spec(variables={"V_10M": {"index": "other"}}))
    assert "index 'other'" in err.message
    spec = archive.spec()
    spec.source.args["indexes"] = {"main": str(tmp_path / "missing.db")}
    [err] = _findings(dataset_path, spec)
    assert "not found" in err.message


def test_grib_root_relative_to_spec(archive, dataset_path, write_spec):
    """A relative `root` resolves against the spec file's directory."""
    from varda_hive.honeycomb import load_spec  # pylint: disable=import-outside-toplevel
    archive.add_clean()
    spec = load_spec(
        write_spec("source:\n  plugin: varda_hive.sources.grib_index:GribIndexSource\n"
                   "  args: {root: grib, indexes: {main: index.db}, skip: {U_10M: x}}\n"
                   "sampling: {n_dates: 2}\n"))
    findings = _findings(dataset_path, spec)
    assert [f for f in findings if f.severity != Severity.INFO] == []
    assert _summary(findings).details["identical"] == ["T_2M", "T_850", "V_10M"]
