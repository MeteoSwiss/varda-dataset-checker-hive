# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee with the grib-index source, on GRIB files and an anemoi-style index written here."""
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from conftest import NG, VARIABLES, build_dataset
import numpy as np
import pytest
import zarr

from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run

eccodes = pytest.importorskip("eccodes")

COLUMNS = ["valid_datetime", "shortName", "level", "typeOfLevel", "number", "step"]

NJ, LAT0, LON0 = 5, 48.0, 5.0  # the test grid: NJ rows from LAT0 southwards, 0.5 deg columns from LON0


def grid_coordinates(n_points: int, lat0: float = LAT0) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes of the regular test grid, in GRIB order (rows north to south)."""
    ni = n_points // NJ
    return np.repeat(lat0 - np.arange(NJ), ni), np.tile(LON0 + 0.5 * np.arange(ni), NJ)


def _message(values: np.ndarray, packing: str = "grid_ieee", lat0: float = LAT0) -> bytes:
    """A GRIB2 message of `values` on the regular test grid; NaN become missing values in a bitmap."""
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        ni = len(values) // NJ
        eccodes.codes_set(handle, "Ni", ni)
        eccodes.codes_set(handle, "Nj", NJ)
        for key, value in [("latitudeOfFirstGridPointInDegrees", lat0),
                           ("latitudeOfLastGridPointInDegrees", lat0 - (NJ - 1)),
                           ("longitudeOfFirstGridPointInDegrees", LON0),
                           ("longitudeOfLastGridPointInDegrees", LON0 + 0.5 * (ni - 1)),
                           ("iDirectionIncrementInDegrees", 0.5), ("jDirectionIncrementInDegrees", 1.0)]:
            eccodes.codes_set(handle, key, value)
        eccodes.codes_set(handle, "packingType", packing)
        if packing == "grid_simple":
            eccodes.codes_set(handle, "bitsPerValue", 12)
        values = values.astype(np.float64)
        if np.isnan(values).any():  # a marker other than the decoder's default 9999, so 9999 can be a value
            eccodes.codes_set(handle, "missingValue", -1.0e30)
            eccodes.codes_set(handle, "bitmapPresent", 1)
            values = np.where(np.isnan(values), -1.0e30, values)
        eccodes.codes_set_values(handle, values)
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


class GribArchive:
    """GRIB messages of the clean dataset, appended to one file and indexed like `anemoi-datasets grib-index`."""

    def __init__(self, root: Path, dataset: Path):
        self.root = root
        self.db = root / "index.db"
        z = zarr.open_group(str(dataset), mode="r+")
        lat, lon = grid_coordinates(z["data"].shape[-1])  # the dataset is on the GRIB grid
        z.create_array("latitudes", data=lat, overwrite=True)
        z.create_array("longitudes", data=lon, overwrite=True)
        self.data, self.dates = z["data"][:], [str(d) for d in z["dates"][:].astype("datetime64[s]")]
        (root / "data").mkdir()
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE paths (id INTEGER PRIMARY KEY, path TEXT)")
            con.execute("CREATE TABLE grib_index (_id INTEGER PRIMARY KEY, _path_id INTEGER, _offset INTEGER, "
                        "_length INTEGER, " + ", ".join(f'"{c}" TEXT' for c in COLUMNS) + ")")
            con.execute("INSERT INTO paths VALUES (1, 'data/fields.grib')")

    def add(self, t: int, values: np.ndarray, packing: str = "grid_ieee", lat0: float = LAT0, **keys: str) -> None:
        message = _message(values, packing, lat0)
        path = self.root / "data" / "fields.grib"
        offset = path.stat().st_size if path.exists() else 0
        with open(path, "ab") as f:
            f.write(message)
        row = {"valid_datetime": self.dates[t], "number": "0", "step": "0"} | keys
        with sqlite3.connect(self.db) as con:
            con.execute(
                "INSERT INTO grib_index (_path_id, _offset, _length, " + ", ".join(f'"{c}"' for c in row) +
                ") VALUES (1, ?, ?, " + ", ".join("?" * len(row)) + ")", (offset, len(message), *row.values()))

    def add_clean(self, skip=(), numbers=("0", )) -> None:
        """T_2M, T_850 as stored; U_10M in mm-like units (x1000); V_10M; FIS (constant) is not indexed.
        Member m of the dataset gets GRIB `number` numbers[m]."""
        for t in range(len(self.dates)):
            for name in VARIABLES[1:]:
                if (name, t) in skip:
                    continue
                for member, number in enumerate(numbers):
                    values = self.data[t, VARIABLES.index(name), member]
                    if name == "T_850":
                        self.add(t, values, shortName="T", level="850", typeOfLevel="isobaricInhPa", number=number)
                    elif name == "U_10M":
                        self.add(t, values.astype(np.float64) * 1000, shortName="U_KM", number=number)
                    else:
                        self.add(t, values, shortName=name, number=number)

    def spec(self, variables=None, skip=None, members=None, **sampling) -> Spec:
        args: dict = {
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
        } | ({
            "members": members
        } if members is not None else {})
        return Spec(source={
            "plugin": "varda_hive.sources.grib_index:GribIndexSource",
            "args": args
        },
                    sampling={"n_dates": "all"} | sampling)


@pytest.fixture
def archive(tmp_path, dataset_path):
    (tmp_path / "grib").mkdir()
    return GribArchive(tmp_path / "grib", dataset_path)


def _report(path, spec):
    return run(CheckContext(Dataset(path), spec), bees=["spot"])


def _findings(path, spec):
    return _report(path, spec).findings


def _errors(findings):
    return [f for f in findings if f.severity == Severity.ERROR]


def _edit_metadata(path: Path, **mars_or_meta) -> None:
    """Replace variables_metadata entries of the dataset at `path`."""
    attrs = json.loads((path / ".zattrs").read_text())
    attrs["variables_metadata"] |= mars_or_meta
    (path / ".zattrs").write_text(json.dumps(attrs))


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
    """A bad override fails its own variable only; the others are still compared."""
    archive.add_clean()
    for variables, message in [
        ({
            "V_10M": {
                "match": {
                    "paramName": "x"
                }
            }
        }, "no column ['paramName']"),
        ({
            "V_10M": {
                "index": "other"
            }
        }, "index 'other'"),
        ({
            "V_10M": {
                "match": {
                    "shortName": "V_10M"
                },
                "scal": 2
            }
        }, "unknown keys ['scal']"),
        ({
            "V_10M": {
                "match": {}
            }
        }, "empty `match`"),
    ]:
        findings = _findings(dataset_path, archive.spec(variables=variables))
        [err] = _errors(findings)
        assert err.variable == "V_10M" and message in err.message and "source variable V_10M" in err.message
        assert _summary(findings).details["identical"] == ["T_2M", "T_850"]
    spec = archive.spec()
    spec.source.args["indexes"] = {"main": str(tmp_path / "missing.db")}
    [err] = _findings(dataset_path, spec)  # the source itself can't be built: the check crashes
    assert "not found" in err.message
    spec.source.args["indexes"] = {"main": []}
    [err] = _findings(dataset_path, spec)
    assert "lists no DB" in err.message


@pytest.mark.parametrize("override", [None, {}])
def test_grib_override_without_match_uses_metadata(archive, dataset_path, override):
    """`V_10M:` (None) or `{}`: the match comes from the metadata, as without an override."""
    archive.add_clean()
    findings = _findings(dataset_path, archive.spec(variables={"V_10M": override}))
    assert _errors(findings) == []
    assert "V_10M" in _summary(findings).details["identical"]


def test_grib_metadata_mapping(archive, dataset_path):
    """Not-comparable reasons and level matching from variables_metadata."""
    archive.add_clean(skip={("V_10M", t) for t in range(len(archive.dates))})
    for t in range(len(archive.dates)):
        values = archive.data[t, VARIABLES.index("V_10M"), 0]
        archive.add(t, values, shortName="V_10M", level="10")
        archive.add(t, values + 1, shortName="V_10M", level="20")  # another level: ambiguous without `level`
    _edit_metadata(
        dataset_path,
        T_850={"mars": {
            "param": "T",
            "levtype": "pl",
            "levelist": 850.0
        }},  # float: matches "850"
        V_10M={"mars": {
            "param": "V_10M",
            "levtype": "sfc",
            "levelist": 10
        }})
    findings = _findings(dataset_path, archive.spec())
    assert _errors(findings) == []
    assert _summary(findings).details["identical"] == ["T_2M", "T_850", "V_10M"]

    _edit_metadata(
        dataset_path,
        T_2M={
            "process": "accumulation",
            "period": ["0h", "6h"],
            "mars": {
                "param": "T_2M"
            }
        },
        T_850={"mars": {
            "param": "T",
            "levtype": "pl"
        }},  # no levelist: a mapping error
        U_10M={"computed_forcing": True},
        V_10M={"mars": {
            "levtype": "sfc"
        }})
    findings = _findings(dataset_path, archive.spec())
    assert _summary(findings).details["not_comparable"] == {
        "FIS": "constant field, not in the time-indexed source",
        "T_2M": "derived field (accumulation over 0h..6h)",
        "V_10M": "no mars metadata",
    }
    [err] = _errors(findings)
    assert err.variable == "T_850" and "without `levelist`" in err.message
    assert list(_summary(findings).details["within_tolerance"]) == ["U_10M"]  # an override beats computed_forcing


def test_grib_nothing_compared_fails_strict(archive, dataset_path):
    """A source that finds nothing, or a spec skipping everything, can't pass under --strict."""
    findings = _findings(dataset_path, archive.spec())  # empty archive, e.g. a wrong root or date format
    assert {f.variable for f in _errors(findings)} == {"T_2M", "T_850", "U_10M", "V_10M"}
    assert all("on any of 8 sampled dates" in f.message for f in _errors(findings))
    report = _report(dataset_path, archive.spec(skip={v: "-" for v in VARIABLES[1:]}))
    assert _errors(report.findings) == []
    assert [f.message for f in report.findings
            if f.skipped] == ["skipped: no sampled value was compared with the source"]
    assert report.exit_code() == 0 and report.exit_code(strict=True) == 1


def test_grib_index_list_and_offset(archive, dataset_path):
    """An index given as a list of DBs (e.g. per day) and an `offset` override."""
    archive.add_clean()
    for t in range(len(archive.dates)):
        archive.add(t, archive.data[t, VARIABLES.index("V_10M"), 0].astype(np.float64) + 10, shortName="V_PLUS10")
    second = archive.root / "second.db"
    shutil.copy(archive.db, second)
    half = archive.dates[4]
    for db, keep in [(archive.db, "<"), (second, ">=")]:
        with sqlite3.connect(db) as con:
            con.execute(f"DELETE FROM grib_index WHERE NOT valid_datetime {keep} ?", (half, ))
    spec = archive.spec(variables={"V_10M": {"match": {"shortName": "V_PLUS10"}, "offset": -10}})
    spec.source.args["indexes"] = {"main": ["index.db", "second.db"]}
    findings = _findings(dataset_path, spec)
    assert _errors(findings) == []
    details = _summary(findings).details
    assert details["identical"] == ["T_2M", "T_850", "V_10M"] and list(details["within_tolerance"]) == ["U_10M"]


def test_grib_ensemble(tmp_path):
    """Every member is matched by its GRIB `number`, default 0, 1, ... or `members`; an index without a
    `number` column fails only the variables that use it."""
    path = build_dataset(tmp_path / "ens.zarr")
    group = zarr.open_group(str(path), mode="r+")
    data = group["data"][:]
    two = np.concatenate([data, data + 1], axis=2)
    group.create_array("data", data=two, chunks=(1, len(VARIABLES), 2, NG), overwrite=True)
    (tmp_path / "grib").mkdir()
    archive = GribArchive(tmp_path / "grib", path)
    archive.add_clean(numbers=("1", "2"))
    findings = _findings(path, archive.spec(members=[1, 2]))
    assert _errors(findings) == []
    assert _summary(findings).details["identical"] == ["T_2M", "T_850", "V_10M"]
    assert _summary(findings).details["n_members"] == 2
    [unavailable] = {f.message for f in _errors(_findings(path, archive.spec()))}  # numbers 0, 1: no member 0
    assert "on any of 8 sampled dates" in unavailable
    group["data"][2, VARIABLES.index("T_2M"), 1, 3] += 1  # member 1 only
    [err] = _errors(_findings(path, archive.spec(members=[1, 2])))
    assert (err.variable, err.details["members"], err.details["first_points"]) == ("T_2M", [1], [3])

    with sqlite3.connect(archive.root / "nonumber.db") as con:
        con.execute("CREATE TABLE grib_index (_id INTEGER PRIMARY KEY, _path_id INTEGER, _offset INTEGER, "
                    "_length INTEGER, valid_datetime TEXT, shortName TEXT)")
    spec = archive.spec(members=[1, 2], variables={"V_10M": {"index": "other", "match": {"shortName": "V_10M"}}})
    spec.source.args["indexes"] = {"main": "index.db", "other": "nonumber.db"}
    errors = {e.variable: e.message for e in _errors(_findings(path, spec))}
    assert sorted(errors) == ["T_2M", "V_10M"]  # T_2M: the value changed above
    assert "index 'other' has no `number` column" in errors["V_10M"]
    [err] = _findings(path, archive.spec(members=[1]))
    assert "`members` lists 1 GRIB numbers, dataset has 2 members" in err.message


def test_grib_bitmap_missing_values(archive, dataset_path):
    """Missing values of a GRIB bitmap are NaN: identical to NaN in the dataset, an ERROR elsewhere."""
    archive.add_clean(skip={("T_2M", 2)})
    values = archive.data[2, VARIABLES.index("T_2M"), 0].astype(np.float64)
    values[[5, 9]] = np.nan
    archive.add(2, values, shortName="T_2M")
    zarr.open_group(str(dataset_path), mode="r+")["data"][2, VARIABLES.index("T_2M"), 0, 5] = np.nan
    [err] = _errors(_findings(dataset_path, archive.spec()))
    assert (err.variable, err.details["n_differ"], err.details["n_nan_mismatch"]) == ("T_2M", 1, 1)
    assert err.details["first_points"] == [9]
    zarr.open_group(str(dataset_path), mode="r+")["data"][2, VARIABLES.index("T_2M"), 0, 9] = np.nan
    findings = _findings(dataset_path, archive.spec())
    assert _errors(findings) == [] and "T_2M" in _summary(findings).details["identical"]


def test_grib_bitmap_value_equal_to_missing_value(archive, dataset_path):
    """Only the bitmap marks missing points: a real value equal to eccodes' missingValue (9999) stays."""
    archive.add_clean(skip={("T_2M", 2)})
    values = archive.data[2, VARIABLES.index("T_2M"), 0].astype(np.float64)
    values[5], values[7] = np.nan, 9999.0
    archive.add(2, values, shortName="T_2M")
    data = zarr.open_group(str(dataset_path), mode="r+")["data"]
    data[2, VARIABLES.index("T_2M"), 0, 5] = np.nan
    data[2, VARIABLES.index("T_2M"), 0, 7] = 9999.0
    findings = _findings(dataset_path, archive.spec())
    assert _errors(findings) == [] and "T_2M" in _summary(findings).details["identical"]


def test_grib_grid_not_the_datasets(archive, dataset_path):
    """A message on another grid (here shifted 1 deg north) is an ERROR of the variable, not "differ"."""
    archive.add_clean(skip={("V_10M", t) for t in range(len(archive.dates))})
    for t in range(len(archive.dates)):
        archive.add(t, archive.data[t, VARIABLES.index("V_10M"), 0], shortName="V_10M", lat0=49.0)
    findings = _findings(dataset_path, archive.spec())
    [err] = _errors(findings)
    assert err.variable == "V_10M" and "is not the dataset's: point 0 is at (49.0000, 5.0000)" in err.message
    assert _summary(findings).details["identical"] == ["T_2M", "T_850"]
    spec = archive.spec()
    spec.source.args["check_grid"] = False  # e.g. coordinates the dataset stores differently
    assert _errors(_findings(dataset_path, spec)) == []


def test_grib_odd_paths_and_path_column(tmp_path, dataset_path):
    """DB paths with URI characters, and an index with a `path` key column of its own."""
    root = tmp_path / "odd#dir?x=1%"
    root.mkdir()
    archive = GribArchive(root, dataset_path)
    archive.add_clean()
    with sqlite3.connect(archive.db) as con:
        con.execute('ALTER TABLE grib_index ADD COLUMN "path" TEXT')
        con.execute("UPDATE grib_index SET path = 'not/the/file.grib'")
    findings = _findings(dataset_path, archive.spec())
    assert _errors(findings) == []
    assert _summary(findings).details["identical"] == ["T_2M", "T_850", "V_10M"]


def test_grib_without_eccodes(archive, dataset_path, monkeypatch):
    """Without eccodes the check fails once, with the install hint, not once per variable."""
    archive.add_clean()
    monkeypatch.setitem(sys.modules, "eccodes", None)
    [err] = _findings(dataset_path, archive.spec())
    assert err.severity == Severity.ERROR and "install varda-dataset-checker-hive[grib]" in err.message


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


def test_grib_members_single(archive, dataset_path):
    """`members` matches `number` for a single-member dataset too, e.g. one member of an ensemble archive."""
    archive.add_clean(numbers=("5", ))
    for t in range(len(archive.dates)):  # another member in the same archive
        archive.add(t, archive.data[t, VARIABLES.index("T_2M"), 0] + 1, shortName="T_2M", number="4")
    findings = _findings(dataset_path, archive.spec(members=[5]))
    assert _errors(findings) == [] and _summary(findings).details["identical"] == ["T_2M", "T_850", "V_10M"]
    [err] = _errors(_findings(dataset_path, archive.spec()))  # without members: both messages match
    assert err.variable == "T_2M" and "2 GRIB messages match" in err.message


def test_grib_never_available_accepted_by_spec(archive, dataset_path):
    """A variable the source never has is an ERROR, unless ignore_warnings accepts it."""
    archive.add_clean(skip={("V_10M", t) for t in range(len(archive.dates))})
    assert [f.variable for f in _errors(_findings(dataset_path, archive.spec()))] == ["V_10M"]
    spec = archive.spec().model_copy(update={"ignore_warnings": {"matches_source": ["V_10M"]}})
    findings = _findings(dataset_path, spec)
    assert _errors(findings) == [] and [f for f in findings if f.severity == Severity.WARNING] == []
    [accepted] = [f for f in findings if f.variable == "V_10M"]
    assert accepted.message.endswith("(ignored by spec)")
