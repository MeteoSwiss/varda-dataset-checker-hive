# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""FdbSource with a fake pyfdb: GRIB messages built with eccodes, served by request."""
from datetime import datetime, timedelta
import io
import json
import sys
import types

import numpy as np
import pytest
import zarr

from conftest import NG, VARIABLES
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run
from varda_hive.sources.fdb import step, time_request_keys

eccodes = pytest.importorskip("eccodes")
from test_grib_source import _message, grid_coordinates  # pylint: disable=wrong-import-position

BASE = {"class": "rd", "stream": "reanl", "expver": "r001"}


def _key(request: dict) -> frozenset:
    return frozenset((str(k), str(v)) for k, v in request.items())


class FakeFDB:
    """Serves `messages` by request; records every request and the config it was opened with."""
    messages: dict[frozenset, bytes] = {}
    requests: list[dict] = []
    configs: list = []

    def __init__(self, config=None, user_config=None):
        FakeFDB.configs.append((config, user_config))

    def retrieve(self, request: dict) -> io.BytesIO:
        assert all(isinstance(v, str) for v in request.values()), request  # pyfdb wants str (or int)
        FakeFDB.requests.append(dict(request))
        return io.BytesIO(FakeFDB.messages.get(_key(request), b""))


@pytest.fixture
def fdb(monkeypatch, dataset_path):
    """A fake FDB holding the clean dataset, as requested with offset_from_date (run 00:00, step = hour).
    The dataset is put on the GRIB test grid; T_2M has a paramId in its metadata."""
    FakeFDB.messages, FakeFDB.requests, FakeFDB.configs = {}, [], []
    monkeypatch.setitem(sys.modules, "pyfdb", types.SimpleNamespace(FDB=FakeFDB))
    z = zarr.open_group(str(dataset_path), mode="r+")
    lat, lon = grid_coordinates(NG)
    z.create_array("latitudes", data=lat, overwrite=True)
    z.create_array("longitudes", data=lon, overwrite=True)
    attrs = json.loads((dataset_path / ".zattrs").read_text())
    attrs["variables_metadata"]["T_2M"]["grib"] = {"paramId": 500011, "shortName": "T_2M"}
    (dataset_path / ".zattrs").write_text(json.dumps(attrs))
    data = z["data"][:]
    for t in range(data.shape[0]):
        for name, request in (("T_2M", {
                "param": "500011",
                "levtype": "sfc"
        }), ("T_850", {
                "param": "T",
                "levtype": "pl",
                "levelist": "850"
        }), ("U_10M", {
                "param": "U_10M",
                "levtype": "sfc"
        }), ("V_10M", {
                "param": "V_10M",
                "levtype": "sfc"
        })):
            put(t, request, data[t, VARIABLES.index(name), 0])
    return data


def put(t: int, request: dict, values: np.ndarray, extra: bytes = b"") -> None:
    full = BASE | request | {"date": "20200301", "time": "0000", "step": str(t)}
    FakeFDB.messages[_key(full)] = _message(values) + extra


def _spec(**args) -> Spec:
    return Spec(source={
        "plugin": "varda_hive.sources.fdb:FdbSource",
        "args": {
            "request": BASE,
            "offset_from_date": True
        } | args
    },
                sampling={"n_dates": "all"})


def _findings(path, spec):
    return run(CheckContext(Dataset(path), spec), bees=["spot"]).findings


def _summary(findings):
    [summary] = [f for f in findings if f.message.startswith("spot summary")]
    return summary.details


def test_fdb_clean(fdb, dataset_path):
    findings = _findings(dataset_path, _spec(fdb_config={"type": "local"}))
    assert [f for f in findings if f.severity != Severity.INFO] == []
    details = _summary(findings)
    assert details["identical"] == ["T_2M", "T_850", "U_10M", "V_10M"]
    assert details["not_comparable"] == {"FIS": "constant field, not in the time-indexed source"}
    assert FakeFDB.configs == [({"type": "local"}, None)]  # one FDB for the run
    # paramId when the metadata has one, mars.param otherwise; levelist only when set
    assert BASE | {
        "param": "500011",
        "levtype": "sfc",
        "date": "20200301",
        "time": "0000",
        "step": "3"
    } in FakeFDB.requests
    assert {
        "param": "T",
        "levtype": "pl",
        "levelist": "850"
    }.items() <= next(r for r in FakeFDB.requests if r["param"] == "T").items()
    assert all("levelist" not in r for r in FakeFDB.requests if r["levtype"] == "sfc")


def test_fdb_detects_changed_value_and_missing(fdb, dataset_path):
    zarr.open_group(str(dataset_path), mode="r+")["data"][4, VARIABLES.index("T_850"), 0, 7] += 0.5
    del FakeFDB.messages[_key(BASE | {
        "param": "V_10M",
        "levtype": "sfc",
        "date": "20200301",
        "time": "0000",
        "step": "3"
    })]
    findings = _findings(dataset_path, _spec())
    [err] = [f for f in findings if f.severity == Severity.ERROR]
    assert (err.variable, err.time, err.details["first_points"]) == ("T_850", "2020-03-01T04:00:00", [7])
    [warn] = [f for f in findings if f.severity == Severity.WARNING]
    assert (warn.variable, warn.details["dates"]) == ("V_10M", ["2020-03-01T03:00:00"])


def test_fdb_overrides(fdb, dataset_path):
    """An override's request is merged on top of the default one, scale/offset apply to the values."""
    for t in range(fdb.shape[0]):
        put(t, {"param": "U_10M", "levtype": "pl", "levelist": "800"}, fdb[t, VARIABLES.index("U_10M"), 0] * 1000)
        put(t, {"param": "500007", "levtype": "sfc"}, fdb[0, 0, 0])
    findings = _findings(
        dataset_path,
        _spec(
            variables={
                "U_10M": {
                    "request": {
                        "levtype": "pl",
                        "levelist": 800
                    },
                    "scale": 0.001
                },
                "FIS": {
                    "request": {
                        "param": 500007
                    }
                },  # constant: compared because it has an override
            }))
    assert [f for f in findings if f.severity != Severity.INFO] == []
    details = _summary(findings)
    assert details["not_comparable"] == {} and "U_10M" in details["within_tolerance"]
    assert "FIS" in details["identical"]


def test_fdb_not_comparable_rules(fdb, dataset_path):
    attrs = json.loads((dataset_path / ".zattrs").read_text())
    meta = attrs["variables_metadata"]
    meta["T_850"]["mars"]["levtype"] = "unknown"
    meta["U_10M"] |= {"process": "accumulation", "period": ["0h", "1h"]}
    meta["V_10M"] = {"computed_forcing": True}
    (dataset_path / ".zattrs").write_text(json.dumps(attrs))
    details = _summary(_findings(dataset_path, _spec(skip={"T_2M": "masked"})))
    assert details["not_comparable"] == {
        "FIS": "constant field, not in the time-indexed source",
        "T_2M": "masked",
        "T_850": "levtype unknown (interpolated, e.g. from model to pressure levels)",
        "U_10M": "derived field (accumulation over 0h..1h)",
        "V_10M": "computed forcing",
    }
    # an override makes a derived field comparable (e.g. a maximum that is the raw GRIB field)
    assert "U_10M" in _summary(_findings(dataset_path, _spec(variables={"U_10M": None})))["identical"]


def test_fdb_step_zero_from_previous_date(fdb, dataset_path):
    """At midnight, step 24 of the previous day's run; for the listed variables only."""
    midnight = BASE | {"param": "V_10M", "levtype": "sfc", "date": "20200301", "time": "0000", "step": "0"}
    previous = midnight | {"date": "20200229", "step": "24"}
    FakeFDB.messages[_key(previous)] = FakeFDB.messages.pop(_key(midnight))
    findings = _findings(dataset_path, _spec(step_zero_from_previous_date=["V_10M"]))
    assert [f for f in findings if f.severity != Severity.INFO] == []
    assert previous in FakeFDB.requests and midnight not in FakeFDB.requests
    assert BASE | {"param": "U_10M", "levtype": "sfc", "date": "20200301", "time": "0000", "step": "0"} \
        in FakeFDB.requests  # not listed: step 0
    # true: for all variables
    findings = _findings(dataset_path, _spec(step_zero_from_previous_date=True))
    assert {f.variable for f in findings if f.severity == Severity.WARNING} == {"T_2M", "T_850", "U_10M"}


@pytest.mark.parametrize("when, offset, zero, expected", [
    (datetime(2020, 1, 1, 23, 30), True, False, {
        "date": "20200101",
        "time": "0000",
        "step": "23h30m"
    }),
    (datetime(2020, 1, 2, 0, 10), True, False, {
        "date": "20200102",
        "time": "0000",
        "step": "10m"
    }),
    (datetime(2020, 1, 2, 6), True, False, {
        "date": "20200102",
        "time": "0000",
        "step": "6"
    }),
    (datetime(2020, 1, 2), True, False, {
        "date": "20200102",
        "time": "0000",
        "step": "0"
    }),
    (datetime(2020, 1, 2), True, True, {
        "date": "20200101",
        "time": "0000",
        "step": "24"
    }),
    (datetime(2020, 1, 2, 0, 10), True, True, {
        "date": "20200102",
        "time": "0000",
        "step": "10m"
    }),
    (datetime(2020, 1, 2, 6), False, False, {
        "date": "20200102",
        "time": "0600"
    }),
])
def test_fdb_time_request_keys(when, offset, zero, expected):
    """As the anemoi fdb source requests them, with the step spelled as FDB indexes it."""
    assert time_request_keys(when, offset, zero) == expected


def test_fdb_step_must_be_whole_minutes():
    assert step(timedelta(hours=1, minutes=10)) == "1h10m"
    for bad in (timedelta(seconds=30), timedelta(hours=-1)):
        with pytest.raises(ValueError, match="whole number of minutes"):
            step(bad)


def test_fdb_duplicate_message_and_wrong_grid(fdb, dataset_path):
    put(2, {"param": "500011", "levtype": "sfc"}, fdb[2, 1, 0], extra=_message(fdb[2, 1, 0] + 1))
    put(0, {"param": "V_10M", "levtype": "sfc"}, np.zeros(NG + 5))
    errors = {f.variable: f.message for f in _findings(dataset_path, _spec()) if f.severity == Severity.ERROR}
    assert set(errors) == {"T_2M", "V_10M"}
    assert "2 GRIB messages match" in errors["T_2M"] and "55 points" in errors["V_10M"]


def test_fdb_config_errors(fdb, dataset_path):
    """A bad override fails its own variable only."""
    for variables, message in [({
            "V_10M": {
                "requets": {}
            }
    }, "unknown keys ['requets']"), ({
            "V_10M": {
                "request": [1]
            }
    }, "`request` must be a mapping")]:
        findings = _findings(dataset_path, _spec(variables=variables))
        [err] = [f for f in findings if f.severity == Severity.ERROR]
        assert err.variable == "V_10M" and message in err.message and "source variable V_10M" in err.message
        assert _summary(findings)["identical"] == ["T_2M", "T_850", "U_10M"]
    attrs = json.loads((dataset_path / ".zattrs").read_text())
    attrs["variables_metadata"]["V_10M"] = {}
    (dataset_path / ".zattrs").write_text(json.dumps(attrs))
    [err] = [f for f in _findings(dataset_path, _spec(variables={"V_10M": {}})) if f.severity == Severity.ERROR]
    assert "no `param`" in err.message
    # mars metadata without param: the override's request gives it; without an override, not comparable
    attrs["variables_metadata"]["V_10M"] = {"mars": {"levtype": "sfc"}}
    (dataset_path / ".zattrs").write_text(json.dumps(attrs))
    findings = _findings(dataset_path, _spec(variables={"V_10M": {"request": {"param": "V_10M"}}}))
    assert [f for f in findings if f.severity == Severity.ERROR] == []
    assert "V_10M" in _summary(findings)["identical"]
    assert _summary(_findings(dataset_path, _spec()))["not_comparable"]["V_10M"] == "no mars metadata"
    [err] = _findings(dataset_path, _spec(request=["class", "rd"]))
    assert "must be a mapping" in err.message


def test_fdb_ensemble_members(fdb, dataset_path):
    """Every member is retrieved with its GRIB `number`."""
    for t in range(fdb.shape[0]):
        put(t, {"param": "500011", "levtype": "sfc", "number": "5"}, fdb[t, 1, 0])
    findings = _findings(dataset_path, _spec(members=[5], skip={v: "-" for v in ("T_850", "U_10M", "V_10M")}))
    assert _summary(findings)["identical"] == ["T_2M"]
    assert {r["number"] for r in FakeFDB.requests} == {"5"}
    [err] = _findings(dataset_path, _spec(members=[1, 2]))
    assert "`members` lists 2 GRIB numbers, dataset has 1 members" in err.message


def test_fdb_without_pyfdb(fdb, dataset_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyfdb", None)
    [err] = _findings(dataset_path, _spec())
    assert err.severity == Severity.ERROR and "install varda-dataset-checker-hive[fdb]" in err.message
