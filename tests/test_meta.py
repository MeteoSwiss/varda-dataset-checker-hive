# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""meta-bee checks on synthetic faults: each fault is a small change of `.zattrs` or of an array of
the clean test dataset, and must be reported by the expected check. The same faults as the Varda-tiny
`m-*`/`a-*` scenarios; tests/test_detection.py runs those on a subset of the real dataset."""
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from conftest import NG, NT, VARIABLES, build_dataset
from faults import set_array, set_attrs
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, load_spec, run
from varda_hive.honeycomb.dataset import format_frequency, parse_frequency
from varda_hive.bees.meta.metafile import fingerprint, save_meta

SPEC = f"""
variables:
  FIS:
  T_2M: {{param: T_2M, levtype: sfc}}
  T_850: {{param: T, levtype: pl, levelist: 850}}
  U_10M:
  V_10M:
start_date: 2020-03-01T00:00:00
end_date: 2020-03-01T{NT - 1:02d}:00:00
frequency: 1h
grid_points: {NG}
resolution: 2km
bbox: {{lat_min: 45, lat_max: 48, lon_min: 5, lon_max: 11}}
allow_nans: false
missing_dates: []
reference: reference.meta.yaml
"""


def _varmeta(path: Path, variable: str, key: str, value) -> None:
    metadata = json.loads((path / ".zattrs").read_text())["variables_metadata"]
    metadata[variable]["mars"][key] = value
    set_attrs(path, variables_metadata=metadata)


def _dates(path: Path) -> np.ndarray:
    return zarr.open_group(str(path), mode="r")["dates"][:].astype("datetime64[s]")


def _with(values: np.ndarray, index: int, value) -> np.ndarray:
    values = values.copy()
    values[index] = value
    return values


def _drop_last_date(p: Path) -> None:
    """One date fewer than time steps in `data`, with consistent date attributes."""
    d = _dates(p)[:-1].copy()
    set_array(p, "dates", d)
    set_attrs(p, end_date=str(d[-1]))


def _swap_dates(p: Path) -> None:
    d = _dates(p)
    d[[3, 4]] = d[[4, 3]]
    set_array(p, "dates", d)


def _duplicate_date(p: Path) -> None:
    d = _dates(p)
    d[4] = d[3]
    set_array(p, "dates", d)


def _irregular_dates(p: Path) -> None:
    d = _dates(p)
    d[5] += np.timedelta64(30, "m")
    set_array(p, "dates", d)


def _swap_latlon(p: Path) -> None:
    root = zarr.open_group(str(p), mode="r")
    lat, lon = root["latitudes"][:], root["longitudes"][:]
    set_array(p, "latitudes", lon)
    set_array(p, "longitudes", lat)


def _constant_fields(p: Path) -> None:
    set_attrs(p, constant_fields=["T_2M"])


# fault -> (mutation, check expected to report an ERROR without spec or reference, or None)
FAULTS = {
    "dates-swap": (_swap_dates, "dates_valid"),
    "dates-duplicate": (_duplicate_date, "dates_valid"),
    "dates-irregular": (_irregular_dates, "dates_valid"),
    "frequency": (lambda p: set_attrs(p, frequency="3h"), "frequency_consistent"),
    "dates-attr": (lambda p: set_attrs(p, start_date="2020-03-02T00:00:00"), "date_attrs_consistent"),
    "missing-dates-outside": (lambda p: set_attrs(p, missing_dates=["2021-01-01T00:00:00"]), "missing_dates_valid"),
    "field-shape": (lambda p: set_attrs(p, field_shape=[NG + 1]), "shape_consistent"),
    "shape-attr": (lambda p: set_attrs(p, shape=[NT, 9, 1, NG]), "shape_consistent"),
    "lat-out-of-range": (lambda p: set_array(p, "latitudes", np.linspace(80, 95, NG)), "coordinates_valid"),
    "varmeta-unknown-variable": (lambda p: _varmeta_extra(p), "variables_metadata_consistent"),
    "constant-fields": (_constant_fields, "variables_metadata_consistent"),
    "dates-nat": (lambda p: set_array(p, "dates", _with(_dates(p), 2, np.datetime64("NaT", "s"))), "dates_valid"),
    "lat-not-finite":
    (lambda p: set_array(p, "latitudes", _with(np.linspace(45, 48, NG), 7, np.nan)), "coordinates_valid"),
    "lat-shape": (lambda p: set_array(p, "latitudes", np.linspace(45, 48, NG - 1)), "shape_consistent"),
    "stats-shape": (lambda p: set_array(p, "mean", np.zeros(len(VARIABLES) - 1)), "shape_consistent"),
    "dates-count": (_drop_last_date, "shape_consistent"),
    # Only visible with a spec or a reference:
    "latlon-swap": (_swap_latlon, None),
    "lat-offset": (lambda p: set_array(p, "latitudes", np.linspace(45.5, 48.5, NG)), None),
    "varmeta-level": (lambda p: _varmeta(p, "T_850", "levelist", 800), None),
    "varmeta-param": (lambda p: _varmeta(p, "T_2M", "param", "TD_2M"), None),
    "missing-dates": (lambda p: set_attrs(p, missing_dates=["2020-03-01T03:00:00"]), None),
    "allow-nans": (lambda p: set_attrs(p, allow_nans=True), None),
    "resolution": (lambda p: set_attrs(p, resolution="1km"), None),
    "extra-attr": (lambda p: set_attrs(p, extra_note="hello"), None),
}

# fault -> checks expected to report an ERROR with the full spec (and the reference)
WITH_SPEC = {
    "latlon-swap": {"grid_matches_spec", "matches_reference"},
    "lat-offset": {"grid_matches_spec", "matches_reference"},
    "varmeta-level": {"variables_metadata_match_spec"},
    "varmeta-param": {"variables_metadata_match_spec"},
    "missing-dates": {"flags_match_spec"},
    "allow-nans": {"flags_match_spec"},
    "resolution": {"grid_matches_spec", "matches_reference"},
    "dates-attr": {"date_attrs_consistent", "matches_reference"},
    "frequency": {"frequency_consistent", "matches_reference"},
    "extra-attr": set(),  # a WARNING from matches_reference
}


def _varmeta_extra(p: Path) -> None:
    metadata = json.loads((p / ".zattrs").read_text())["variables_metadata"]
    metadata["T_2M_COPY"] = {"mars": {"param": "T_2M"}}
    set_attrs(p, variables_metadata=metadata)


@pytest.fixture
def spec_dir(tmp_path):
    """A full spec and the reference meta file of the clean dataset."""
    d = tmp_path / "checks"
    d.mkdir()
    (d / "spec.yaml").write_text(SPEC)
    save_meta(fingerprint(Dataset(build_dataset(tmp_path / "reference.zarr"))), d / "reference.meta.yaml")
    return d


def _findings(path: Path, spec: Spec):
    return run(CheckContext(Dataset(path), spec), bees=["meta"]).findings


def _error_checks(path: Path, spec: Spec) -> set[str]:
    return {f.check for f in _findings(path, spec) if f.severity == Severity.ERROR}


def test_clean_with_full_spec(dataset_path, spec_dir):
    findings = _findings(dataset_path, load_spec(spec_dir / "spec.yaml"))
    assert [f for f in findings if f.severity >= Severity.WARNING or f.skipped] == []
    [summary] = [f for f in findings if f.check == "matches_reference"]
    assert "compared with the reference" in summary.message and "differs" not in summary.message


@pytest.mark.parametrize("fault", FAULTS)
def test_fault_found_without_spec(fault, dataset_path):
    mutate, expected = FAULTS[fault]
    mutate(dataset_path)
    errors = _error_checks(dataset_path, Spec())
    assert errors == ({expected} if expected else set())


@pytest.mark.parametrize("fault", WITH_SPEC)
def test_fault_found_with_spec(fault, dataset_path, spec_dir):
    FAULTS[fault][0](dataset_path)
    spec = load_spec(spec_dir / "spec.yaml")
    assert _error_checks(dataset_path, spec) == WITH_SPEC[fault]
    warnings_ = {f.check for f in _findings(dataset_path, spec) if f.severity == Severity.WARNING}
    assert "matches_reference" in warnings_ | WITH_SPEC[fault]


def test_findings_are_precise(dataset_path, spec_dir):
    _swap_dates(dataset_path)
    [f] = [f for f in _findings(dataset_path, Spec()) if f.check == "dates_valid"]
    assert f.time == "2020-03-01T03:00:00" and "index 4" in f.message
    _varmeta(dataset_path, "T_850", "levelist", 800)
    spec = load_spec(spec_dir / "spec.yaml")
    [f] = [f for f in _findings(dataset_path, spec) if f.check == "variables_metadata_match_spec"]
    assert f.variable == "T_850" and "800" in f.message and "850" in f.message


def test_extra_variable(tmp_path, spec_dir):
    extra = build_dataset(tmp_path / "extra.zarr", variables=VARIABLES + ["T_2M_COPY"])
    errors = _error_checks(extra, load_spec(spec_dir / "spec.yaml"))
    assert errors == {"variables_match_spec", "matches_reference"}


def test_extra_timesteps(tmp_path, spec_dir):
    longer = build_dataset(tmp_path / "longer.zarr", nt=NT + 2)
    errors = _error_checks(longer, load_spec(spec_dir / "spec.yaml"))
    assert errors == {"dates_match_spec", "matches_reference"}


def test_units_from_spec(dataset_path, write_spec):
    spec = load_spec(write_spec("variables: {T_2M: {min: 180, max: 340, units: K}}\n"))
    [f] = [f for f in _findings(dataset_path, spec) if f.check == "variables_metadata_match_spec"]
    assert f.severity == Severity.WARNING and "no units for 1 variables" in f.message
    metadata = json.loads((dataset_path / ".zattrs").read_text())["variables_metadata"]
    metadata["T_2M"]["units"] = "degC"
    set_attrs(dataset_path, variables_metadata=metadata)
    [f] = [f for f in _findings(dataset_path, spec) if f.check == "variables_metadata_match_spec"]
    assert f.severity == Severity.ERROR and f.variable == "T_2M"


def test_many_findings_are_capped(dataset_path):
    set_array(dataset_path, "dates", _dates(dataset_path)[::-1].copy())
    findings = [f for f in _findings(dataset_path, Spec()) if f.check == "dates_valid"]
    assert len(findings) == NT - 1  # below the cap
    metadata = {"T_2M": {}}
    set_attrs(dataset_path, variables_metadata=metadata)
    lacking = [f for f in _findings(dataset_path, Spec()) if f.check == "variables_metadata_consistent"]
    assert len(lacking) == len(VARIABLES) - 1


@pytest.mark.parametrize("text, step", [("1h", "1h"), ("6h", "6h"), ("1d", "1d"), ("PT6H", "6h"), ("P1D", "1d"),
                                        ("30m", "30m"), (3, "3h"), ("PT1H30M", "90m")])
def test_frequency(text, step):
    assert format_frequency(parse_frequency(text)) == step


def test_bad_frequency(dataset_path):
    with pytest.raises(ValueError):
        parse_frequency("hourly")
    set_attrs(dataset_path, frequency="hourly")
    assert _error_checks(dataset_path, Spec()) == {"frequency_consistent"}
    set_attrs(dataset_path, frequency=None)
    assert {f.check
            for f in _findings(dataset_path, Spec()) if f.severity == Severity.WARNING} == {"frequency_consistent"}


def test_invalid_variable_entries(write_spec):
    with pytest.raises(ValueError, match="scalar"):
        load_spec(write_spec("variables: {T_2M: {param: [a, b]}}\n"))
    with pytest.raises(ValueError, match="min 300.0 > max 200.0"):
        load_spec(write_spec("variables: {T_2M: {min: 300, max: 200}}\n"))
    with pytest.raises(ValueError):
        load_spec(write_spec("variables: {T_2M: {min: cold}}\n"))


def test_templates_load(tmp_path):
    """The templates, copied as the README says, form a valid spec."""
    templates = Path(__file__).resolve().parents[1] / "templates"
    (tmp_path / "spec.yaml").write_text((templates / "spec.example.yaml").read_text())
    (tmp_path / "data.ref.yaml").write_text("format: varda-hive-ref/1\n")
    spec = load_spec(tmp_path / "spec.yaml")
    assert spec.bbox is not None and spec.reference_policy.ignore
    assert spec.ranges() and spec.expected_metadata() and spec.constant_field_names()
    # the anchors share one range
    assert spec.ranges()["U_10M"] == spec.ranges()["V_10M"]


def test_constant_fields_both_directions(dataset_path):
    """FIS is flagged constant_in_time but no longer listed; T_2M is listed but not flagged."""
    _constant_fields(dataset_path)
    findings = [f for f in _findings(dataset_path, Spec()) if f.check == "variables_metadata_consistent"]
    assert {(f.variable, f.severity) for f in findings} == {("FIS", Severity.ERROR), ("T_2M", Severity.ERROR)}


def test_duplicate_variable_names(dataset_path):
    set_attrs(dataset_path, variables=["FIS", "T_2M", "T_2M", "U_10M", "V_10M"])
    findings = [f for f in _findings(dataset_path, Spec()) if f.check == "shape_consistent"]
    assert [f.message for f in findings] == ["duplicate names in `variables`: ['T_2M']"]


def test_metadata_warnings(dataset_path):
    """Things that are suspicious but not wrong: absent attributes, coordinates that look like radians."""
    set_attrs(dataset_path, start_date=None)
    set_array(dataset_path, "latitudes", np.linspace(0.78, 0.84, NG))
    set_array(dataset_path, "longitudes", np.linspace(0.08, 0.19, NG))
    warnings_ = {(f.check, f.message) for f in _findings(dataset_path, Spec()) if f.severity == Severity.WARNING}
    assert ("date_attrs_consistent", "no `start_date` attribute") in warnings_
    assert {m
            for c, m in warnings_ if c == "coordinates_valid"} == {
                "all `latitudes` within [-3.142, 3.142]: are they in radians?",
                "all `longitudes` within [-6.283, 6.283]: are they in radians?",
            }


def test_malformed_attributes_are_findings_not_crashes(dataset_path):
    """A dataset fault is a precise finding, not a "check crashed" error."""
    set_attrs(dataset_path, start_date="soon", missing_dates=["never"])
    findings = _findings(dataset_path, Spec())
    assert not [f for f in findings if "crashed" in f.message]
    errors = {(f.check, f.message) for f in findings if f.severity == Severity.ERROR}
    assert ("date_attrs_consistent", "`start_date` attribute 'soon' is not a date") in errors
    assert ("missing_dates_valid", "`missing_dates` entry 'never' is not a date") in errors


def test_missing_variables_attribute(dataset_path):
    set_attrs(dataset_path, variables=None)
    findings = run(CheckContext(Dataset(dataset_path), Spec()), bees=["meta", "stats"]).findings
    [f] = [f for f in findings if f.check == "shape_consistent"]
    assert f.severity == Severity.ERROR and "no `variables` attribute" in f.message
    # checks that need the names fail with that reason, not a bare KeyError
    crashed = [f for f in findings if "crashed" in f.message]
    assert crashed and all("dataset has no `variables` attribute" in f.message for f in crashed)


def test_long_list_of_findings_is_capped(tmp_path):
    path = build_dataset(tmp_path / "long.zarr", nt=15)
    set_array(path, "dates", _dates(path)[::-1].copy())
    findings = [f for f in _findings(path, Spec()) if f.check == "dates_valid"]
    assert len(findings) == 11 and findings[-1].message == "... and 4 more unordered dates"


def test_variables_metadata_entry_or_key_missing(dataset_path, write_spec):
    spec = load_spec(write_spec("variables: {T_2M: {param: T_2M, unit: K}, W_SO: {param: W_SO}}\n"))
    findings = [f for f in _findings(dataset_path, spec) if f.check == "variables_metadata_match_spec"]
    assert {(f.variable, f.message.split(",")[0])
            for f in findings} == {
                ("T_2M", "`unit` missing from `variables_metadata`"),
                ("W_SO", "no `variables_metadata` entry to compare with the spec"),
            }


def test_missing_dates_with_space_separator(dataset_path):
    """`missing_dates` written as `2020-03-01 03:00:00` are still recognised as dates of the dataset."""
    set_attrs(dataset_path, missing_dates=["2020-03-01 03:00:00"])
    assert Dataset(dataset_path).missing_indices == {3}
    assert _error_checks(dataset_path, Spec()) == set()
