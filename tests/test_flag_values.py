# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flag values: spec field, stats-bee sample (flag_mask, summarise, flag_share), physical_ranges."""
import copy

import numpy as np
from pydantic import ValidationError
import pytest
import zarr

from varda_hive.bees.stats.statsfile import (SUMMARY, Sample, compare_stats, compute_sample, compute_stats, flag_mask,
                                             summarise)
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run
from varda_hive.honeycomb.spec import VariableSpec, load_spec

from conftest import NG, NT, VARIABLES, build_dataset

COL = {c: i for i, c in enumerate(SUMMARY)}


# spec
def _var_spec(**var):
    return Spec(variables={"CIN_ML": var})


def test_flag_values_typed_not_metadata():
    v = VariableSpec(min=0.0, max=10000.0, flag_values=[-999.9], param="CIN_ML")
    assert v.flag_values == [-999.9]
    assert v.metadata == {"param": "CIN_ML"}
    assert _var_spec(min=0.0, max=1.0, flag_values=[-1.0]).expected_metadata() == {}


def test_flag_inside_range_rejected():
    with pytest.raises(ValidationError, match="inside the physical range"):
        VariableSpec(min=0.0, max=10.0, flag_values=[5.0])
    with pytest.raises(ValidationError, match="inside the physical range"):
        VariableSpec(min=0.0, flag_values=[100.0])


def test_flag_without_bounds_rejected():
    with pytest.raises(ValidationError, match="flag_values needs min or max"):
        VariableSpec(flag_values=[-999.9])


def test_ranges_carry_flags():
    r = _var_spec(min=0.0, max=10000.0, flag_values=[-999.9]).ranges()["CIN_ML"]
    assert r.flag_values == [-999.9]
    assert _var_spec(min=0.0, max=1.0).ranges()["CIN_ML"].flag_values == []


def test_load_spec_yaml(tmp_path):
    f = tmp_path / "spec.yaml"
    f.write_text("variables:\n  CIN_ML: {min: 0.0, max: 10000.0, units: J kg-1, flag_values: [-999.9]}\n")
    spec = load_spec(f)
    assert spec.ranges()["CIN_ML"].flag_values == [-999.9]
    assert spec.expected_metadata() == {}


# stats sample
def _step():
    """(2 variables, 1, 10): variable 0 has three -999.9 flags, variable 1 none."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 2, size=(2, 1, 10)).astype("f4")
    x[0, 0, :3] = np.float32(-999.9)
    return x


def test_flag_mask_matches_in_stored_dtype():
    values = np.array([-999.9, 1.0, 2.0], dtype="f4")
    assert float(values[0]) != -999.9  # float32 is not the python float
    assert flag_mask(values, [-999.9]).tolist() == [True, False, False]
    assert not flag_mask(values, []).any()


def test_summarise_flags_only_mixed_none():
    x = _step()
    for flags, n_flag in (([[-999.9], [-999.9]], (3, 0)), ([[-999.9], None], (3, 0)), ([None,
                                                                                        None], (0, 0)), (None, (0, 0))):
        out = summarise(x, None, flags)
        assert out.shape == (2, len(SUMMARY))
        assert out[:, COL["n_flag"]].tolist() == list(n_flag)
        # minimum, maximum and mean still include the flags
        assert out[0, COL["minimum"]] == np.float32(-999.9)
        assert out[0, COL["mean"]] < 0
        if flags and flags[0]:
            assert out[0, COL["valid_minimum"]] == x[0, 0, 3:].min()
            assert out[0, COL["valid_maximum"]] == x[0, 0, 3:].max()
        else:
            assert out[0, COL["valid_minimum"]] == out[0, COL["minimum"]]
            assert out[0, COL["valid_maximum"]] == out[0, COL["maximum"]]
        assert out[1, COL["valid_minimum"]] == out[1, COL["minimum"]]
        assert out[1, COL["valid_maximum"]] == out[1, COL["maximum"]]


def test_summarise_all_flags_and_subsample():
    x = np.full((1, 1, 10), -999.9, dtype="f4")
    out = summarise(x, np.array([0, 1]), [[-999.9]])
    assert out[0, COL["n_flag"]] == 10  # full domain, not the points
    assert np.isnan(out[0, COL["valid_minimum"]]) and np.isnan(out[0, COL["valid_maximum"]])


def _flagged_dataset(tmp_path):
    path = build_dataset(tmp_path / "flags.zarr")
    data = zarr.open_group(str(path), mode="r+")["data"]
    block = data[:]
    block[:, 1, 0, :10] = np.float32(-999.9)  # T_2M: 10 of 50 points on every date
    data[:] = block
    return path


def _stats_spec(flags):
    entry = {"min": 200, "max": 400}
    return Spec(variables={
        "FIS": None,
        "T_2M": entry | ({
            "flag_values": [-999.9]
        } if flags else {})
    },
                sampling={"n_dates": 4})


def test_aggregate_flag_share_only_for_flagged_variables(tmp_path):
    ds = Dataset(_flagged_dataset(tmp_path))
    sample = compute_sample(ds, _stats_spec(True))
    assert sample.flags == {"T_2M": [-999.9]}
    agg = sample.aggregate()
    assert agg["T_2M"]["flag_share"] == 10 / 50
    assert all("flag_share" not in v for n, v in agg.items() if n != "T_2M")
    assert agg["T_2M"]["minimum"] == float(np.float32(-999.9))
    per_date = sample.per_date("T_2M")
    assert set(per_date) == set(SUMMARY)
    assert per_date["n_flag"] == [10] * len(sample.dates)


def test_stats_file_without_flags_has_no_flag_share(tmp_path):
    ds = Dataset(_flagged_dataset(tmp_path))
    sample = compute_sample(ds, _stats_spec(False))
    assert sample.flags == {}
    stats = compute_stats(ds, _stats_spec(False))
    assert all("flag_share" not in v for v in stats["sample"].values())


def test_changed_flag_share_is_a_warning(tmp_path):
    ds = Dataset(_flagged_dataset(tmp_path))
    ref = compute_stats(ds, _stats_spec(True))
    new = copy.deepcopy(ref)
    new["sample"]["T_2M"]["flag_share"] = 0.5
    findings = [f for f in compare_stats(new, ref, atol_stdev=0.0).findings if "flag_share" in f.message]
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARNING
    assert isinstance(compute_sample(ds, _stats_spec(True)), Sample)


# physical_ranges
NAMES = VARIABLES + ["CIN_ML"]
CIN = NAMES.index("CIN_ML")
FLAG_DATE = NT - 1  # all flags
VARS = {"min": 0.0, "max": 10000.0}


def _store(path, data):
    """Write the CIN_ML data and recompute the stored statistics from all of it."""
    root = zarr.open_group(str(path), mode="r+")
    root["data"][:, CIN] = data
    v = root["data"][:].astype("f8")
    for key, fn in (("mean", np.mean), ("stdev", np.std), ("minimum", np.min), ("maximum", np.max)):
        root[key][:] = fn(v, axis=(0, 2, 3))


@pytest.fixture
def flagged(tmp_path):
    path = build_dataset(tmp_path / "flag.zarr", variables=NAMES)
    rng = np.random.default_rng(1)
    data = rng.uniform(0, 6000, size=(NT, 1, NG)).astype("f4")
    data[:, :, :NG // 2] = np.float32(-999.9)
    data[FLAG_DATE] = np.float32(-999.9)
    _store(path, data)
    return path


def _ranges_spec(flags=True, **extra):
    cin = {**VARS, **({"flag_values": [-999.9]} if flags else {})}
    return Spec(variables={"CIN_ML": cin}, sampling={"n_dates": "all"}, **extra)


def _findings(path, spec, select="physical_ranges", severity=Severity.ERROR):
    report = run(CheckContext(Dataset(path), spec), bees=["stats"], select=[select])
    return [f for f in report.findings if f.severity == severity and not f.skipped]


def test_flags_are_not_errors(flagged):
    assert _findings(flagged, _ranges_spec()) == []


def test_flag_share_info(flagged):
    [f] = [f for f in _findings(flagged, _ranges_spec(), severity=Severity.INFO) if f.variable == "CIN_ML"]
    expected = (NT * NG // 2 + (NG - NG // 2)) / (NT * NG)
    assert f.message == f"CIN_ML: {expected:.1%} flag values (-999.9) on {NT} sampled dates"
    assert f.details["flag_values"] == [-999.9]
    assert f.details["n_flag"] == NT * NG // 2 + (NG - NG // 2)
    assert f.details["flag_share"] == pytest.approx(expected)


def test_all_flag_date_is_not_an_error(flagged):
    errors = _findings(flagged, _ranges_spec())
    assert not [f for f in errors if f.time == "2020-03-01T07:00:00"]


def test_non_flag_out_of_range_value_is_an_error(flagged):
    root = zarr.open_group(str(flagged), mode="r+")
    root["data"][2, CIN, 0, NG - 1] = -500.0
    errors = _findings(flagged, _ranges_spec())
    [f] = [f for f in errors if f.time is not None]
    assert (f.variable, f.time) == ("CIN_ML", "2020-03-01T02:00:00")
    assert f.details["minimum"] == -500.0


def test_stored_minimum_flag_is_not_an_error(flagged):
    stored = zarr.open_group(str(flagged), mode="r")["minimum"][CIN]
    assert stored == pytest.approx(-999.9, abs=1e-3)
    assert _findings(flagged, _ranges_spec()) == []


def test_stored_minimum_non_flag_out_of_range_is_an_error(flagged):
    zarr.open_group(str(flagged), mode="r+")["minimum"][CIN] = -500.0
    [f] = _findings(flagged, _ranges_spec())
    assert f.time is None and "stored minimum/maximum" in f.message


def test_without_flag_values_the_flags_are_errors(flagged):
    errors = _findings(flagged, _ranges_spec(flags=False))
    assert {f.time for f in errors if f.time} == {f"2020-03-01T0{i}:00:00" for i in range(NT)}
    assert any("stored minimum/maximum" in f.message for f in errors)


def test_statistics_consistent_is_clean(flagged):
    spec = _ranges_spec()
    assert _findings(flagged, spec, "statistics_consistent") == []
    assert _findings(flagged, spec, "statistics_consistent", Severity.WARNING) == []


def test_stored_maximum_none_does_not_crash(flagged):
    zarr.open_group(str(flagged), mode="r+")["maximum"][CIN] = np.nan
    assert _findings(flagged, _ranges_spec()) == []
    zarr.open_group(str(flagged), mode="r+")["minimum"][CIN] = np.nan
    assert _findings(flagged, _ranges_spec()) == []
