# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""stats-bee, meta-bee and spot-bee on the Varda-tiny fault scenarios (MRB-1032), on a small real dataset.

`tests/data/varda-tiny-subset.zarr` is a 0.5 MB subset of Varda-tiny (see
`tests/data/make_varda_tiny_subset.py`). Each scenario copies it to `tmp_path` and applies the
faults of the Varda-tiny generator's manifest with the primitives of `tests/faults.py`. The faults
are at time index 28 (index 100 of Varda-tiny), so the spec always samples it and the step before.
spot-bee compares with `tests/data/grib/`, the original GRIB messages cut to the same points.
"""
import shutil
from pathlib import Path

import pytest
import yaml

from faults import apply
from varda_hive.bees.meta.metafile import fingerprint, save_meta
from varda_hive.bees.stats.statsfile import compute_stats, save_stats
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, load_spec, run

DATA = Path(__file__).parent / "data"
CLEAN = DATA / "varda-tiny-subset.zarr"
T = 28  # time index of the faults
# Loose physical ranges: they catch unit errors and garbage, not unusual weather.
RANGES = """\
variables:
  T_2M:        {min: 200.0, max: 330.0, units: K}
  T_850:       {min: 200.0, max: 320.0, units: K}
  U_10M:       {min: -80.0, max: 80.0, units: m s-1}
  V_10M:       {min: -80.0, max: 80.0, units: m s-1}
  PS:          {min: 50000.0, max: 110000.0, units: Pa}
  TOT_PREC_1H: {min: 0.0, max: 0.3, units: m}
  CLCT:        {min: 0.0, max: 105.0, units: "%"}   # ICON slightly exceeds 100
"""

# +50 K in Varda-tiny, where about 1 in 10 spiked points ends above 330 K; on the 5 spiked points of the
# subset none might, so +80 K puts all of them out of range.
SPIKE = {"variables": ["T_2M"], "index": T, "mode": "spike", "fraction": 0.01, "amplitude": 80.0}
VARMETA_LEVEL = ("set_attr", {"path": ["variables_metadata", "T_850", "mars", "levelist"], "value": 800})
STATS_MEAN = ("scale_statistic", {"variable": "T_2M", "stat": "mean", "factor": 1.5})
STATS_MINMAX = ("swap_statistics", {"variable": "PS", "a": "minimum", "b": "maximum"})
EXTRA_VARIABLE = ("add_variable_copy", {"name": "T_2M_COPY", "copy_of": "T_2M"})
EXTRA_TIMESTEPS = ("append_timesteps", {"count": 24})
FREQUENCY = ("set_attr", {"path": ["frequency"], "value": "3h"})
CONSTANT_FIELDS = ("edit_constant_fields", {"remove": ["FIS"], "add": ["T_2M"]})


def _step(variables, mode, **params):
    return ("corrupt_timestep", {"variables": variables, "index": T, "mode": mode, **params})


# The scenarios of the Varda-tiny manifest, as (primitive, params) pairs.
SCENARIOS = {
    "m-varmeta-level": [VARMETA_LEVEL],
    "m-varmeta-param": [("set_attr", {
        "path": ["variables_metadata", "T_2M", "mars", "param"],
        "value": "TD_2M"
    })],
    "m-variables-swap": [("swap_variable_names", {
        "a": "U_10M",
        "b": "V_10M"
    })],
    "m-constant-fields": [CONSTANT_FIELDS],
    "m-frequency": [FREQUENCY],
    "m-dates-attr": [("shift_date_attrs", {
        "keys": ["start_date", "end_date"],
        "hours": 24
    })],
    "m-missing-dates": [("add_missing_date", {
        "index": T
    })],
    "m-stats-dates": [("set_attr", {
        "path": ["statistics_end_date"],
        "value": "2020-04-15T23:00:00"
    })],
    "m-field-shape": [("set_attr", {
        "path": ["field_shape"],
        "value": [504]
    })],
    "m-latlon-offset": [("perturb_coordinates", {
        "offset": {
            "latitudes": 0.5
        }
    })],
    "m-latlon-swap": [("perturb_coordinates", {
        "swap": True
    })],
    "m-dates-swap": [("corrupt_dates", {
        "mode": "swap",
        "index": T
    })],
    "m-dates-duplicate": [("corrupt_dates", {
        "mode": "duplicate",
        "index": T
    })],
    "a-extra-variable": [EXTRA_VARIABLE],
    "a-extra-level": [("add_pressure_level", {
        "param": "T",
        "level": 800,
        "between": [700, 850]
    })],
    "a-extra-timesteps": [EXTRA_TIMESTEPS],
    "a-extra-attr": [("set_attr", {
        "path": ["extra_note"],
        "value": "attribute not present in the reference"
    })],
    "d-stats-mean": [STATS_MEAN],
    "d-stats-stdev": [("set_statistic", {
        "variable": "T_850",
        "stat": "stdev",
        "value": 0.0
    })],
    "d-stats-minmax": [STATS_MINMAX],
    "d-stats-subtle": [("scale_statistic", {
        "variable": "T_2M",
        "stat": "mean",
        "factor": 1.001
    })],
    "d-shift-consistent": [("offset_variable", {
        "variable": "T_850",
        "offset": 2.0
    })],
    "d-step-nan": [_step(["T_2M"], "nan")],
    "d-step-zero": [_step(["T_850"], "zero")],
    "d-step-spike": [("corrupt_timestep", SPIKE)],
    "d-step-spike-stats-updated": [("corrupt_timestep", {
        **SPIKE, "update_statistics": True
    })],
    "d-step-unit": [_step(["T_2M"], "scale", offset=-273.15)],
    "d-step-swap": [_step(["U_10M"], "swap", other="V_10M")],
    "d-step-stale": [_step(["TOT_PREC_1H"], "stale")],
    "d-step-shift": [_step(["T_2M"], "roll", shift=16)],  # 1000 of 30676 points in Varda-tiny
    "d-step-allvars-nan": [_step("all", "nan")],
    "d-step-subtle": [_step(["T_850"], "scale", factor=1.0005)],
    "d-step-constant": [_step(["FIS"], "scale", offset=500.0)],
    "x-meta-data": [VARMETA_LEVEL, STATS_MEAN, ("corrupt_timestep", SPIKE)],
    "x-extra-data": [EXTRA_VARIABLE, STATS_MINMAX],
    "x-meta-extra": [FREQUENCY, EXTRA_TIMESTEPS],
    "x-everything":
    [CONSTANT_FIELDS, EXTRA_VARIABLE, STATS_MEAN,
     _step(["T_2M"], "nan"),
     _step(["U_10M"], "swap", other="V_10M")],
}

# scenario -> stats checks expected to report an ERROR
STATS_DETECTS = {
    "d-stats-mean": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-stdev": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-minmax": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-subtle": {"matches_reference_stats"},  # below the sampling error; caught exactly with n_dates: all
    "d-shift-consistent": {"matches_reference_stats"},  # self-consistent
    "d-step-spike-stats-updated": {"physical_ranges", "matches_reference_stats"},
    "d-step-nan": {"finite_values", "matches_reference_stats"},
    "d-step-allvars-nan": {"finite_values", "matches_reference_stats"},
    "d-step-zero": {"physical_ranges", "matches_reference_stats"},
    "d-step-spike": {"physical_ranges", "matches_reference_stats"},
    "d-step-unit": {"physical_ranges", "matches_reference_stats"},
    "d-step-swap": {"matches_reference_stats"},
    "d-step-stale": {"matches_reference_stats"},
    "d-step-subtle": {"matches_reference_stats"},
    "d-step-constant": {"constant_fields", "matches_reference_stats"},
    "m-stats-dates": {"statistics_window_valid"},
    # Rolled in space: same statistics over the full domain, but other values on the sampled points.
    # Only found with a point subset (n_points below the number of grid points), not with `n_points: all`.
    "d-step-shift": {"matches_reference_stats"},
    # Metadata and additional faults that stats-bee catches as a side effect (most others are meta-bee's).
    "m-constant-fields": {"constant_fields"},
    "m-variables-swap": {"matches_reference_stats"},
    "m-dates-swap": {"matches_reference_stats"},
    "m-dates-duplicate": {"matches_reference_stats"},
    "m-missing-dates": {"matches_reference_stats"},  # the listed date is no longer sampled
    "a-extra-variable": {"matches_reference_stats"},
    "a-extra-level": {"matches_reference_stats"},
    "x-meta-data": {"statistics_consistent", "physical_ranges", "matches_reference_stats"},
}

# scenario -> meta checks expected to report an ERROR, with a full spec and the clean dataset's meta file
META_DETECTS = {
    "m-varmeta-level": {"variables_metadata_match_spec"},
    "m-varmeta-param": {"variables_metadata_match_spec"},
    "m-variables-swap": {"matches_reference"},
    "m-constant-fields": {"variables_metadata_consistent"},
    "m-frequency": {"frequency_consistent", "matches_reference"},
    "m-dates-attr": {"date_attrs_consistent", "matches_reference"},
    "m-missing-dates": {"flags_match_spec"},
    "m-field-shape": {"shape_consistent", "matches_reference"},
    "m-latlon-offset": {"grid_matches_spec", "matches_reference"},
    "m-latlon-swap": {"grid_matches_spec", "matches_reference"},
    "m-dates-swap": {"dates_valid", "matches_reference"},
    "m-dates-duplicate": {"dates_valid", "matches_reference"},
    "a-extra-variable": {"variables_match_spec", "matches_reference"},
    "a-extra-level": {"variables_match_spec", "matches_reference"},
    "a-extra-timesteps": {"dates_match_spec", "matches_reference"},
    "x-meta-data": {"variables_metadata_match_spec"},
    "x-meta-extra": {"frequency_consistent", "dates_match_spec", "matches_reference"},
    "x-everything": {"variables_metadata_consistent", "variables_match_spec", "matches_reference"},
}
# scenario -> meta checks expected to report a WARNING (differences only the reference sees)
META_WARNS = {
    "m-stats-dates": {"matches_reference"},  # the ERROR is stats-bee's statistics_window_valid
    "a-extra-attr": {"matches_reference"},
}

# scenario -> variables spot-bee finds differing from the GRIB source
SPOT_DETECTS = {
    "d-step-nan": {"T_2M"},
    "d-step-zero": {"T_850"},
    "d-step-spike": {"T_2M"},
    "d-step-unit": {"T_2M"},
    "d-step-swap": {"U_10M", "V_10M"},
    "d-step-stale": {"TOT_PREC_1H"},
    "d-step-subtle": {"T_850"},
    "d-step-shift": {"T_2M"},
    "d-shift-consistent": {"T_850"},
    "m-variables-swap": {"U_10M", "V_10M"},
    "m-varmeta-param": {"T_2M"},  # compared with TD_2M
    "a-extra-attr": set(),
    "d-stats-mean": set(),  # statistics only: stats-bee's job
    "d-step-constant": set(),  # FIS is a constant field, not in the time-indexed source
}


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stats")
    include = ", ".join(str(d) for d in Dataset(CLEAN).dates[[T - 1, T]])
    (tmp / "spec.yaml").write_text(RANGES + f"sampling: {{n_dates: 8, n_points: 200, include_dates: [{include}]}}\n")
    save_stats(compute_stats(Dataset(CLEAN), load_spec(tmp / "spec.yaml")), tmp / "reference.json")
    (tmp / "spec.yaml").write_text((tmp / "spec.yaml").read_text() + "reference: reference.json\n")
    return load_spec(tmp / "spec.yaml")


@pytest.fixture(scope="module")
def meta_spec(tmp_path_factory):
    """A full spec written from the clean dataset (in a real repo it comes from the recipe)."""
    tmp = tmp_path_factory.mktemp("meta")
    ds = Dataset(CLEAN)
    attrs = ds.attrs
    save_meta(fingerprint(ds), tmp / "reference.meta.yaml")
    spec = {
        "variables": {
            v: {
                k: m["mars"][k]
                for k in ("param", "levtype", "levelist") if k in m["mars"]
            }
            for v, m in attrs["variables_metadata"].items()
        },
        "start_date": attrs["start_date"],
        "end_date": attrs["end_date"],
        "frequency": attrs["frequency"],
        "grid_points": int(ds.data.shape[-1]),
        "resolution": attrs["resolution"],
        "bbox": {
            "lat_min": float(ds.latitudes.min()),
            "lat_max": float(ds.latitudes.max()),
            "lon_min": float(ds.longitudes.min()),
            "lon_max": float(ds.longitudes.max()),
        },
        "allow_nans": attrs["allow_nans"],
        "missing_dates": [],
        "reference": "reference.meta.yaml",
    }
    (tmp / "spec.yaml").write_text(yaml.safe_dump(spec))
    return load_spec(tmp / "spec.yaml")


def _inject(scenario: str, tmp_path: Path) -> Path:
    path = Path(shutil.copytree(CLEAN, tmp_path / f"{scenario}.zarr"))
    return apply(path, SCENARIOS[scenario])


def _checks(path: Path, spec, bee: str, severity: Severity = Severity.ERROR) -> set[str]:
    report = run(CheckContext(Dataset(path), spec), bees=[bee])
    return {f.check for f in report.findings if f.severity == severity}


def test_scenarios_cover_the_tables():
    assert set(STATS_DETECTS) | set(META_DETECTS) | set(META_WARNS) | set(SPOT_DETECTS) <= set(SCENARIOS)


def test_clean_has_no_errors(spec):
    assert _checks(CLEAN, spec, "stats") == set()


@pytest.mark.parametrize("scenario", sorted(STATS_DETECTS))
def test_fault_detected(scenario, spec, tmp_path):
    assert STATS_DETECTS[scenario] <= _checks(_inject(scenario, tmp_path), spec, "stats")


def test_meta_clean(meta_spec):
    assert _checks(CLEAN, meta_spec, "meta") == set()
    assert _checks(CLEAN, meta_spec, "meta", Severity.WARNING) == set()


@pytest.mark.parametrize("scenario", sorted(META_DETECTS))
def test_meta_fault_detected(scenario, meta_spec, tmp_path):
    assert META_DETECTS[scenario] <= _checks(_inject(scenario, tmp_path), meta_spec, "meta")


@pytest.mark.parametrize("scenario", sorted(META_WARNS))
def test_meta_fault_warned(scenario, meta_spec, tmp_path):
    assert META_WARNS[scenario] <= _checks(_inject(scenario, tmp_path), meta_spec, "meta", Severity.WARNING)


@pytest.mark.parametrize("scenario", sorted(s for s in STATS_DETECTS if s.startswith("d-")))
def test_meta_no_false_positive_on_data_faults(scenario, meta_spec, tmp_path):
    assert _checks(_inject(scenario, tmp_path), meta_spec, "meta") == set()


@pytest.fixture(scope="module")
def spot_spec():
    pytest.importorskip("eccodes")
    return Spec(sampling={
        "n_dates": 3,
        "n_points": "all",
        "include_dates": [str(d) for d in Dataset(CLEAN).dates[[T - 1, T]]]
    },
                tolerances={"source_per_variable": {
                    "TOT_PREC_1H": {
                        "atol": 1e-6
                    }
                }},
                source={
                    "plugin": "varda_hive.sources.grib_index:GribIndexSource",
                    "args": {
                        "root": str(DATA),
                        "indexes": {
                            "ana": "grib-index-ana.db",
                            "fg": "grib-index-fg.db"
                        },
                        "variables": {
                            "TOT_PREC_1H": {
                                "index": "fg",
                                "match": {
                                    "shortName": "TOT_PREC"
                                },
                                "scale": 0.001
                            }
                        },
                    },
                })


def _spot(path: Path, spec: Spec):
    findings = run(CheckContext(Dataset(path), spec), bees=["spot"]).findings
    [summary] = [f for f in findings if f.message.startswith("spot summary")]
    return [f for f in findings if f.severity == Severity.ERROR], summary.details


def test_spot_clean_matches_grib(spot_spec):
    errors, details = _spot(CLEAN, spot_spec)
    assert errors == []
    assert set(details["identical"]) == {"T_2M", "T_700", "T_850", "U_10M", "V_10M", "PS", "TOT_PREC_1H", "CLCT"}
    assert details["within_tolerance"] == {} and details["differ"] == {} and details["unavailable"] == {}
    assert details["not_comparable"] == {"FIS": "constant field, not in the time-indexed source"}


@pytest.mark.parametrize("scenario", sorted(SPOT_DETECTS))
def test_spot_fault_detected(scenario, spot_spec, tmp_path):
    errors, details = _spot(_inject(scenario, tmp_path), spot_spec)
    assert {f.variable for f in errors} == SPOT_DETECTS[scenario]
    assert set(details["differ"]) == SPOT_DETECTS[scenario]
    if scenario.startswith("d-step"):
        assert {f.time for f in errors} <= {str(Dataset(CLEAN).dates[T])}
