# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The stats-bee on Varda-tiny and its faulty variants (MRB-1032).

Opt-in: `uv run pytest -m faulty`. Skipped when the datasets are not on this machine. Set
HIVE_FAULTY_DIR (the directory with manifest.json) and HIVE_CLEAN_DS to use other copies.
The faults are at time index 100, so the spec always samples it (and 99, for stale steps);
a random sample alone would hit it with probability n_dates / 720.
"""
import json
import os
from pathlib import Path

import pytest

from varda_hive.honeycomb import CheckContext, Dataset, Severity, load_spec, run
from varda_hive.bees.stats.statsfile import compute_stats, save_stats

FAULTY = Path(os.environ.get("HIVE_FAULTY_DIR", "/scratch/mch/huppd/ich1-analysis-tiny-dataset/faulty"))
CLEAN = Path(
    os.environ.get(
        "HIVE_CLEAN_DS", "/scratch/mch/huppd/varda/ich1-analysis-anemoi-dataset/ml_test/datasets/"
        "varda-kenda-ch1-2km-2020-2020-1h-v1.0-pl13-test.zarr"))
# Loose physical ranges: they catch unit errors and garbage, not unusual weather.
RANGES = """\
variables:
  T_2M:        {min: 200.0, max: 330.0, units: K}
  T_850:       {min: 200.0, max: 320.0, units: K}
  U_10M:       {min: -80.0, max: 80.0, units: m s-1}
  V_10M:       {min: -80.0, max: 80.0, units: m s-1}
  PS:          {min: 50000.0, max: 110000.0, units: Pa}
  TOT_PREC_1H: {min: 0.0, max: 0.3, units: m}
  CLCT:        {min: 0.0, max: 105.0, units: "%"}   # ICON slightly exceeds 100 (103 on varda-tiny)
"""

# scenario -> checks expected to report an ERROR
EXPECTED = {
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
    # Only found with a point subset (the default n_points), not with `n_points: all`.
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

pytestmark = [
    pytest.mark.faulty,
    pytest.mark.skipif(not (FAULTY / "manifest.json").exists() or not CLEAN.exists(),
                       reason="Varda-tiny datasets not available"),
]


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("faulty")
    (tmp / "spec.yaml").write_text(RANGES + "sampling: {n_dates: 20, include_dates: [99, 100]}\n")
    save_stats(compute_stats(Dataset(CLEAN), load_spec(tmp / "spec.yaml")), tmp / "reference.json")
    (tmp / "spec.yaml").write_text((tmp / "spec.yaml").read_text() + "reference: reference.json\n")
    return load_spec(tmp / "spec.yaml")


def _error_checks(path, spec):
    report = run(CheckContext(Dataset(path), spec), bees=["stats"])
    return {f.check for f in report.findings if f.severity == Severity.ERROR}


def _path(scenario):
    return Path(json.loads((FAULTY / "manifest.json").read_text())["datasets"][scenario]["path"])


def test_clean_has_no_errors(spec):
    assert _error_checks(CLEAN, spec) == set()


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_fault_detected(scenario, spec):
    assert EXPECTED[scenario] <= _error_checks(_path(scenario), spec)


# --- meta-bee ---------------------------------------------------------------------------------

# scenario -> meta checks expected to report an ERROR, with a full spec and the clean dataset's meta file
META_EXPECTED = {
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
META_WARNINGS = {
    "m-stats-dates": {"matches_reference"},  # the ERROR is stats-bee's statistics_window_valid
    "a-extra-attr": {"matches_reference"},
}


@pytest.fixture(scope="module")
def meta_spec(tmp_path_factory):
    """A full spec written from the clean dataset (in a real repo it comes from the recipe)."""
    import yaml  # pylint: disable=import-outside-toplevel

    from varda_hive.bees.meta.metafile import fingerprint, save_meta  # pylint: disable=import-outside-toplevel

    tmp = tmp_path_factory.mktemp("meta")
    ds = Dataset(CLEAN)
    attrs = ds.attrs
    variables = {
        v: ({
            k: m["mars"][k]
            for k in ("param", "levtype", "levelist") if k in m["mars"]
        } if "mars" in m else {})
        for v, m in attrs["variables_metadata"].items()
    }
    save_meta(fingerprint(ds), tmp / "reference.meta.yaml")
    spec = {
        "variables": {
            v: variables.get(v, {})
            for v in attrs["variables"]
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


def _meta_checks(path, spec, severity):
    report = run(CheckContext(Dataset(path), spec), bees=["meta"])
    return {f.check for f in report.findings if f.severity == severity}


def test_meta_clean(meta_spec):
    assert _meta_checks(CLEAN, meta_spec, Severity.ERROR) == set()
    assert _meta_checks(CLEAN, meta_spec, Severity.WARNING) == set()


@pytest.mark.parametrize("scenario", sorted(META_EXPECTED))
def test_meta_fault_detected(scenario, meta_spec):
    assert META_EXPECTED[scenario] <= _meta_checks(_path(scenario), meta_spec, Severity.ERROR)


@pytest.mark.parametrize("scenario", sorted(META_WARNINGS))
def test_meta_fault_warned(scenario, meta_spec):
    assert META_WARNINGS[scenario] <= _meta_checks(_path(scenario), meta_spec, Severity.WARNING)


@pytest.mark.parametrize("scenario", sorted(s for s in EXPECTED if s.startswith("d-")))
def test_meta_no_false_positive_on_data_faults(scenario, meta_spec):
    assert _meta_checks(_path(scenario), meta_spec, Severity.ERROR) == set()
