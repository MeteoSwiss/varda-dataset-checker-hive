"""The stats-bee on Varda-tiny and its faulty variants (MRB-1032).

Opt-in: `uv run pytest -m faulty`. Skipped when the datasets are not on this machine.
The faults are at time index 100, so the spec always samples it (and 99, for stale steps);
a random sample alone would hit it with probability n_dates / 720.
"""
import json
from pathlib import Path

import pytest

from varda_hive.honeycomb import CheckContext, Dataset, Severity, load_spec, run
from varda_hive.honeycomb.statsfile import compute_stats, save_stats

FAULTY = Path("/scratch/mch/huppd/ich1-analysis-tiny-dataset/faulty")
CLEAN = Path("/scratch/mch/huppd/varda/ich1-analysis-anemoi-dataset/ml_test/datasets/"
             "varda-kenda-ch1-2km-2020-2020-1h-v1.0-pl13-test.zarr")
RANGES = Path(__file__).resolve().parents[1] / "templates" / "physical_ranges.example.yaml"

# scenario -> checks expected to report an ERROR
EXPECTED = {
    "d-stats-mean": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-stdev": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-minmax": {"statistics_consistent", "matches_reference_stats"},
    "d-stats-subtle": {"matches_reference_stats"},  # below the sampling error; caught exactly with n_dates: all
    "d-shift-consistent": {"matches_reference_stats"},  # self-consistent
    "d-step-spike-stats-updated": {"physical_ranges", "matches_reference_stats"},
    "d-step-nan": {"no_nans", "matches_reference_stats"},
    "d-step-allvars-nan": {"no_nans", "matches_reference_stats"},
    "d-step-zero": {"physical_ranges", "matches_reference_stats"},
    "d-step-spike": {"physical_ranges", "matches_reference_stats"},
    "d-step-unit": {"physical_ranges", "matches_reference_stats"},
    "d-step-swap": {"matches_reference_stats"},
    "d-step-stale": {"matches_reference_stats"},
    "d-step-subtle": {"matches_reference_stats"},
    "d-step-constant": {"constant_fields", "matches_reference_stats"},
}

pytestmark = [
    pytest.mark.faulty,
    pytest.mark.skipif(not (FAULTY / "manifest.json").exists() or not CLEAN.exists(),
                       reason="Varda-tiny datasets not available"),
]


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("faulty")
    (tmp / "spec.yaml").write_text(f"ranges_file: {RANGES}\nsampling: {{n_dates: 20, include_dates: [99, 100]}}\n")
    save_stats(compute_stats(Dataset(CLEAN), load_spec(tmp / "spec.yaml")), tmp / "reference.json")
    (tmp / "spec.yaml").write_text((tmp / "spec.yaml").read_text() + "reference_stats: reference.json\n")
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


@pytest.mark.xfail(reason="rolled field keeps mean/stdev/min/max: needs spot-bee", strict=True)
def test_rolled_field_detected(spec):
    assert _error_checks(_path("d-step-shift"), spec)
