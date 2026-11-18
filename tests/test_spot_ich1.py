# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee with the grib-index source on Varda-tiny (ICH1 test dataset) and its faulty variants.

Opt-in: `uv run pytest -m faulty`. Skipped when the data or eccodes are missing. HIVE_ICH1_ROOT is
the ich1-analysis-anemoi-dataset checkout holding ml_test/ (GRIB archive and grib-index DBs);
HIVE_FAULTY_DIR and HIVE_CLEAN_DS as in test_faulty_tiny.py. The faults are at time index 100.
"""
import json
import os
from pathlib import Path

import pytest

from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run

from test_faulty_tiny import CLEAN, FAULTY

ROOT = Path(os.environ.get("HIVE_ICH1_ROOT", "/scratch/mch/huppd/varda/ich1-analysis-anemoi-dataset"))
INDEXES = {"ana": "ml_test/processed/grib-index-ana.db", "fg": "ml_test/processed/grib-index-fg.db"}

# scenario -> variables expected to differ from the GRIB source
EXPECTED = {
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
    "m-varmeta-param": {"T_2M"},
    "a-extra-attr": set(),
    "d-stats-mean": set(),  # statistics only: stats-bee's job
    "d-step-constant": set(),  # FIS is a constant field, not in the time-indexed source
}

pytestmark = [
    pytest.mark.faulty,
    pytest.mark.skipif(not all((ROOT / p).exists() for p in INDEXES.values()) or not CLEAN.exists(),
                       reason="ICH1 test dataset or grib index not available"),
]


@pytest.fixture(scope="module")
def spec():
    pytest.importorskip("eccodes")
    return Spec(sampling={
        "n_dates": 3,
        "n_points": "all",
        "include_dates": [99, 100]
    },
                tolerances={"source_per_variable": {
                    "TOT_PREC_1H": {
                        "atol": 1e-6
                    }
                }},
                source={
                    "plugin": "varda_hive.sources.grib_index:GribIndexSource",
                    "args": {
                        "root": str(ROOT),
                        "indexes": INDEXES,
                        "variables": {
                            "W_SO": {
                                "index": "fg",
                                "match": {
                                    "shortName": "W_SO",
                                    "level:d": "0.01"
                                }
                            },
                            "TOT_PREC_1H": {
                                "index": "fg",
                                "match": {
                                    "shortName": "TOT_PREC"
                                },
                                "scale": 0.001
                            },
                        },
                        "skip": {
                            "TOT_PREC_6H": "6 h accumulation of TOT_PREC"
                        },
                    },
                })


def _run(path, spec):
    findings = run(CheckContext(Dataset(path), spec), bees=["spot"]).findings
    [summary] = [f for f in findings if f.message.startswith("spot summary")]
    return findings, summary.details


def test_clean_matches_grib(spec):
    findings, details = _run(CLEAN, spec)
    assert [f for f in findings if f.severity == Severity.ERROR] == []
    assert {"T_850", "T_2M", "W_SO", "TOT_PREC_1H", "CAPE_ML"} <= set(details["identical"])
    assert details["within_tolerance"] == {} and details["differ"] == {} and details["unavailable"] == {}
    reasons = details["not_comparable"]
    assert reasons["TOT_PREC_6H"] == "6 h accumulation of TOT_PREC"
    assert reasons["insolation"] == "computed forcing"
    assert reasons["FIS"] == "constant field, not in the time-indexed source"


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_faulty_variant(scenario, spec):
    manifest = json.loads((FAULTY / "manifest.json").read_text())["datasets"]
    findings, details = _run(Path(manifest[scenario]["path"]), spec)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert {f.variable for f in errors} == EXPECTED[scenario]
    assert set(details["differ"]) == EXPECTED[scenario]
    if scenario.startswith("d-step"):
        assert {f.time for f in errors} <= {"2020-03-05T04:00:00"}
