# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee with FdbSource on the REA-L-CH1 10 min CI dataset, against the real FDB.

Opt-in: `uv run pytest -m fdb`, inside the FDB uenv with the dataset's eccodes definitions:

    uenv run --view fdb:rea-l-ch1 fdb/5.21:v2 -- bash -c '
      export FDB5_CONFIG_FILE=/store_new/mch/msopr/rea-l-ch1/fdb/config_spinup.yaml
      export ECCODES_DEFINITION_PATH=<eccodes-cosmo-resources definitions>
      export HIVE_FDB_DATASET=<path to>/varda-realch1-fdb-1km-2014-2025-10m-sfc-v1.zarr
      uv run pytest -m fdb'

FDB5_CONFIG_FILE selects the FDB root the dataset was built from (here spinup). Skipped without
pyfdb or HIVE_FDB_DATASET.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run

DATASET = Path(os.environ.get("HIVE_FDB_DATASET", "/nonexistent"))

pytestmark = [
    pytest.mark.fdb,
    pytest.mark.skipif(not DATASET.exists(), reason="HIVE_FDB_DATASET not set or missing"),
    # not imported here: pyfdb imported before eccodes segfaults in the fdb uenv (FdbSource imports eccodes first)
    pytest.mark.skipif(importlib.util.find_spec("pyfdb") is None, reason="pyfdb not installed"),
]

DIRECT = ["CAPE_MU", "CLCT", "HZEROCL", "PMSL", "PS", "SNOWLMT", "TD_2M", "T_2M", "T_G", "U_10M", "V_10M"]
CONSTANTS = ["FIS", "FR_LAND", "LPI", "PLCOV", "SKC", "SOILTYP", "SSO_GAMMA", "SSO_SIGMA", "SSO_STDH", "SSO_THETA"]
FORCINGS = [
    "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude", "insolation", "sin_julian_day", "sin_latitude",
    "sin_local_time", "sin_longitude"
]

SOURCE = {
    "plugin": "varda_hive.sources.fdb:FdbSource",
    "args": {
        "request": {
            "class": "rd",
            "stream": "reanl",
            "expver": "r001",
            "model": "icon-rea-l-ch1",
            "type": "cf"
        },
        "offset_from_date": True,
        "step_zero_from_previous_date": ["VMAX_10M"],
        "variables": {
            "VMAX_10M": {},  # `process: maximum` over the preceding 10 min, but the raw GRIB field
            # requested as pressure levels and mapped to sfc by the recipe's flavour
            "CLCL": {
                "request": {
                    "levtype": "pl",
                    "levelist": 800
                }
            },
            "CLCM": {
                "request": {
                    "levtype": "pl",
                    "levelist": 400
                }
            },
            "CLCH": {
                "request": {
                    "levtype": "pl",
                    "levelist": 0
                }
            },
        },
        "skip": {
            "CIN_MU": "masked below -999 by apply_mask_fields"
        },
    },
}


def test_spot_realch1_fdb():
    spec = Spec(source=SOURCE, sampling={"n_dates": "all", "n_points": 2000})
    findings = run(CheckContext(Dataset(DATASET), spec), bees=["spot"]).findings
    assert [f for f in findings if f.severity >= Severity.WARNING] == []
    [summary] = [f for f in findings if f.message.startswith("spot summary")]
    details = summary.details
    assert sorted(details["identical"]) == sorted(DIRECT + ["VMAX_10M", "CLCL", "CLCM", "CLCH"])
    assert sorted(details["not_comparable"]) == sorted(["CIN_MU", "TOT_PREC_10M"] + CONSTANTS + FORCINGS)
