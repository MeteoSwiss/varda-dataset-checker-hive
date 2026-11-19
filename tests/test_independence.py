# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The bees are independent: each one imports and runs with the other bees made unimportable, so a
bug in one bee can never break another."""
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from varda_hive.honeycomb import BEES

SCRIPT = """
import sys
bee, others, dataset, spec = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4:]
for other in others:
    sys.modules[f"varda_hive.bees.{other}"] = None  # any import of it raises ImportError
from varda_hive import cli
from varda_hive.honeycomb import checks_for
assert checks_for(bee), f"{bee} has no checks"
code = getattr(cli, f"{bee}_bee_main")(["check", dataset] + (["--spec", spec[0]] if spec else []))
assert code in (0, 1), f"{bee}-bee check exited {code}"
loaded = {m.split(".")[2] for m in sys.modules if m.startswith("varda_hive.bees.") and sys.modules[m] is not None}
assert loaded == {bee}, f"{bee}-bee loaded other bees: {sorted(loaded - {bee})}"
"""


@pytest.mark.parametrize("bee", BEES)
def test_bee_runs_without_the_others(bee: str, dataset_path: Path) -> None:
    others = ",".join(b for b in BEES if b != bee)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, bee, others, str(dataset_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]


def test_spot_with_grib_source_runs_without_the_others(dataset_path: Path, tmp_path: Path) -> None:
    """spot-bee with a source plug-in from varda_hive.sources loads it, and still no other bee. The index is
    empty, so nothing is decoded, but binding the source imports eccodes."""
    pytest.importorskip("eccodes")
    with sqlite3.connect(tmp_path / "index.db") as con:
        con.execute("CREATE TABLE paths (id INTEGER PRIMARY KEY, path TEXT)")
        con.execute("CREATE TABLE grib_index (_id INTEGER PRIMARY KEY, _path_id INTEGER, _offset INTEGER, "
                    "_length INTEGER, valid_datetime TEXT, shortName TEXT, level TEXT, typeOfLevel TEXT)")
    spec = tmp_path / "spec.yaml"
    spec.write_text("source:\n  plugin: varda_hive.sources.grib_index:GribIndexSource\n"
                    "  args: {indexes: {main: index.db}}\n")
    check = SCRIPT + 'assert "varda_hive.sources.grib_index" in sys.modules, "source not loaded"\n'
    others = ",".join(b for b in BEES if b != "spot")
    result = subprocess.run(
        [sys.executable, "-c", check, "spot", others,
         str(dataset_path), str(spec)],
        capture_output=True,
        text=True,
        check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "check crashed" not in result.stdout, result.stdout[-2000:]  # exit 1 alone could hide a crash
