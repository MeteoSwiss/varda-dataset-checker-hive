# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The bees are independent: each one imports and runs with the other bees made unimportable, so a
bug in one bee can never break another."""
import subprocess
import sys
from pathlib import Path

import pytest

from varda_hive.honeycomb import BEES

SCRIPT = """
import sys
bee, others, dataset = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
for other in others:
    sys.modules[f"varda_hive.bees.{other}"] = None  # any import of it raises ImportError
from varda_hive import cli
from varda_hive.honeycomb import checks_for
assert checks_for(bee), f"{bee} has no checks"
code = getattr(cli, f"{bee}_bee_main")(["check", dataset])
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
