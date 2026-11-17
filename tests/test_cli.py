# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CLI exit codes, as CI relies on them: 0 pass, 1 dataset failed, 2 configuration error, 3 internal error."""
import json

import numpy as np
import zarr

from varda_hive import cli
from varda_hive.cli import hive_main, stats_bee_main


def test_config_errors_exit_2(dataset_path, write_spec, tmp_path, capsys):
    assert hive_main([str(tmp_path / "nope.zarr")]) == 2
    assert hive_main([str(dataset_path), "--spec", str(write_spec("variabels: [T_2M]\n"))]) == 2
    assert hive_main([str(dataset_path), "--spec", str(tmp_path / "nope.yaml")]) == 2
    assert hive_main([str(dataset_path), "--select", "no_nan"]) == 2
    assert hive_main([str(dataset_path), "--spec", str(write_spec("frequency: 1x\n"))]) == 2
    assert stats_bee_main(["compare", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "unknown check(s) no_nan" in err


def test_internal_error_exits_3(dataset_path, monkeypatch, capsys):

    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "run", boom)
    assert hive_main([str(dataset_path)]) == 3
    assert "internal error" in capsys.readouterr().err


def test_strict_and_report(dataset_path, tmp_path, capsys):
    assert hive_main([str(dataset_path)]) == 0  # no spec: everything skipped, but passes
    assert hive_main([str(dataset_path), "--strict"]) == 1
    capsys.readouterr()
    report = tmp_path / "report.json"
    assert hive_main([str(dataset_path), "--quiet", "--report", str(report)]) == 0
    assert "checks run" in capsys.readouterr().out  # text on stdout, JSON in the file
    assert json.loads(report.read_text())["summary"]["ERROR"] == 0


def test_dump_writes_valid_json_with_inf(dataset_path, tmp_path):
    zarr.open_group(str(dataset_path), mode="r+")["data"][:, 1, 0, 0] = np.inf
    out = tmp_path / "s.json"
    assert stats_bee_main(["dump", str(dataset_path), "-o", str(out)]) == 0
    stats = json.loads(out.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert stats["format"] == "varda-hive-stats/4" and stats["sample"]["T_2M"]["n_inf"] == len(
        stats["sampling"]["dates"])
    assert stats_bee_main(["check", str(dataset_path), "--select", "finite_values"]) == 1


def test_meta_bee_dump_and_compare(dataset_path, tmp_path, capsys):
    from varda_hive.cli import meta_bee_main

    ref = tmp_path / "ref.meta.yaml"
    assert meta_bee_main(["dump", str(dataset_path), "-o", str(ref)]) == 0
    assert meta_bee_main(["compare", str(dataset_path), str(ref)]) == 0
    out = capsys.readouterr().out
    assert "identical" in out and "0 error" in out
    assert meta_bee_main(["compare", str(dataset_path), str(ref), "--all"]) == 0
    assert "arrays.dates.values" in capsys.readouterr().out

    metadata = json.loads((dataset_path / ".zattrs").read_text())
    metadata["variables"][1], metadata["variables"][2] = metadata["variables"][2], metadata["variables"][1]
    metadata["variables_metadata"]["T_850"]["mars"]["levelist"] = 800
    (dataset_path / ".zattrs").write_text(json.dumps(metadata))
    assert meta_bee_main(["compare", str(dataset_path), str(ref)]) == 1
    out = capsys.readouterr().out
    assert "order differs" in out and "variables_metadata.T_850.mars.levelist" in out
    assert meta_bee_main(["compare", str(dataset_path), str(ref), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["ERROR"] == 1
    assert meta_bee_main(["compare", str(dataset_path), str(ref), "--format", "markdown"]) == 1
    assert "| variables | `variables` |" in capsys.readouterr().out

    # --reference with a meta file, in check and hive
    assert meta_bee_main(["check", str(dataset_path), "--reference", str(ref), "--select", "matches_reference"]) == 1
    assert hive_main([str(dataset_path), "--reference", str(tmp_path / "nope.yaml")]) == 2
    assert meta_bee_main(["compare", str(dataset_path), str(tmp_path / "nope.yaml")]) == 2


def test_markdown_report(dataset_path, capsys):
    assert hive_main([str(dataset_path), "--format", "markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### varda-hive: passed") and "<details>" in out
