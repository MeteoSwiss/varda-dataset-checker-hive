# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CLI exit codes, as CI relies on them: 0 pass, 1 dataset failed, 2 configuration error, 3 internal error."""
import json
import shutil

import numpy as np
import yaml
import zarr

from varda_hive import cli
from varda_hive.bees.meta.metafile import fingerprint, save_meta
from varda_hive.bees.stats.statsfile import compute_stats, load_stats, save_stats
from varda_hive.cli import hive_main, meta_bee_main, spot_bee_main, stats_bee_main
from varda_hive.honeycomb import Dataset, load_spec


def test_config_errors_exit_2(dataset_path, write_spec, tmp_path, capsys):
    assert hive_main([str(tmp_path / "nope.zarr")]) == 2
    assert hive_main([str(dataset_path), "--spec", str(write_spec("variabels: [T_2M]\n"))]) == 2
    assert hive_main([str(dataset_path), "--spec", str(tmp_path / "nope.yaml")]) == 2
    assert hive_main([str(dataset_path), "--select", "no_nan"]) == 2
    assert hive_main([str(dataset_path), "--spec", str(write_spec("frequency: 1x\n"))]) == 2
    assert hive_main(["compare", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")]) == 2
    # an explicit --reference that does not exist is a typo, not a first reference
    assert hive_main([str(dataset_path), "--reference", str(tmp_path / "nope.yaml")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "unknown check(s) no_nan" in err


def test_internal_error_exits_3(dataset_path, monkeypatch, capsys):

    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "run", boom)
    assert hive_main([str(dataset_path)]) == 3
    assert "internal error" in capsys.readouterr().err


def test_strict(dataset_path, capsys):
    assert hive_main([str(dataset_path)]) == 0  # no spec: everything skipped, but passes
    assert hive_main([str(dataset_path), "--strict"]) == 1
    assert "checks run" in capsys.readouterr().out


def _ref(out):
    return yaml.safe_load((out / "ref.yaml").read_text())


def _untimed(dump):
    """`dump` without its `created` timestamps (to the second), which differ between two runs."""
    if isinstance(dump, dict):
        return {k: _untimed(v) for k, v in dump.items() if k != "created"}
    return dump


def test_out_writes_report_and_ref(dataset_path, tmp_path, write_spec, capsys):
    """`--out` writes report.md, report.json and ref.yaml, also when the checks fail."""
    out = tmp_path / "hive"
    assert hive_main([str(dataset_path), "--quiet", "--out", str(out)]) == 0
    assert "checks run" in capsys.readouterr().out  # the text report still goes to stdout
    assert json.loads((out / "report.json").read_text())["summary"]["ERROR"] == 0
    assert (out / "report.md").read_text().startswith("### varda-hive: passed")
    ref = _ref(out)
    assert ref["format"] == "varda-hive-ref/1" and set(ref) == {"format", "meta", "stats"}
    assert ref["meta"]["format"] == "varda-hive-meta/1" and ref["stats"]["format"] == "varda-hive-stats/5"

    failing = tmp_path / "failing"
    spec = write_spec("variables: [T_2M, NOPE]\n")
    assert hive_main([str(dataset_path), "--spec", str(spec), "--out", str(failing)]) == 1
    assert json.loads((failing / "report.json").read_text())["summary"]["ERROR"] > 0
    assert "failed" in (failing / "report.md").read_text()
    assert _untimed(_ref(failing)) == _untimed(ref)  # the dump does not depend on the outcome of the checks

    # a single bee writes only its section; spot-bee has none
    assert meta_bee_main(["check", str(dataset_path), "--out", str(tmp_path / "meta")]) == 0
    assert set(_ref(tmp_path / "meta")) == {"format", "meta"}
    assert spot_bee_main(["check", str(dataset_path), "--out", str(tmp_path / "spot")]) == 0
    assert _ref(tmp_path / "spot") == {"format": "varda-hive-ref/1"}


def test_hive_on_zarr_format_3(dataset_path, dataset_path_v3, tmp_path):
    """All bees run on a zarr format 3 dataset, with the same dump as format 2."""
    assert hive_main([str(dataset_path_v3), "--out", str(tmp_path / "v3")]) == 0
    assert hive_main([str(dataset_path), "--out", str(tmp_path / "v2")]) == 0
    v2, v3 = _ref(tmp_path / "v2"), _ref(tmp_path / "v3")
    assert v3["meta"]["attrs"] == v2["meta"]["attrs"] and v3["meta"]["arrays"] == v2["meta"]["arrays"]
    run_specific = {"dataset", "provenance"}
    for section in ("meta", "stats"):
        assert _untimed({k: v for k, v in v3[section].items() if k not in run_specific}) == \
            _untimed({k: v for k, v in v2[section].items() if k not in run_specific})


def test_out_without_out_writes_nothing(dataset_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hive_main([str(dataset_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.zarr"]


def test_failed_dump_is_a_finding(dataset_path, tmp_path, monkeypatch, capsys):
    import varda_hive.bees.meta.checks as meta_checks  # pylint: disable=import-outside-toplevel

    reference = tmp_path / "ref" / "ref.yaml"
    assert hive_main([str(dataset_path), "--out", str(reference.parent)]) == 0
    capsys.readouterr()
    calls = []

    def boom(ds):
        calls.append(ds)
        raise RuntimeError("broken")

    monkeypatch.setattr(meta_checks, "fingerprint", boom)
    assert hive_main(
        [str(dataset_path), "--reference",
         str(reference), "--out",
         str(tmp_path / "hive"), "--format", "json"]) == 1
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert any(f["check"] == "dump" and "dump failed" in f["message"] for f in findings)
    assert any(f["check"] == "matches_reference" and "check crashed" in f["message"] for f in findings)
    assert len(calls) == 1  # the failure is kept: not built again for --out
    assert set(_ref(tmp_path / "hive")) == {"format", "stats"}


def test_out_safeguards(dataset_path, tmp_path, write_spec, capsys):
    # --out must not overwrite the reference it is compared with
    assert hive_main([str(dataset_path), "--bees", "meta", "--out", str(tmp_path)]) == 0
    spec = write_spec("reference: ref.yaml\n")
    assert hive_main([str(dataset_path), "--spec", str(spec), "--out", str(tmp_path)]) == 2
    assert "would overwrite the reference" in capsys.readouterr().err
    # a failure to write --out comes after the report, so the findings are not lost
    (tmp_path / "file").write_text("")
    assert hive_main([str(dataset_path), "--out", str(tmp_path / "file")]) == 2
    captured = capsys.readouterr()
    assert "checks run" in captured.out and "hive: error" in captured.err


def test_three_workflows(dataset_path, tmp_path, write_spec, capsys):
    """One command for the first reference, a check against it and accepting an intended change."""
    spec = write_spec("reference: data.ref.yaml\nsampling: {n_dates: 3, seed: 1}\n")
    reference, out = tmp_path / "data.ref.yaml", tmp_path / "hive"
    # --select: only the reference checks decide the exit code; ref.yaml is the whole dump regardless
    command = [
        str(dataset_path), "--spec",
        str(spec), "--bees", "meta", "stats", "--select", "matches_reference", "matches_reference_stats", "--strict",
        "--out",
        str(out), "--format", "json"
    ]

    # first reference: the reference checks skip (exit 1 under --strict, not 2), ref.yaml is written
    assert hive_main(command) == 1
    findings = json.loads(capsys.readouterr().out)["findings"]
    skipped = {f["check"]: f["message"] for f in findings if f["skipped"]}
    assert set(skipped) == {"matches_reference", "matches_reference_stats"}
    assert all(f"reference {reference} not found" in message for message in skipped.values())
    assert set(_ref(out)) == {"format", "meta", "stats"}
    shutil.copy(out / "ref.yaml", reference)

    # reference exists: passes, and ref.yaml is the dump of the spec's sample (what `stats-bee dump` and
    # `meta-bee dump` wrote), on the reference's dates
    assert hive_main(command) == 0
    capsys.readouterr()
    ref = _ref(out)
    ds = Dataset(dataset_path)
    assert _untimed(ref["stats"]) == _untimed(json.loads(json.dumps(compute_stats(ds, load_spec(spec)))))
    assert ref["stats"]["sampling"]["dates"] == load_stats(reference)["sampling"]["dates"]
    assert _untimed(ref["meta"]) == _untimed(json.loads(json.dumps(fingerprint(ds))))

    # an intended change: report.md lists the differences; accepting it is a copy
    zarr.open_group(str(dataset_path), mode="r+")["data"][:, 3] *= 2
    metadata = json.loads((dataset_path / ".zattrs").read_text())
    metadata["variables_metadata"]["T_850"]["mars"]["levelist"] = 800
    (dataset_path / ".zattrs").write_text(json.dumps(metadata))
    assert hive_main(command) == 1
    failed = {(f["check"], f["severity"]) for f in json.loads(capsys.readouterr().out)["findings"]}
    assert {("matches_reference", "WARNING"), ("matches_reference_stats", "ERROR")} <= failed
    report = (out / "report.md").read_text()
    assert "variables_metadata.T_850.mars.levelist" in report and "T_850" in report
    assert _ref(out)["stats"]["sampling"]["dates"] == ref["stats"]["sampling"]["dates"]  # the spec's sample
    shutil.copy(out / "ref.yaml", reference)
    assert hive_main(command) == 0


def test_hive_compare(dataset_path, tmp_path, capsys):
    """`hive compare` compares two dumps section by section, as the reference checks do."""
    assert hive_main([str(dataset_path), "--out", str(tmp_path / "a")]) == 0
    ref = tmp_path / "a" / "ref.yaml"
    assert hive_main([str(dataset_path), "--reference", str(ref), "--out", str(tmp_path / "b")]) == 0
    new = tmp_path / "b" / "ref.yaml"
    capsys.readouterr()
    assert hive_main(["compare", str(new), str(ref)]) == 0
    out = capsys.readouterr().out
    assert "meta: " in out and "identical" in out and "5/5 variables identical" in out
    assert hive_main(["compare", str(new), str(ref), "--all"]) == 0
    assert "arrays.dates.values" in capsys.readouterr().out

    content = yaml.safe_load(new.read_text())
    variables = content["meta"]["attrs"]["variables"]
    variables[1], variables[2] = variables[2], variables[1]
    content["meta"]["attrs"]["variables_metadata"]["T_850"]["mars"]["levelist"] = 800
    new.write_text(yaml.safe_dump(content))
    assert hive_main(["compare", str(new), str(ref)]) == 1
    out = capsys.readouterr().out
    assert "order differs" in out and "variables_metadata.T_850.mars.levelist" in out
    assert hive_main(["compare", str(new), str(ref), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["ERROR"] == 1
    assert hive_main(["compare", str(new), str(ref), "--format", "markdown"]) == 1
    assert "| variables | `variables` |" in capsys.readouterr().out

    # a section on one side only is skipped; a file that is no ref file is a configuration error
    assert meta_bee_main(["check", str(dataset_path), "--out", str(tmp_path / "m")]) == 0
    capsys.readouterr()
    assert hive_main(["compare", str(tmp_path / "m" / "ref.yaml"), str(ref), "--format", "json"]) == 0
    findings = json.loads(capsys.readouterr().out)["findings"]
    missing = tmp_path / "m" / "ref.yaml"
    assert [f["message"] for f in findings if f["skipped"]] == [f"skipped: no `stats` section in {missing}"]
    assert hive_main(["compare", str(missing), str(ref), "--strict"]) == 1

    # a section without a comparison (spot-bee has no dump) is skipped, not a crash
    content = yaml.safe_load(ref.read_text())
    (tmp_path / "spot.ref.yaml").write_text(yaml.safe_dump(content | {"spot": {"format": "x"}}))
    assert hive_main(["compare", str(tmp_path / "spot.ref.yaml"), str(ref)]) == 0
    assert "no comparison for section `spot`" in capsys.readouterr().out
    # an older section format: a configuration error with a hint, not an internal error
    content["stats"]["format"] = "varda-hive-stats/4"
    (tmp_path / "old.ref.yaml").write_text(yaml.safe_dump(content))
    assert hive_main(["compare", str(tmp_path / "old.ref.yaml"), str(ref)]) == 2
    assert "replace it with the `ref.yaml` of `hive ... --out`" in capsys.readouterr().err
    # a dataset is not a ref file
    assert hive_main(["compare", str(dataset_path), str(ref)]) == 2
    assert "a dataset, not a ref file" in capsys.readouterr().err
    save_meta(fingerprint(Dataset(dataset_path)), tmp_path / "x.meta.yaml")
    assert hive_main(["compare", str(tmp_path / "x.meta.yaml"), str(ref)]) == 2
    assert "not a ref file" in capsys.readouterr().err


def test_inf_in_data(dataset_path, tmp_path):
    zarr.open_group(str(dataset_path), mode="r+")["data"][:, 1, 0, 0] = np.inf
    out = tmp_path / "hive"
    assert stats_bee_main(["check", str(dataset_path), "--out", str(out)]) == 1  # finite_values
    stats = _ref(out)["stats"]
    assert stats["sample"]["T_2M"]["n_inf"] == len(stats["sampling"]["dates"])
    json.loads((out / "report.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_markdown_report(dataset_path, capsys):
    assert hive_main([str(dataset_path), "--format", "markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### varda-hive: passed") and "<details>" in out


def test_reference_kinds(dataset_path, tmp_path, capsys):
    """A meta file is no stats reference and a stats file no meta reference: the other bee skips."""
    meta, stats = tmp_path / "ref.meta.yaml", tmp_path / "ref.stats.yaml"
    save_meta(fingerprint(Dataset(dataset_path)), meta)
    save_stats(compute_stats(Dataset(dataset_path), load_spec(None)), stats)
    for ref, skipped in ((meta, "matches_reference_stats"), (stats, "matches_reference"), (dataset_path,
                                                                                           "matches_reference_stats")):
        assert hive_main([str(dataset_path), "--reference", str(ref), "--format", "json"]) == 0
        findings = json.loads(capsys.readouterr().out)["findings"]
        assert [f["check"] for f in findings if f.get("skipped") and "reference" in f["check"]] == [skipped]
