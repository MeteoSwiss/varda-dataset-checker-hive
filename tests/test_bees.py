"""Each template check passes on a clean dataset and fires on a broken one."""
import json
from pathlib import Path

import numpy as np
import zarr

from varda_hive.cli import hive_main, meta_bee_main
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, load_spec, run
from varda_hive.honeycomb.statsfile import compute_stats, save_stats

from conftest import VARIABLES, build_dataset

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def _errors(path, bee, spec=None):
    report = run(CheckContext(Dataset(path), spec or Spec()), bees=[bee])
    return [f for f in report.findings if f.severity == Severity.ERROR]


def test_meta_variables(dataset_path, tmp_path):
    spec = Spec(variables=VARIABLES)
    assert _errors(dataset_path, "meta", spec) == []
    extra = build_dataset(tmp_path / "extra.zarr", variables=VARIABLES + ["T_2M_COPY"])
    assert [(f.variable, f.message) for f in _errors(extra, "meta", spec)] == [("T_2M_COPY", "variable not in spec")]


def test_meta_skips_without_spec(dataset_path):
    report = run(CheckContext(Dataset(dataset_path), Spec()), bees=["meta"])
    assert [f.severity for f in report.findings] == [Severity.INFO]


def test_stats_no_nans(dataset_path):
    spec = Spec(sampling={"n_dates": 1, "include_dates": [3]})
    assert _errors(dataset_path, "stats", spec) == []
    zarr.open_group(str(dataset_path), mode="r+")["data"][3, 1] = np.nan
    [f] = _errors(dataset_path, "stats", spec)
    assert (f.variable, f.time) == ("T_2M", "2020-03-01T03:00:00")


def _zarr_source_spec(reference: Path, **sampling) -> Spec:
    return Spec(source={"plugin": "source_plugin_example.py:ZarrSource", "args": {"path": str(reference)}},
                sampling=sampling, base_dir=TEMPLATES)


def test_spot_matches_source(dataset_path, tmp_path):
    reference = build_dataset(tmp_path / "reference.zarr")  # identical copy
    spec = _zarr_source_spec(reference, n_dates=2, include_dates=[5])
    assert _errors(dataset_path, "spot", spec) == []
    zarr.open_group(str(dataset_path), mode="r+")["data"][5, 3] += 1.0
    [f] = _errors(dataset_path, "spot", spec)
    assert (f.variable, f.time) == ("U_10M", "2020-03-01T05:00:00")


def test_cli(dataset_path, write_spec, capsys):
    spec = write_spec("variables: [T_2M]\n")
    assert meta_bee_main([str(dataset_path), "--spec", str(spec)]) == 1
    assert "variable not in spec" in capsys.readouterr().out
    assert hive_main([str(dataset_path), "--format", "json"]) == 0
    capsys.readouterr()
    assert hive_main(["--list"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in listing if not line.startswith(" ")] == ["meta-bee", "stats-bee", "spot-bee"]
    assert "  no_nans " in "\n".join(listing)


def _stats_findings(path, spec, select, severity=Severity.ERROR):
    report = run(CheckContext(Dataset(path), spec), bees=["stats"], select=[select])
    return [f for f in report.findings if f.severity == severity]


def test_stats_statistics_consistent(dataset_path):
    spec = Spec(sampling={"n_dates": 4})
    assert _stats_findings(dataset_path, spec, "statistics_consistent") == []
    root = zarr.open_group(str(dataset_path), mode="r+")
    root["mean"][2] *= 1.5
    root["stdev"][3] = 0.0
    root["minimum"][4], root["maximum"][4] = root["maximum"][4], root["minimum"][4]
    flagged = {(f.variable, f.message.split()[1]) for f in _stats_findings(dataset_path, spec, "statistics_consistent")}
    assert ("T_850", "mean") in flagged
    assert ("U_10M", "stdev") in flagged
    assert ("V_10M", "statistics") in flagged  # minimum > maximum


def test_stats_statistics_consistent_exact(dataset_path):
    spec = Spec(sampling={"n_dates": "all"})
    assert _stats_findings(dataset_path, spec, "statistics_consistent") == []
    zarr.open_group(str(dataset_path), mode="r+")["mean"][1] *= 1.0001  # far below any sampling error
    [f] = _stats_findings(dataset_path, spec, "statistics_consistent")
    assert f.variable == "T_2M" and f.details["exact"]


def test_stats_physical_ranges(dataset_path, write_spec):
    write_spec("T_2M: {min: 200.0, max: 330.0, units: K}\n", name="ranges.yaml")
    spec = load_spec(write_spec("ranges_file: ranges.yaml\nsampling: {n_dates: 1, include_dates: [6]}\n"))
    assert _stats_findings(dataset_path, spec, "physical_ranges") == []
    zarr.open_group(str(dataset_path), mode="r+")["data"][6, 1, 0, 0] = 400.0
    [f] = _stats_findings(dataset_path, spec, "physical_ranges")
    assert (f.variable, f.time) == ("T_2M", "2020-03-01T06:00:00")


def test_stats_constant_fields(dataset_path):
    spec = Spec(sampling={"n_dates": "all"})
    assert _stats_findings(dataset_path, spec, "constant_fields", Severity.WARNING) == []
    assert _stats_findings(dataset_path, spec, "constant_fields") == []
    data = zarr.open_group(str(dataset_path), mode="r+")["data"]
    data[2, 0] += 1.0  # constant field FIS changes
    data[3, 2] = 0.0   # T_850 zeroed
    [f] = _stats_findings(dataset_path, spec, "constant_fields")
    assert f.variable == "FIS" and f.details["dates"] == ["2020-03-01T02:00:00"]
    [w] = _stats_findings(dataset_path, spec, "constant_fields", Severity.WARNING)
    assert (w.variable, w.time) == ("T_850", "2020-03-01T03:00:00")


def test_stats_constant_field_not_listed(dataset_path):
    attrs = json.loads((dataset_path / ".zattrs").read_text())
    attrs["constant_fields"] = []
    (dataset_path / ".zattrs").write_text(json.dumps(attrs))
    [w] = _stats_findings(dataset_path, Spec(sampling={"n_dates": 3}), "constant_fields", Severity.WARNING)
    assert w.variable == "FIS"


def test_stats_matches_reference_stats(dataset_path, tmp_path, write_spec):
    save_stats(compute_stats(Dataset(dataset_path), Spec(sampling={"n_dates": 3})), tmp_path / "ref.json")
    spec = load_spec(write_spec("reference_stats: ref.json\nsampling: {n_dates: 1, seed: 7}\n"))
    assert _stats_findings(dataset_path, spec, "matches_reference_stats") == []
    zarr.open_group(str(dataset_path), mode="r+")["data"][:, 3] *= 2
    assert {f.variable for f in _stats_findings(dataset_path, spec, "matches_reference_stats")} == {"U_10M"}
