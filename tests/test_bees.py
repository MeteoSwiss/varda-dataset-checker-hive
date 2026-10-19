"""Each template check passes on a clean dataset and fires on a broken one."""
from pathlib import Path

import numpy as np
import zarr

from varda_hive.cli import hive_main, meta_bee_main
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run

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
    assert hive_main(["--list"]) == 0
