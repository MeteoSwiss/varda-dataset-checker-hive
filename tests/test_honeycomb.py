import json

from pydantic import ValidationError
import pytest

from varda_hive.honeycomb import (
    BEES, CheckContext, Dataset, Severity, Spec, checks_for, error, load_source, load_spec, run, warning,
)
from varda_hive.honeycomb.findings import Report
from varda_hive.honeycomb.registry import _REGISTRY, check
from varda_hive.honeycomb.report import to_json, to_text
from varda_hive.honeycomb.sampling import sample_points, sample_time_indices

from conftest import NT, VARIABLES


def test_dataset_access(dataset_path):
    ds = Dataset(dataset_path)
    assert ds.variables == VARIABLES
    assert len(ds.dates) == NT
    assert ds.date_str(1) == "2020-03-01T01:00:00"
    assert ds.step(0).shape == (len(VARIABLES), 1, 50)


def test_spec_empty_and_strict(write_spec):
    assert load_spec(None) == Spec()
    with pytest.raises(ValidationError):
        load_spec(write_spec("variabels: [T_2M]\n"))  # typo must fail


def test_spec_resolves_relative_paths(write_spec, tmp_path):
    spec = load_spec(write_spec("ranges_file: ranges.yaml\nsampling: {n_dates: 2}\n"))
    assert spec.ranges_file == tmp_path / "ranges.yaml"
    assert spec.sampling.n_dates == 2


def test_sampling_is_seeded_and_includes_dates():
    spec = Spec(sampling={"seed": 3, "n_dates": 3, "include_dates": [7]})
    a = sample_time_indices(100, spec.sampling)
    assert a == sample_time_indices(100, spec.sampling)
    assert 7 in a and len(a) in (3, 4)
    assert len(sample_points(10, spec.sampling)) == 10


def test_report_exit_codes():
    report = Report("x")
    report.add("meta", "c", [warning("w")])
    assert report.exit_code() == 0
    assert report.exit_code(Severity.WARNING) == 1
    report.add("meta", "d", [error("e", variable="T_2M")])
    assert report.exit_code() == 1
    assert report.findings[1].bee == "meta" and report.findings[1].check == "d"
    assert json.loads(to_json(report))["summary"] == {"INFO": 0, "WARNING": 1, "ERROR": 1}
    assert "ERROR   meta.d: e [T_2M]" in to_text(report)


def test_every_bee_has_a_check():
    for bee in BEES:
        assert checks_for(bee), bee


def test_crashing_check_becomes_error(dataset_path):
    @check("meta", name="_crashes")
    def crashes(ctx):
        raise RuntimeError("boom")
        yield

    try:
        report = run(CheckContext(Dataset(dataset_path), Spec()), bees=["meta"], select=["_crashes"])
    finally:
        _REGISTRY["meta"].pop("_crashes")
    assert [f.severity for f in report.findings] == [Severity.ERROR]
    assert "boom" in report.findings[0].message


def test_load_source_from_file(write_spec, tmp_path, dataset_path):
    (tmp_path / "my_source.py").write_text(
        "class S:\n"
        "    def __init__(self, scale):\n        self.scale = scale\n"
        "    def fetch(self, variable, date):\n        return None\n"
    )
    spec = load_spec(write_spec("source: {plugin: my_source.py:S, args: {scale: 2}}\n"))
    assert load_source(spec).scale == 2
    assert load_source(Spec()) is None


def test_load_source_rejects_non_source(write_spec, tmp_path):
    (tmp_path / "bad.py").write_text("class S:\n    pass\n")
    with pytest.raises(TypeError):
        load_source(load_spec(write_spec("source: {plugin: bad.py:S}\n")))
