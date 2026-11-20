# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import numpy as np

from pydantic import ValidationError
import pytest

from varda_hive.honeycomb import (
    BEES,
    CheckContext,
    Dataset,
    PhysicalRange,
    Severity,
    Spec,
    checks_for,
    error,
    load_source,
    load_spec,
    run,
    skipped,
    warning,
)
from varda_hive.honeycomb.findings import Report
from varda_hive.honeycomb.registry import _REGISTRY, check
from varda_hive.honeycomb.report import to_json, to_text
from varda_hive.honeycomb.sampling import random_time_indices, sample_points, sample_time_indices, sampled_dates

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
    with pytest.raises(ValidationError, match="not a frequency"):
        load_spec(write_spec("frequency: 1x\n"))
    assert load_spec(write_spec("frequency: PT6H\n")).frequency == "PT6H"


def test_spec_resolves_relative_paths(write_spec, tmp_path):
    write_spec("{}", name="ref.yaml")
    spec = load_spec(write_spec("reference: ref.yaml\nsampling: {n_dates: 2}\n"))
    assert spec.reference == tmp_path / "ref.yaml"
    assert spec.sampling.n_dates == 2


def test_spec_checks_files(write_spec, tmp_path):
    # a missing reference is the first reference: the reference checks skip, `--out` writes it
    assert load_spec(write_spec("reference: nope.yaml\n")).reference == tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="calibrated_tolerances not found"):
        load_spec(write_spec("calibrated_tolerances: nope.yaml\n"))
    for removed in ("ranges_file", "variables_file", "reference_meta", "reference_stats"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            load_spec(write_spec(f"{removed}: x.yaml\n"))


SINGLE_FILE_SPEC = """
anchors:
  wind: &wind {min: -80, max: 80, units: m s-1}
variables:
  FIS: {constant: true, param: FIS, constant_in_time: true}
  T_2M: {min: 200, max: 330, units: K, allow_nans: true, param: T_2M, levtype: sfc}
  T_850: {param: T, levtype: pl, levelist: 850}
  U_10M: *wind
  V_10M: {<<: *wind, param: V_10M}
  W:
"""


def test_spec_single_file_views(write_spec):
    """A `variables` mapping with YAML anchors gives the per-variable views the checks use."""
    spec = load_spec(write_spec(SINGLE_FILE_SPEC))
    with pytest.raises(ValidationError, match="Extra inputs"):  # an unknown top-level key is still a typo
        load_spec(write_spec(SINGLE_FILE_SPEC.replace("anchors:", "anchor:")))
    assert spec.variable_names() == ["FIS", "T_2M", "T_850", "U_10M", "V_10M", "W"]
    assert spec.ranges() == {
        "T_2M": PhysicalRange(min=200, max=330, units="K"),
        "U_10M": PhysicalRange(min=-80, max=80, units="m s-1"),
        "V_10M": PhysicalRange(min=-80, max=80, units="m s-1"),
    }
    assert spec.units() == {"T_2M": "K", "U_10M": "m s-1", "V_10M": "m s-1"}
    assert spec.expected_metadata() == {
        "FIS": {
            "param": "FIS",
            "constant_in_time": True
        },
        "T_2M": {
            "param": "T_2M",
            "levtype": "sfc"
        },
        "T_850": {
            "param": "T",
            "levtype": "pl",
            "levelist": 850
        },
        "V_10M": {
            "param": "V_10M"
        },
    }
    assert spec.constant_field_names() == ["FIS"]
    assert spec.nan_variables() == {"T_2M"}


def test_spec_allow_nans_forms():
    assert Spec().nan_variables() is None
    assert Spec(allow_nans=True).nan_variables() is True
    assert Spec(allow_nans=["A"]).nan_variables() == {"A"}
    assert Spec(variables={"A": None, "B": None}).nan_variables() == set()  # a mapping always says
    assert Spec(variables=["A"]).nan_variables() is None and Spec(variables=["A"]).constant_field_names() is None
    with pytest.raises(ValidationError, match="both for the spec and for variables"):
        Spec(allow_nans=False, variables={"A": {"allow_nans": True}})
    with pytest.raises(ValidationError, match=r"not in `variables`: \['B'\]"):
        Spec(allow_nans=["B"], variables=["A"])


def test_sampling_excludes_indices():
    for stratified in (False, True):
        for n_dates in (3, 10, "all"):
            spec = Spec(sampling={"n_dates": n_dates})
            picked = random_time_indices(10, spec.sampling, stratified=stratified, exclude={2, 3, 4})
            assert picked and not {2, 3, 4} & set(picked)
    spec = Spec(sampling={"n_dates": 4, "include_dates": ["2020-03-01T03:00:00"]})
    assert 3 not in sample_time_indices(spec, _dates(10), {3})
    # nothing excluded: the same draw as before
    assert random_time_indices(100, spec.sampling, stratified=True) == \
        random_time_indices(100, spec.sampling, stratified=True, exclude={200})


def test_select_validates_names(dataset_path):
    ctx = CheckContext(Dataset(dataset_path), Spec())
    with pytest.raises(ValueError, match="unknown check"):
        run(ctx, select=["no_nan"])
    with pytest.raises(ValueError, match="unknown check"):
        run(ctx, bees=["stats"], select=["variables_match_spec"])  # a meta check
    assert run(ctx, select=["meta.variables_match_spec"]).checks_run == ["meta.variables_match_spec"]


def test_strict_fails_on_skipped():
    report = Report("x")
    report.add("meta", "c", [skipped("no `variables` in spec")])
    assert report.findings[0].message == "skipped: no `variables` in spec"
    assert report.exit_code() == 0 and report.exit_code(strict=True) == 1


def _dates(n, start="2020-03-01T00:00:00"):
    return np.datetime64(start, "s") + np.arange(n) * np.timedelta64(3600, "s")


def test_sampling_is_seeded_and_includes_dates():
    spec = Spec(sampling={"seed": 3, "n_dates": 3, "include_dates": ["2020-03-01T07:00:00"]})
    a = sample_time_indices(spec, _dates(100))
    assert a == sample_time_indices(spec, _dates(100))
    assert 7 in a and len(a) in (3, 4)
    assert len(sample_points(10, spec.sampling)) == 10


def test_sampling_follows_the_spec_period():
    """Drawn on the spec's period, the sample is the same on every dataset that has those dates; without a
    period it is drawn on the dataset's dates."""
    period = {"start_date": "2020-03-01T00:00:00", "end_date": "2020-03-02T23:00:00", "frequency": "1h"}
    spec = Spec(sampling={"seed": 1, "n_dates": 5}, **period)
    picked, absent = sampled_dates(spec, _dates(48))
    assert len(picked) == 5 and not absent
    longer = sampled_dates(spec, _dates(60))[0]
    assert longer == picked  # extra dates in the dataset don't move the sample
    shifted, absent = sampled_dates(spec, _dates(72, start="2020-02-29T00:00:00"))  # the same dates, other indices
    assert sorted(t - 24 for t in shifted) == sorted(picked) and not absent
    shorter, absent = sampled_dates(spec, _dates(12))
    assert set(shorter) <= set(picked) and len(shorter) + len(absent) == 5  # the dataset lacks the others
    # the spec's missing_dates are never drawn; the dataset's only count without them
    first = str(_dates(48)[min(picked)])
    without = sampled_dates(Spec(sampling={"seed": 1, "n_dates": 5}, missing_dates=[first], **period), _dates(48))[0]
    assert min(picked) not in without
    # no period: drawn on the dataset's dates, so another time axis gives another sample
    bare = Spec(sampling={"seed": 1, "n_dates": 5})
    assert sampled_dates(bare, _dates(48))[0] == picked
    assert sampled_dates(bare, _dates(60))[0] != picked


def test_include_dates_are_dates():
    with pytest.raises(ValidationError, match="not time indices"):
        Spec(sampling={"include_dates": [3]})


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
    (tmp_path / "my_source.py").write_text("class S:\n"
                                           "    def __init__(self, scale):\n        self.scale = scale\n"
                                           "    def fetch(self, variable, date):\n        return None\n")
    spec = load_spec(write_spec("source: {plugin: my_source.py:S, args: {scale: 2}}\n"))
    assert load_source(spec).scale == 2
    assert load_source(Spec()) is None


def test_load_source_passes_base_dir(write_spec, tmp_path):
    (tmp_path / "dir_source.py").write_text("class S:\n"
                                            "    def __init__(self, base_dir):\n        self.base_dir = base_dir\n"
                                            "    def fetch(self, variable, date):\n        return None\n")
    assert load_source(load_spec(write_spec("source: {plugin: dir_source.py:S}\n"))).base_dir == tmp_path


def test_load_source_rejects_non_source(write_spec, tmp_path):
    (tmp_path / "bad.py").write_text("class S:\n    pass\n")
    with pytest.raises(TypeError):
        load_source(load_spec(write_spec("source: {plugin: bad.py:S}\n")))
