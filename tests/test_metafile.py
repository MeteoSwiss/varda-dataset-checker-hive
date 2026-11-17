# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Meta files: fingerprint, round trip, and the comparison policy."""
import json

import pytest

from conftest import VARIABLES, build_dataset
from varda_hive.honeycomb import Dataset, ReferencePolicy, Severity
from varda_hive.bees.meta.metafile import (ADDED, DIFFERS, IDENTICAL, IGNORED, MISSING, compare_meta, fingerprint,
                                           load_meta, load_reference, save_meta, severity_of)


def _statuses(comparison) -> dict[str, str]:
    return {i.path: i.status for i in comparison.items if i.status != IDENTICAL}


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_round_trip(dataset_path, tmp_path, suffix):
    meta = fingerprint(Dataset(dataset_path))
    assert "values" not in meta["arrays"]["data"] and len(meta["arrays"]["dates"]["values"]) == 64
    assert meta["arrays"]["dates"]["summary"].endswith("step 1h")
    save_meta(meta, tmp_path / f"m{suffix}")
    loaded = load_meta(tmp_path / f"m{suffix}")
    assert loaded == json.loads(json.dumps(meta))
    assert load_reference(tmp_path / f"m{suffix}") == loaded


def test_rejects_other_files(tmp_path):
    (tmp_path / "s.json").write_text('{"format": "varda-hive-stats/4"}')
    with pytest.raises(ValueError, match="not a meta file"):
        load_meta(tmp_path / "s.json")
    (tmp_path / "old.json").write_text('{"format": "varda-hive-meta/0"}')
    with pytest.raises(ValueError, match="regenerate it with `meta-bee dump`"):
        load_meta(tmp_path / "old.json")
    with pytest.raises(FileNotFoundError):
        load_reference(tmp_path / "nope.zarr")


def test_identical_and_churn(tmp_path):
    a = fingerprint(Dataset(build_dataset(tmp_path / "a.zarr")))
    b = fingerprint(Dataset(build_dataset(tmp_path / "b.zarr")))
    b["attrs"]["uuid"], b["attrs"]["latest_write_timestamp"] = "x", "now"
    a["attrs"]["recipe"] = {"build": {"group_by": "monthly"}}
    comparison = compare_meta(a, b)
    assert _statuses(comparison) == {"uuid": IGNORED, "latest_write_timestamp": IGNORED, "recipe": IGNORED}
    assert [f.severity for f in comparison.findings] == [Severity.INFO]  # the summary only
    assert comparison.counts[IGNORED] == 3 and comparison.counts[DIFFERS] == 0


def test_differences(tmp_path):
    reference = fingerprint(Dataset(build_dataset(tmp_path / "a.zarr")))
    actual = fingerprint(Dataset(build_dataset(tmp_path / "b.zarr", seed=1, variables=VARIABLES + ["X"])))
    actual["attrs"]["variables_metadata"]["T_850"]["mars"]["levelist"] = 800
    actual["attrs"]["new_key"] = 1
    del actual["attrs"]["resolution"]
    comparison = compare_meta(actual, reference)
    statuses = _statuses(comparison)
    assert statuses["variables"] == DIFFERS and statuses["arrays.data.shape"] == DIFFERS
    assert statuses["variables_metadata.T_850.mars.levelist"] == DIFFERS
    assert statuses["new_key"] == ADDED and statuses["resolution"] == MISSING
    by_path = {f.details.get("path"): f for f in comparison.findings}
    assert by_path["variables"].severity == Severity.ERROR and "added ['X']" in by_path["variables"].message
    assert by_path["variables_metadata.T_850.mars.levelist"].variable == "T_850"
    assert by_path["variables_metadata.T_850.mars.levelist"].severity == Severity.WARNING
    assert by_path["arrays.mean.shape"].severity == Severity.ERROR


def test_values_changed_same_summary(tmp_path):
    reference = fingerprint(Dataset(build_dataset(tmp_path / "a.zarr")))
    actual = fingerprint(Dataset(build_dataset(tmp_path / "b.zarr", seed=1)))
    by_path = {i.path: i for i in compare_meta(actual, reference).items}
    assert by_path["arrays.mean.values"].status == DIFFERS
    assert by_path["arrays.mean.values"].severity == Severity.INFO  # not bit-reproducible
    assert by_path["arrays.latitudes.values"].status == IDENTICAL


def test_order_and_duplicates_notes(tmp_path):
    reference = fingerprint(Dataset(build_dataset(tmp_path / "a.zarr")))
    actual = json.loads(json.dumps(reference))
    actual["attrs"]["variables"] = VARIABLES[::-1]
    [item] = [i for i in compare_meta(actual, reference).items if i.path == "variables"]
    assert item.note.startswith("same elements, order differs")
    actual["attrs"]["variables"] = VARIABLES + VARIABLES[:1]
    [item] = [i for i in compare_meta(actual, reference).items if i.path == "variables"]
    assert "duplicates" in item.note


def test_policy_overrides():
    policy = ReferencePolicy(ignore=["variables_metadata.*.mars.date"], error=["constant_fields"], info=["extra_*"])
    assert severity_of("variables_metadata.T_2M.mars.date", policy) == "ignore"
    assert severity_of("variables_metadata.T_2M.mars.param", policy) == Severity.WARNING
    assert severity_of("constant_fields", policy) == Severity.ERROR
    assert severity_of("extra_note", policy) == Severity.INFO
    assert severity_of("uuid") == "ignore" and severity_of("arrays.dates.values") == Severity.ERROR
    assert severity_of("something_new") == Severity.WARNING


def test_zarr_and_meta_file_reference_agree(dataset_path, tmp_path):
    reference = build_dataset(tmp_path / "ref.zarr", seed=3)
    save_meta(fingerprint(Dataset(reference)), tmp_path / "ref.meta.yaml")
    actual = fingerprint(Dataset(dataset_path))
    from_zarr = compare_meta(actual, load_reference(reference))
    from_file = compare_meta(actual, load_reference(tmp_path / "ref.meta.yaml"))
    assert from_zarr.items == from_file.items
