# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The sample, stats files (generation, round-trip, comparison), calibration, and the
`stats-bee dump` / `compare` / `calibrate` commands."""
import json

import numpy as np
import pytest
import yaml
import zarr

from varda_hive.cli import stats_bee_main
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, load_spec, run
from varda_hive.honeycomb.sampling import sample_points
from varda_hive.bees.stats.calibrate import calibrate
from varda_hive.bees.stats.statsfile import compare_stats, compute_sample, compute_stats, load_stats, save_stats

from conftest import NG, NT, VARIABLES, build_dataset


def _dump(path, **sampling):
    return compute_stats(Dataset(path), Spec(sampling=sampling))


def _severities(comparison):
    return [f.severity for f in comparison.findings]


@pytest.mark.parametrize("name", ["s.json", "s.yaml"])
def test_round_trip_is_exact(dataset_path, tmp_path, name):
    stats = _dump(dataset_path, n_dates=3, include_dates=[7])
    stats["stored"]["T_2M"]["mean"] = 1e-17  # tiny floats too
    save_stats(stats, tmp_path / name)
    assert load_stats(tmp_path / name) == stats
    assert "2020-03-01T07:00:00" in stats["sampling"]["dates"]


def test_stats_file_size_is_independent_of_the_dataset_and_sample(tmp_path):
    """Only the list of sampled dates grows, with n_dates; nothing grows with the dataset."""
    small = build_dataset(tmp_path / "small.zarr", nt=8)
    large = build_dataset(tmp_path / "large.zarr", nt=300)

    def size(path, **sampling):
        stats = _dump(path, **sampling)
        del stats["sampling"]["dates"], stats["sampling"]["random"]
        return len(json.dumps(stats, sort_keys=True).encode())

    sizes = {size(small, n_dates=4, n_points=10), size(large, n_dates=4, n_points=10), size(large, n_dates=40)}
    assert max(sizes) - min(sizes) < 40, sizes  # digits of numbers and counts only


def test_sample_aggregate_is_exact(dataset_path):
    spec = Spec(sampling={"n_dates": 4, "n_points": 20, "include_dates": [7]})
    ds = Dataset(dataset_path)
    sample = compute_sample(ds, spec)
    points = sample_points(NG, spec.sampling)
    full = ds.data[:][sample.time_index].astype("f8")  # (date, variable, 1, grid)
    agg = sample.aggregate()
    for i, name in enumerate(VARIABLES):
        x = full[:, i, ..., points].ravel()  # mean and stdev: on the sampled points
        assert np.isclose(agg[name]["mean"], x.mean(), rtol=1e-12)
        assert np.isclose(agg[name]["stdev"], x.std(), rtol=1e-9)
        # minimum and maximum: over the full domain of the sampled dates
        assert (agg[name]["minimum"], agg[name]["maximum"]) == (full[:, i].min(), full[:, i].max())


def test_extremes_and_counts_cover_points_outside_the_subset(dataset_path):
    """A spike or a NaN on a single grid point that is not sampled is still found."""
    spec = Spec(sampling={"n_dates": 1, "n_points": 10, "include_dates": [6]})
    outside = int(np.setdiff1d(np.arange(NG), sample_points(NG, spec.sampling))[0])
    data = zarr.open_group(str(dataset_path), mode="r+")["data"]
    data[6, 1, 0, outside] = 400.0  # T_2M spike
    data[6, 3, 0, outside] = np.nan  # U_10M NaN
    sample = compute_sample(Dataset(dataset_path), spec)
    k = sample.time_index.index(6)
    assert sample.per_date("T_2M")["maximum"][k] == 400.0
    assert sample.per_date("U_10M")["n_nan"][k] == 1
    assert sample.per_date("U_10M")["n_finite"][k] == 10  # the moments only see the sampled points


def test_full_pass_recomputes_stored_statistics(dataset_path):
    sample = compute_sample(Dataset(dataset_path), Spec(sampling={"n_dates": "all", "n_points": "all"}))
    estimates = sample.estimate()
    for name in VARIABLES:
        stored, rec = sample.stored[name], estimates[name]
        assert rec["exact"] and rec["se_mean"] == 0
        for s in ("mean", "stdev", "minimum", "maximum"):
            assert np.isclose(rec[s], stored[s], rtol=1e-6, atol=1e-6), (name, s)


def test_dates_cover_the_whole_range_estimate_uses_the_window(tmp_path):
    path = build_dataset(tmp_path / "ds.zarr", nt=40)
    attrs = json.loads((path / ".zattrs").read_text())
    attrs["statistics_end_date"] = "2020-03-01T19:00:00"  # window: indices 0-19
    (path / ".zattrs").write_text(json.dumps(attrs))
    sample = compute_sample(Dataset(path), Spec(sampling={"n_dates": 8}))
    assert any(t >= 20 for t in sample.time_index)  # value checks see the dates after the window
    assert all(t < 20 for t, r in zip(sample.time_index, sample.random) if r)  # the estimate doesn't
    assert sample.n_window == 20


def test_yaml_and_json_compare_identical(dataset_path, tmp_path):
    stats = _dump(dataset_path, n_dates=4)
    save_stats(stats, tmp_path / "s.yaml")
    save_stats(stats, tmp_path / "s.json")
    assert "sample:" in (tmp_path / "s.yaml").read_text()
    comparison = compare_stats(load_stats(tmp_path / "s.yaml"), load_stats(tmp_path / "s.json"))
    assert _severities(comparison) == [Severity.INFO]
    assert all(c["close"] == c["differs"] == 0 for c in comparison.counts.values())


def test_compare_differs_and_close(dataset_path):
    reference = _dump(dataset_path, n_dates=2, include_dates=[5])
    zarr.open_group(str(dataset_path), mode="r+")["data"][5, 1] += 0.01
    stats = compute_stats(Dataset(dataset_path), Spec(), dates_from=reference)
    assert stats["sampling"]["dates"] == reference["sampling"]["dates"]

    differs = [f for f in compare_stats(stats, reference).findings if f.severity == Severity.ERROR]
    assert {f.variable for f in differs} == {"T_2M"}
    # one date changed: mean and stdev of the sample move, its global min/max not necessarily
    assert {"sample mean differs", "sample stdev differs"} <= {f.message.split(":")[0] for f in differs}
    close = compare_stats(stats, reference, atol_stdev=0.01)  # 0.01 * stdev (~5) covers the change
    assert Severity.ERROR not in _severities(close) and Severity.WARNING in _severities(close)


def test_compare_aligns_dates_not_positions(dataset_path, tmp_path):
    reference = _dump(dataset_path, n_dates=3)
    longer = build_dataset(tmp_path / "longer.zarr", nt=NT + 5)
    stats = compute_stats(Dataset(longer), Spec(), dates_from=reference)
    assert stats["sampling"]["dates"] == reference["sampling"]["dates"]
    assert not any("dates" in f.message for f in compare_stats(stats, reference).findings)
    # without --dates-from the seed draws other dates on the longer time axis: reported
    other = compute_stats(Dataset(longer), Spec(sampling={"n_dates": 3}))
    assert any("sampled dates differ" in f.message for f in compare_stats(other, reference).findings)


def test_compare_reports_sampling_and_variable_mismatch(dataset_path, tmp_path):
    reference = _dump(dataset_path, n_dates="all", n_points=10)
    shorter = build_dataset(tmp_path / "shorter.zarr", variables=VARIABLES[:-1], nt=NT - 2)
    stats = compute_stats(Dataset(shorter), Spec(sampling={"n_points": 10}), dates_from=reference)
    assert stats["sampling"]["missing_dates"] == reference["sampling"]["dates"][-2:]
    messages = [f.message for f in compare_stats(stats, reference).findings if f.severity == Severity.ERROR]
    assert ("sampled dates differ from the reference's: 2 missing, 0 extra (use --dates-from); "
            "sample statistics not compared") in messages
    assert not any(m.startswith("sample ") for m in messages)  # only the stored statistics are compared
    assert "variable missing (present in reference stats)" in messages
    other_points = compute_stats(Dataset(dataset_path),
                                 Spec(sampling={
                                     "n_points": 10,
                                     "seed": 1
                                 }),
                                 dates_from=reference)
    assert any("sample differs from the reference's" in f.message
               for f in compare_stats(other_points, reference).findings)


def _diurnal_dataset(path, nt=300):
    """A clean dataset where T_850 is "insolation": zero at night, skewed in time."""
    path = build_dataset(path, nt=nt)
    root = zarr.open_group(str(path), mode="r+")
    hours = np.arange(nt) % 24
    values = root["data"][:].astype("f8")
    values[:, 2] = np.where((hours >= 7) & (hours < 18), 1.0, 0.0)[:, None, None] * values[:, 2]
    root["data"][:] = values.astype("f4")
    v = root["data"][:].astype("f8")
    root["mean"][:], root["stdev"][:] = v.mean(axis=(0, 2, 3)), v.std(axis=(0, 2, 3))
    root["minimum"][:], root["maximum"][:] = v.min(axis=(0, 2, 3)), v.max(axis=(0, 2, 3))
    return path


def test_sampled_test_has_no_false_errors(tmp_path):
    """The distribution-free bound never raises an ERROR on a clean, skewed dataset."""
    path = _diurnal_dataset(tmp_path / "long.zarr")
    for seed in range(20):
        report = run(CheckContext(Dataset(path), Spec(sampling={
            "seed": seed,
            "n_dates": 8,
            "n_points": 20
        })),
                     bees=["stats"],
                     select=["statistics_consistent"])
        assert report.count(Severity.ERROR) == 0, (seed, report.findings)


def test_calibrate(tmp_path):
    path = _diurnal_dataset(tmp_path / "long.zarr")
    spec = Spec(sampling={"n_dates": 8, "n_points": 20})
    tolerances = calibrate(Dataset(path), spec, n_calib=120)
    assert tolerances == calibrate(Dataset(path), spec, n_calib=120) | {"created": tolerances["created"]}
    assert tolerances["sampling"]["n_estimate"] == 8 and tolerances["calibration"]["n_calib"] == 120
    sd = Dataset(path).statistic("stdev")
    for i, name in enumerate(VARIABLES[1:], start=1):  # FIS is constant in time: tolerance from the bias only
        tol = tolerances["variables"][name]
        default = 5 * sd[i] * (np.sqrt(1 - 8 / 300) / np.sqrt(8) + np.sqrt(1 - 20 / NG) / np.sqrt(20))
        assert 0 < tol["mean"] < default, (name, tol["mean"], default)
        assert tol["mean"] >= abs(tol["spatial_bias_mean"])


def test_calibrated_tolerances_in_the_check(tmp_path, write_spec):
    path = _diurnal_dataset(tmp_path / "long.zarr")
    spec_path = write_spec("sampling: {n_dates: 8, n_points: 20}\n")
    assert stats_bee_main(
        ["calibrate",
         str(path), "--spec",
         str(spec_path), "--n-calib", "150", "-o",
         str(tmp_path / "tol.yaml")]) == 0
    tol = yaml.safe_load((tmp_path / "tol.yaml").read_text())["variables"]["U_10M"]["mean"]

    def errors(seed=0):
        spec = load_spec(
            write_spec(f"sampling: {{n_dates: 8, n_points: 20, seed: {seed}}}\n"
                       "calibrated_tolerances: tol.yaml\n"))
        report = run(CheckContext(Dataset(path), spec), bees=["stats"], select=["statistics_consistent"])
        return [f for f in report.findings if f.severity == Severity.ERROR]

    assert errors() == []
    [f] = errors(seed=1)  # the tolerances were made for seed 0
    assert f.message.startswith("calibrated tolerances were made for another sampling")
    # a shift far below the distribution-free bound, but above the calibrated tolerance
    zarr.open_group(str(path), mode="r+")["mean"][3] += 3 * tol
    assert [f.variable for f in errors()] == ["U_10M"]


def test_cli(dataset_path, tmp_path, capsys):
    ref, new = tmp_path / "ref.yaml", tmp_path / "new.json"
    assert stats_bee_main(["dump", str(dataset_path), "-o", str(ref)]) == 0
    assert stats_bee_main(["dump", str(dataset_path), "--dates-from", str(ref), "-o", str(new)]) == 0
    assert stats_bee_main(["compare", str(new), str(ref)]) == 0
    assert "5/5 variables identical" in capsys.readouterr().out

    stats = json.loads(new.read_text())
    stats["stored"]["T_850"]["mean"] += 1
    new.write_text(json.dumps(stats))
    assert stats_bee_main(["compare", str(new), str(ref)]) == 1
    out = capsys.readouterr().out
    assert "T_850" in out and "stored mean differs" in out
    assert stats_bee_main(["compare", str(new), str(ref), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["ERROR"] == 1
    assert stats_bee_main(["check", str(dataset_path)]) == 0
    assert "checks run" in capsys.readouterr().out


def test_old_stats_files_are_rejected(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"format": "varda-hive-stats/2"}))
    with pytest.raises(ValueError, match="regenerate it with `stats-bee dump`"):
        load_stats(tmp_path / "old.json")
