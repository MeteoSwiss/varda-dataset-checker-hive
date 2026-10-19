"""Stats files: generation, round-trip and comparison, and `stats-bee dump` / `stats-bee compare`."""
import json

import numpy as np
import pytest
import zarr

from varda_hive.cli import stats_bee_main
from varda_hive.honeycomb import CheckContext, Dataset, Severity, Spec, run
from varda_hive.honeycomb.statsfile import compare_stats, compute_stats, load_stats, save_stats

from conftest import NT, VARIABLES, build_dataset


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
    assert len(stats["dates"]) in (3, 4) and stats["dates"][-1] == "2020-03-01T07:00:00"


def test_full_pass_recomputes_stored_statistics(dataset_path):
    stats = _dump(dataset_path, n_dates="all")
    for name in VARIABLES:
        stored, rec = stats["stored"][name], stats["recomputed"][name]
        assert rec["exact"] and rec["se_mean"] == 0
        for s in ("mean", "stdev", "minimum", "maximum"):
            assert np.isclose(rec[s], stored[s], rtol=1e-6, atol=1e-6), (name, s)


def test_yaml_and_json_compare_identical(dataset_path, tmp_path):
    stats = _dump(dataset_path, n_dates=4)
    save_stats(stats, tmp_path / "s.yaml")
    save_stats(stats, tmp_path / "s.json")
    assert "per_date:" in (tmp_path / "s.yaml").read_text()
    comparison = compare_stats(load_stats(tmp_path / "s.yaml"), load_stats(tmp_path / "s.json"))
    assert _severities(comparison) == [Severity.INFO]


def test_compare_identical(dataset_path):
    stats = _dump(dataset_path, n_dates=4)
    comparison = compare_stats(stats, stats)
    assert _severities(comparison) == [Severity.INFO]
    assert all(c["close"] == c["differs"] == 0 for c in comparison.counts.values())


def test_compare_differs_and_close(dataset_path):
    reference = _dump(dataset_path, n_dates=2, include_dates=[5])
    zarr.open_group(str(dataset_path), mode="r+")["data"][5, 1] += 0.01
    stats = compute_stats(Dataset(dataset_path), Spec(), dates_from=reference)
    assert stats["dates"] == reference["dates"]

    differs = [f for f in compare_stats(stats, reference).findings if f.severity == Severity.ERROR]
    assert {f.variable for f in differs} == {"T_2M"}
    assert {f.message.split()[0] for f in differs} == {"mean", "minimum", "maximum"}
    assert differs[0].time == "2020-03-01T05:00:00"

    close = compare_stats(stats, reference, atol_stdev=0.01)  # 0.01 * stdev (~5) covers the change
    assert Severity.ERROR not in _severities(close) and Severity.WARNING in _severities(close)


def test_compare_aligns_dates_not_positions(dataset_path, tmp_path):
    reference = _dump(dataset_path, n_dates=3)
    longer = build_dataset(tmp_path / "longer.zarr", nt=NT + 5)
    stats = compute_stats(Dataset(longer), Spec(), dates_from=reference)
    assert stats["dates"] == reference["dates"]
    assert not any("dates" in f.message for f in compare_stats(stats, reference).findings
                   if f.severity > Severity.INFO)


def test_compare_reports_missing_dates_and_variables(dataset_path, tmp_path):
    reference = _dump(dataset_path, n_dates="all")
    shorter = build_dataset(tmp_path / "shorter.zarr", variables=VARIABLES[:-1], nt=NT - 2)
    stats = compute_stats(Dataset(shorter), Spec(), dates_from=reference)
    assert stats["sampling"]["missing_dates"] == reference["dates"][-2:]
    messages = [f.message for f in compare_stats(stats, reference).findings if f.severity == Severity.ERROR]
    assert "2 reference dates not in stats (use --dates-from)" in messages
    assert "variable missing (present in reference stats)" in messages


def test_sampled_test_has_no_false_errors(tmp_path):
    """The distribution-free bound never raises an ERROR on a clean, skewed dataset."""
    path = build_dataset(tmp_path / "long.zarr", nt=300)
    data = zarr.open_group(str(path), mode="r+")["data"]
    hours = np.arange(300) % 24
    values = data[:].astype("f8")
    values[:, 2] = np.where((hours >= 7) & (hours < 18), 1.0, 0.0)[:, None, None] * values[:, 2]  # "insolation"
    data[:] = values.astype("f4")
    root = zarr.open_group(str(path), mode="r+")
    v = data[:].astype("f8")
    root["mean"][:], root["stdev"][:] = v.mean(axis=(0, 2, 3)), v.std(axis=(0, 2, 3))
    root["minimum"][:], root["maximum"][:] = v.min(axis=(0, 2, 3)), v.max(axis=(0, 2, 3))
    for seed in range(20):
        report = run(CheckContext(Dataset(path), Spec(sampling={"seed": seed, "n_dates": 8})),
                     bees=["stats"], select=["statistics_consistent"])
        assert report.count(Severity.ERROR) == 0, (seed, report.findings)


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
    assert stats_bee_main([str(dataset_path)]) == 0  # without a subcommand: run the checks
    assert "checks run" in capsys.readouterr().out
