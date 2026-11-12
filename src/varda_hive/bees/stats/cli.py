# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""stats-bee's extra subcommands: `stats-bee dump` writes a stats file, `stats-bee compare`
compares two of them and `stats-bee calibrate` writes calibrated tolerances. `stats-bee check` is
the generic one, see varda_hive/cli.py."""
from argparse import ArgumentParser, Namespace

from ...honeycomb import Dataset, Report, Severity, load_spec
from ...honeycomb.report import to_json, to_text
from .calibrate import calibrate
from .statsfile import CLOSE, DIFFERS, IDENTICAL, compare_stats, compute_stats, load_stats, save_stats


def _add_dump_args(p: ArgumentParser) -> None:
    p.description = ("Write the stats file of an anemoi dataset: its stored statistics, and the statistics of "
                     "the sample of dates x points. The file size depends only on the number of variables.")
    p.epilog = """examples:
  stats-bee dump data.zarr --spec spec.yaml -o checks/data.stats.yaml        # reference, commit it
  stats-bee dump new.zarr --spec spec.yaml --dates-from checks/data.stats.yaml -o new.stats.yaml

See docs/stats-bee.md for the file layout and how the tolerances work."""
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (sampling settings)")
    p.add_argument("--dates-from",
                   metavar="STATS",
                   help="sample the dates of this stats file (e.g. the reference), so the two compare")
    p.add_argument("-o",
                   "--output",
                   default="-",
                   help="output file, YAML for .yaml/.yml, otherwise JSON (default: JSON to stdout)")


def _dump(args: Namespace) -> int:
    dates_from = load_stats(args.dates_from) if args.dates_from else None
    save_stats(compute_stats(Dataset(args.dataset), load_spec(args.spec), dates_from), args.output)
    return 0


def _add_compare_args(p: ArgumentParser) -> None:
    p.description = ("Compare a stats file with a reference stats file and show what is identical and what "
                     "differs. Exit code 1 if anything differs.")
    p.epilog = """examples:
  stats-bee compare new.stats.yaml checks/data.stats.yaml
  stats-bee compare new.stats.yaml checks/data.stats.yaml --atol-stdev 1e-6 --all

Values are identical (bitwise), close (|diff| <= atol-stdev * stdev of the variable) or differ.
See docs/stats-bee.md."""
    p.add_argument("stats", help="stats file to check")
    p.add_argument("reference", help="reference stats file")
    p.add_argument("--spec", help="dataset spec YAML (tolerances.compare_atol_stdev)")
    p.add_argument("--atol-stdev", type=float, help="override tolerances.compare_atol_stdev")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--all", action="store_true", help="text output: list identical variables too")


def _compare(args: Namespace) -> int:
    stats, reference = load_stats(args.stats), load_stats(args.reference)
    atol = args.atol_stdev if args.atol_stdev is not None else load_spec(args.spec).tolerances.compare_atol_stdev
    comparison = compare_stats(stats, reference, atol)
    report = Report(dataset=f"{stats['dataset']['name']} vs {reference['dataset']['name']}")
    report.add("stats", "compare", comparison.findings)
    if args.format == "json":
        print(to_json(report))
    else:
        rows = [(v, c) for v, c in comparison.counts.items() if args.all or c[CLOSE] or c[DIFFERS]]
        if rows:
            width = max(len(v) for v, _ in rows)
            print(f"{'variable':{width}s} {IDENTICAL:>9s} {CLOSE:>7s} {DIFFERS:>7s}")
            for v, c in rows:
                print(f"{v:{width}s} {c[IDENTICAL]:9d} {c[CLOSE]:7d} {c[DIFFERS]:7d}")
            print()
        print(to_text(report))
    return report.exit_code(Severity[args.fail_on.upper()])


def _add_calibrate_args(p: ArgumentParser) -> None:
    p.description = ("Calibrate the tolerances of statistics_consistent for the spec's sampling: read a large "
                     "calibration sample of dates, simulate the check's sample on it, and write the largest "
                     "sampling error per variable (times a margin), plus the bias of the point subset.")
    p.epilog = """examples:
  stats-bee calibrate data.zarr --spec checks/data.yaml --n-calib 300 -o checks/data.tolerances.yaml

Then set `calibrated_tolerances: data.tolerances.yaml` in the spec. Recalibrate whenever the
sampling (seed, n_dates, n_points) changes. See docs/stats-bee.md."""
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (sampling settings)")
    p.add_argument("--n-calib", type=int, default=300, help="dates to read for calibration (default: 300)")
    p.add_argument("--n-sim", type=int, default=2000, help="simulated samples (default: 2000)")
    p.add_argument("--margin", type=float, default=2.0, help="factor on the largest simulated error (default: 2)")
    p.add_argument("-o",
                   "--output",
                   default="-",
                   help="output file, YAML for .yaml/.yml, otherwise JSON (default: JSON to stdout)")


def _calibrate(args: Namespace) -> int:
    tolerances = calibrate(Dataset(args.dataset), load_spec(args.spec), args.n_calib, args.n_sim, args.margin)
    save_stats(tolerances, args.output)
    return 0


# (name, help, add_args, handler), see `Subcommand` in varda_hive/cli.py.
SUBCOMMANDS = [
    ("dump", "write the stats file of a dataset", _add_dump_args, _dump),
    ("compare", "compare a stats file with a reference stats file", _add_compare_args, _compare),
    ("calibrate", "write calibrated tolerances for statistics_consistent", _add_calibrate_args, _calibrate),
]
