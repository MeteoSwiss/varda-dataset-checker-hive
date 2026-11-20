# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""stats-bee's extra subcommand `stats-bee calibrate`, which writes calibrated tolerances, and its
part of `hive compare` (`compare_section`). `stats-bee check` is the generic one, see
varda_hive/cli.py; the stats dump is written by `--out`."""
from argparse import ArgumentParser, Namespace
from pathlib import Path

from ...honeycomb import Dataset, Finding, Spec, load_spec
from .calibrate import calibrate
from .statsfile import CLOSE, DIFFERS, IDENTICAL, compare_stats, load_stats, save_stats


def compare_section(dump: Path, reference: Path, spec: Spec, args: Namespace) -> tuple[list[Finding], str]:
    """`hive compare` of the stats sections of two ref files: the findings, and per variable the number
    of identical, close and differing values (variables with differences only, unless `args.all`)."""
    atol = args.atol_stdev if args.atol_stdev is not None else spec.tolerances.compare_atol_stdev
    comparison = compare_stats(load_stats(dump), load_stats(reference), atol)
    rows = [(v, c) for v, c in comparison.counts.items() if args.all or c[CLOSE] or c[DIFFERS]]
    if args.format == "json" or not rows:
        return comparison.findings, ""
    width = max(len(v) for v, _ in rows)
    if args.format == "markdown":
        lines = [f"| variable | {IDENTICAL} | {CLOSE} | {DIFFERS} |", "|---|---|---|---|"]
        lines += [f"| {v} | {c[IDENTICAL]} | {c[CLOSE]} | {c[DIFFERS]} |" for v, c in rows]
    else:
        lines = [f"{'variable':{width}s} {IDENTICAL:>9s} {CLOSE:>7s} {DIFFERS:>7s}"]
        lines += [f"{v:{width}s} {c[IDENTICAL]:9d} {c[CLOSE]:7d} {c[DIFFERS]:7d}" for v, c in rows]
    return comparison.findings, "\n".join(lines) + "\n"


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
    ("calibrate", "write calibrated tolerances for statistics_consistent", _add_calibrate_args, _calibrate),
]
