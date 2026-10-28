"""Command-line entry points:
- `hive <dataset>` runs all bees
- `meta-bee check`, `stats-bee check` and `spot-bee check` run one bee each, with the same options
- `stats-bee dump` writes a stats file and `stats-bee compare` compares two of them

Exit code: 0 if no finding reaches --fail-on (default: error), 1 otherwise,
2 on usage errors.
"""
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
import sys
from typing import Callable, Sequence

from .honeycomb import BEES, CheckContext, Dataset, Severity, checks_for, load_spec, run
from .honeycomb.findings import Report
from .honeycomb.report import to_json, to_text
from .honeycomb.statsfile import CLOSE, DIFFERS, IDENTICAL, compare_stats, compute_stats, load_stats, save_stats


BEE_DESCRIPTIONS = {
    "meta": "metadata and structure",
    "stats": "values and statistics",
    "spot": "sampled values vs the source",
}

# An extra subcommand of a bee: (name, help, add_args, handler), with handler(args) -> exit code.
Subcommand = tuple[str, str, Callable[[ArgumentParser], None], Callable[[Namespace], int]]


def _add_check_args(p: ArgumentParser, with_bees: bool) -> None:
    p.add_argument("dataset", nargs="?", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (see templates/spec.example.yaml)")
    p.add_argument("--reference", help="reference dataset, for checks that compare against one")
    p.add_argument("--select", nargs="+", metavar="CHECK", help="run only these checks")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--quiet", action="store_true", help="text output: hide info findings")
    p.add_argument("--list", action="store_true", help="list the bees and their checks and exit")
    if with_bees:
        p.add_argument("--bees", nargs="+", choices=BEES, default=list(BEES))


def _run_checks(args: Namespace, bees: list[str], prog: str) -> int:
    if args.list:
        width = max(len(c.name) for bee in bees for c in checks_for(bee))
        for bee in bees:
            print(f"{bee + '-bee':10s} {BEE_DESCRIPTIONS[bee]}")
            for c in checks_for(bee):
                print(f"  {c.name:{width}s}  {c.description}")
        return 0
    if not args.dataset:
        print(f"{prog}: error: the dataset argument is required", file=sys.stderr)
        return 2

    ctx = CheckContext(
        dataset=Dataset(args.dataset),
        spec=load_spec(args.spec),
        reference=Dataset(args.reference) if args.reference else None,
    )
    report = run(ctx, bees=bees, select=args.select)
    if args.format == "json":
        print(to_json(report))
    else:
        print(to_text(report, min_severity=Severity.WARNING if args.quiet else Severity.INFO))
    return report.exit_code(Severity[args.fail_on.upper()])


def hive_main(argv: list[str] | None = None) -> int:
    p = ArgumentParser(prog="hive", description="Run all bees on an anemoi dataset.")
    _add_check_args(p, with_bees=True)
    args = p.parse_args(argv)
    return _run_checks(args, args.bees, "hive")


def _bee_main(bee: str, description: str, argv: list[str] | None, extra: Sequence[Subcommand] = ()) -> int:
    """A bee's command: `check` plus the bee's `extra` subcommands."""
    prog = f"{bee}-bee"
    p = ArgumentParser(prog=prog, description=description)
    commands = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    check = commands.add_parser("check", help=f"run the {prog} checks on a dataset",
                                description=f"Run the {prog} checks on an anemoi dataset.")
    _add_check_args(check, with_bees=False)
    check.set_defaults(func=lambda args: _run_checks(args, [bee], f"{prog} check"))
    for name, help_text, add_args, handler in extra:
        sub = commands.add_parser(name, help=help_text, formatter_class=RawDescriptionHelpFormatter)
        add_args(sub)
        sub.set_defaults(func=handler)
    args = p.parse_args(argv)
    return args.func(args)


def meta_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("meta", "Check metadata and structure of an anemoi dataset.", argv)


def stats_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("stats", "Check values and statistics of an anemoi dataset, and write and compare "
                     "stats files. See docs/stats-bee.md.", argv, extra=[
                         ("dump", "write the stats file of a dataset", _add_dump_args, _dump),
                         ("compare", "compare a stats file with a reference stats file", _add_compare_args, _compare),
                     ])


def spot_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("spot", "Spot-check an anemoi dataset against its source.", argv)


def _add_dump_args(p: ArgumentParser) -> None:
    p.description = ("Write the stats file of an anemoi dataset: stored statistics, per-date summaries "
                     "and recomputed statistics on the sampled dates.")
    p.epilog = """examples:
  stats-bee dump data.zarr --spec spec.yaml -o checks/data.stats.yaml        # reference, commit it
  stats-bee dump new.zarr --spec spec.yaml --dates-from checks/data.stats.yaml -o new.stats.yaml

See docs/stats-bee.md for the file layout and how the tolerances work."""
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (sampling settings)")
    p.add_argument("--dates-from", metavar="STATS", help="use the dates of this stats file (e.g. the reference)")
    p.add_argument("-o", "--output", default="-",
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
