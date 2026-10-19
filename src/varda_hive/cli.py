"""Command-line entry points: `hive` runs all bees, `meta-bee`,
`stats-bee` and `spot-bee` run one each. All share the same options.
`stats-bee dump` writes a stats file and `stats-bee compare` compares two of them
(dispatched on the first argument, so a dataset path called `dump` must be written `./dump`).

Exit code: 0 if no finding reaches --fail-on (default: error), 1 otherwise,
2 on usage errors.
"""
from argparse import ArgumentParser
import sys

from .honeycomb import BEES, CheckContext, Dataset, Severity, checks_for, load_spec, run
from .honeycomb.findings import Report
from .honeycomb.report import to_json, to_text
from .honeycomb.statsfile import CLOSE, DIFFERS, IDENTICAL, compare_stats, compute_stats, load_stats, save_stats


def _parser(prog: str, description: str, with_bees: bool) -> ArgumentParser:
    p = ArgumentParser(prog=prog, description=description)
    p.add_argument("dataset", nargs="?", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (see templates/spec.example.yaml)")
    p.add_argument("--reference", help="reference dataset, for checks that compare against one")
    p.add_argument("--select", nargs="+", metavar="CHECK", help="run only these checks")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--quiet", action="store_true", help="text output: hide info findings")
    p.add_argument("--list", action="store_true", help="list the available checks and exit")
    if with_bees:
        p.add_argument("--bees", nargs="+", choices=BEES, default=list(BEES))
    return p


def _main(bees: list[str] | None, prog: str, description: str, argv=None) -> int:
    args = _parser(prog, description, with_bees=bees is None).parse_args(argv)
    bees = bees or args.bees

    if args.list:
        for bee in bees:
            for c in checks_for(bee):
                print(f"{bee}.{c.name:30s} {c.description}")
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


def hive_main(argv=None) -> int:
    return _main(None, "hive", "Run all bees on an anemoi dataset.", argv)


def meta_bee_main(argv=None) -> int:
    return _main(["meta"], "meta-bee", "Check metadata and structure of an anemoi dataset.", argv)


def stats_bee_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    subcommands = {"dump": _stats_dump, "compare": _stats_compare}
    if argv and argv[0] in subcommands:
        return subcommands[argv[0]](argv[1:])
    return _main(["stats"], "stats-bee", "Check values and statistics of an anemoi dataset. "
                 "Subcommands: `stats-bee dump` writes a stats file, `stats-bee compare` compares two "
                 "(see `stats-bee dump --help`).", argv)


def spot_bee_main(argv=None) -> int:
    return _main(["spot"], "spot-bee", "Spot-check an anemoi dataset against its source.", argv)


def _stats_dump(argv) -> int:
    p = ArgumentParser(prog="stats-bee dump", description="Write the stats file of an anemoi dataset: stored "
                       "statistics, per-date summaries and recomputed statistics on the sampled dates.")
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (sampling settings)")
    p.add_argument("--dates-from", metavar="STATS", help="use the dates of this stats file (e.g. the reference)")
    p.add_argument("-o", "--output", default="-", help="output JSON file (default: stdout)")
    args = p.parse_args(argv)
    dates_from = load_stats(args.dates_from) if args.dates_from else None
    save_stats(compute_stats(Dataset(args.dataset), load_spec(args.spec), dates_from), args.output)
    return 0


def _stats_compare(argv) -> int:
    p = ArgumentParser(prog="stats-bee compare", description="Compare a stats file with a reference stats file "
                       "and show what is identical and what differs.")
    p.add_argument("stats", help="stats file to check")
    p.add_argument("reference", help="reference stats file")
    p.add_argument("--spec", help="dataset spec YAML (tolerances.compare_atol_stdev)")
    p.add_argument("--atol-stdev", type=float, help="override tolerances.compare_atol_stdev")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--all", action="store_true", help="text output: list identical variables too")
    args = p.parse_args(argv)
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
