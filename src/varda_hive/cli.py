"""Command-line entry points: `hive` runs all bees, `meta-bee`,
`stats-bee` and `spot-bee` run one each. All share the same options.

Exit code: 0 if no finding reaches --fail-on (default: error), 1 otherwise,
2 on usage errors.
"""
from argparse import ArgumentParser
import sys

from .honeycomb import BEES, CheckContext, Dataset, Severity, checks_for, load_spec, run
from .honeycomb.report import to_json, to_text


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
    return _main(["stats"], "stats-bee", "Check values and statistics of an anemoi dataset.", argv)


def spot_bee_main(argv=None) -> int:
    return _main(["spot"], "spot-bee", "Spot-check an anemoi dataset against its source.", argv)
