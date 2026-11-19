# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Command-line entry points:
- `hive <dataset>` runs all bees
- `hive dump <dataset>` writes a ref file: the `dump` of each bee (meta file, stats file) in one file,
  see varda_hive/honeycomb/files.py
- `meta-bee check`, `stats-bee check` and `spot-bee check` run one bee each, with the same options
- extra subcommands of a bee, like `stats-bee dump` or `meta-bee compare`, live in the bee's own
  `cli.py` (see varda_hive/bees/stats/cli.py and varda_hive/bees/meta/cli.py), imported only when that
  bee's command runs, so one bee never loads another

Exit codes, so CI can tell a bad dataset from a broken job:
- 0: no finding reaches --fail-on (default: error), and with --strict no check was skipped
- 1: the dataset failed the checks
- 2: usage or configuration error (missing dataset, invalid spec, unknown check name, ...)
- 3: internal error of the tool (a bug: please report it with the traceback)

A crash inside one check is not an internal error: it is reported as an ERROR finding (exit 1).
"""
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
import functools
import importlib
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Sequence

import yaml

from .honeycomb import BEES, CheckContext, Dataset, Severity, checks_for, load_spec, run
from .honeycomb.files import save_ref
from .honeycomb.report import to_json, to_markdown, to_text

BEE_DESCRIPTIONS = {
    "meta": "metadata and structure",
    "stats": "values and statistics",
    "spot": "sampled values vs the source",
}

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERNAL = 0, 1, 2, 3

# An extra subcommand of a bee: (name, help, add_args, handler), with handler(args) -> exit code.
Subcommand = tuple[str, str, Callable[[ArgumentParser], None], Callable[[Namespace], int]]


def _add_check_args(p: ArgumentParser, with_bees: bool) -> None:
    p.add_argument("dataset", nargs="?", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (see templates/spec.example.yaml)")
    p.add_argument("--reference",
                   help="reference to compare with: a ref file (`hive dump`), a meta or stats file, or a zarr "
                   "(meta-bee only); overrides the spec's `reference`")
    p.add_argument("--select", nargs="+", metavar="CHECK", help="run only these checks")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--quiet", action="store_true", help="text output: hide info findings")
    p.add_argument("--strict",
                   action="store_true",
                   help="fail when a check is skipped, e.g. for a missing spec field (recommended in CI)")
    p.add_argument("--report", metavar="FILE", help="also write the JSON report to FILE")
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
        return EXIT_USAGE

    reference = Path(args.reference) if args.reference else None
    if reference is not None and not reference.exists():  # a usage error (exit 2), not a check crash
        raise FileNotFoundError(f"reference not found: {reference}")
    ctx = CheckContext(dataset=Dataset(args.dataset), spec=load_spec(args.spec), reference=reference)
    report = run(ctx, bees=bees, select=args.select)
    if args.report:
        Path(args.report).write_text(to_json(report) + "\n", encoding="utf-8")
    if args.format == "json":
        print(to_json(report))
    elif args.format == "markdown":
        print(to_markdown(report))
    else:
        print(to_text(report, min_severity=Severity.WARNING if args.quiet else Severity.INFO))
    return report.exit_code(Severity[args.fail_on.upper()], strict=args.strict)


def guarded(prog: str, fn: Callable[[Namespace], int]) -> Callable[[Namespace], int]:
    """Map exceptions escaping `fn` (i.e. outside the checks) to exit codes 2 and 3."""

    @functools.wraps(fn)
    def wrapper(args: Namespace) -> int:
        try:
            return fn(args)
        # Configuration problems: one line, no traceback. pydantic's ValidationError is a ValueError.
        except (OSError, ValueError, yaml.YAMLError) as e:
            print(f"{prog}: error: {e}", file=sys.stderr)
            return EXIT_USAGE
        except Exception:  # pylint: disable=broad-exception-caught
            traceback.print_exc()
            print(f"{prog}: internal error, please report it with the traceback above", file=sys.stderr)
            return EXIT_INTERNAL

    return wrapper


# Bees with a section in a ref file, i.e. a `dump_section` in their cli.py.
DUMP_BEES = ("meta", "stats")


def hive_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["dump"]:
        return _hive_dump_main(argv[1:])
    p = ArgumentParser(prog="hive",
                       description="Run all bees on an anemoi dataset.",
                       epilog="`hive dump <dataset> --spec S -o X.ref.yaml` writes the reference of all bees.")
    _add_check_args(p, with_bees=True)
    args = p.parse_args(argv)
    return guarded("hive", lambda a: _run_checks(a, a.bees, "hive"))(args)


def _hive_dump_main(argv: list[str]) -> int:
    p = ArgumentParser(prog="hive dump",
                       formatter_class=RawDescriptionHelpFormatter,
                       description="Write the ref file of an anemoi dataset: the meta file (`meta-bee dump`) and the "
                       "stats file (`stats-bee dump`) in one file. Commit it next to the spec and set "
                       "`reference: <name>.ref.yaml` in the spec.",
                       epilog="""examples:
  hive dump data.zarr --spec checks/data.yaml -o checks/data.ref.yaml
  hive dump new.zarr --spec checks/data.yaml --dates-from checks/data.ref.yaml -o new.ref.yaml""")
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("--spec", help="dataset spec YAML (sampling settings)")
    p.add_argument("--dates-from",
                   metavar="REF",
                   help="sample the dates of this ref or stats file (e.g. the reference), so the two compare")
    p.add_argument("--bees", nargs="+", choices=DUMP_BEES, default=list(DUMP_BEES), help="sections to write")
    p.add_argument("-o",
                   "--output",
                   default="-",
                   help="output file, YAML for .yaml/.yml, otherwise JSON (default: JSON to stdout)")
    return guarded("hive dump", _hive_dump)(p.parse_args(argv))


def _hive_dump(args: Namespace) -> int:
    dataset, spec = Dataset(args.dataset), load_spec(args.spec)
    dates_from = Path(args.dates_from) if args.dates_from else None
    if dates_from is not None and not dates_from.is_file():
        raise FileNotFoundError(f"--dates-from not found: {dates_from}")
    sections = {bee: _bee_cli(bee).dump_section(dataset, spec, dates_from) for bee in args.bees}
    save_ref(sections, args.output)
    return EXIT_OK


def _bee_cli(bee: str) -> Any:
    """`bee`'s own cli.py; imported by name so only this bee is loaded."""
    return importlib.import_module(f".bees.{bee}.cli", __package__)


def _subcommands(bee: str) -> Sequence[Subcommand]:
    """The extra subcommands of `bee`, from its own `cli.py`."""
    return _bee_cli(bee).SUBCOMMANDS


def _bee_main(bee: str, description: str, argv: list[str] | None, extra: Sequence[Subcommand] = ()) -> int:
    """A bee's command: `check` plus the bee's `extra` subcommands."""
    prog = f"{bee}-bee"
    p = ArgumentParser(prog=prog, description=description)
    commands = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    check = commands.add_parser("check",
                                help=f"run the {prog} checks on a dataset",
                                description=f"Run the {prog} checks on an anemoi dataset.")
    _add_check_args(check, with_bees=False)
    check.set_defaults(func=guarded(f"{prog} check", lambda args: _run_checks(args, [bee], f"{prog} check")))
    for name, help_text, add_args, handler in extra:
        sub = commands.add_parser(name, help=help_text, formatter_class=RawDescriptionHelpFormatter)
        add_args(sub)
        sub.set_defaults(func=guarded(f"{prog} {name}", handler))
    args = p.parse_args(argv)
    return args.func(args)


def meta_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("meta", "Check metadata and structure of an anemoi dataset, and write and compare meta "
                     "files. See docs/meta-bee.md.",
                     argv,
                     extra=_subcommands("meta"))


def stats_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("stats", "Check values and statistics of an anemoi dataset, and write and compare "
                     "stats files. See docs/stats-bee.md.",
                     argv,
                     extra=_subcommands("stats"))


def spot_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("spot", "Spot-check an anemoi dataset against its source.", argv)
