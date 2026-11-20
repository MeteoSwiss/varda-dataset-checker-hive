# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Command-line entry points:
- `hive <dataset>` runs all bees. A check is a dump plus a comparison: each bee builds its dump (its
  section of a ref file, see varda_hive/honeycomb/files.py) once and compares it with the spec and the
  reference. `--out DIR` writes report.md, report.json and ref.yaml (the dump), also when checks fail.
- `meta-bee check`, `stats-bee check` and `spot-bee check` run one bee each, with the same options;
  their `--out` writes only that bee's section into ref.yaml
- `hive compare A.ref.yaml B.ref.yaml` compares two dumps without the data, with the same code as the
  reference checks (each bee's `compare_section` in its own cli.py)
- extra subcommands of a bee, like `stats-bee calibrate`, live in the bee's own `cli.py`, imported
  only when that bee's command runs, so one bee never loads another

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

from .honeycomb import (BEES, CheckContext, Dataset, Report, Severity, checks_for, collect_dumps, has_dump, load_spec,
                        run, skipped)
from .honeycomb.files import REF_FORMAT, file_format, read_file, save_ref
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
                   help="reference to compare with: a ref file (`ref.yaml` of `--out`), or a zarr (meta-bee "
                   "only); overrides the spec's `reference`")
    p.add_argument("--select", nargs="+", metavar="CHECK", help="run only these checks")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--quiet", action="store_true", help="text output: hide info findings")
    p.add_argument("--strict",
                   action="store_true",
                   help="fail when a check is skipped, e.g. for a missing spec field (recommended in CI)")
    p.add_argument("--out",
                   metavar="DIR",
                   help="write report.md, report.json and ref.yaml (the dump, to commit as the reference) to DIR")
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
    out = Path(args.out) if args.out else None
    if out is not None and ctx.reference_path is not None and (out /
                                                               "ref.yaml").resolve() == ctx.reference_path.resolve():
        raise ValueError(f"--out {out} would overwrite the reference {ctx.reference_path}: write to another directory "
                         "and copy ref.yaml once the differences are reviewed")
    report = run(ctx, bees=bees, select=args.select)
    sections = collect_dumps(ctx, bees, report) if out is not None else {}  # a failed dump is a finding
    _print_report(report, args)  # first, so a failure to write --out does not lose the findings
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(to_markdown(report) + "\n", encoding="utf-8")
        (out / "report.json").write_text(to_json(report) + "\n", encoding="utf-8")
        save_ref(sections, out / "ref.yaml")
    return report.exit_code(Severity[args.fail_on.upper()], strict=args.strict)


def _print_report(report: Report, args: Namespace, tables: str = "") -> None:
    if args.format == "json":
        print(to_json(report))
    elif args.format == "markdown":
        print(to_markdown(report, details=not tables) + ("\n" + tables if tables else ""))
    else:
        if tables:
            print(tables)
        print(to_text(report, min_severity=Severity.WARNING if getattr(args, "quiet", False) else Severity.INFO))


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


def hive_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compare"]:
        return _hive_compare_main(argv[1:])
    p = ArgumentParser(prog="hive",
                       formatter_class=RawDescriptionHelpFormatter,
                       description="Run all bees on an anemoi dataset.",
                       epilog="""examples:
  hive data.zarr --spec checks/data.yaml --strict --out hive/
      reference exists: spec and reference checks; hive/ref.yaml is the dataset's dump
      first reference:  the reference checks are skipped; commit hive/ref.yaml as the spec's `reference`
      intended change:  review hive/report.md, then cp hive/ref.yaml checks/data.ref.yaml
  hive compare a.ref.yaml b.ref.yaml        compare two dumps without the data""")
    _add_check_args(p, with_bees=True)
    args = p.parse_args(argv)
    return guarded("hive", lambda a: _run_checks(a, a.bees, "hive"))(args)


def _hive_compare_main(argv: list[str]) -> int:
    p = ArgumentParser(prog="hive compare",
                       formatter_class=RawDescriptionHelpFormatter,
                       description="Compare two ref files (`ref.yaml` of `--out`), section by section, with the "
                       "same comparison as the reference checks, and show what differs.",
                       epilog="""examples:
  hive compare hive/ref.yaml checks/data.ref.yaml
  hive compare hive/ref.yaml checks/data.ref.yaml --spec checks/data.yaml --format markdown --all

meta: differences are ERROR (structure: variables, shapes, dates, grid), WARNING (other metadata) or
INFO, churn (uuid, timestamps, provenance, recipe) is ignored; the spec's `reference_policy` overrides
this. stats: values are identical, close (|diff| <= atol-stdev * stdev) or differ. Exit code 1 if a
difference reaches --fail-on. See docs/meta-bee.md and docs/stats-bee.md.""")
    p.add_argument("dump", help="ref file to check")
    p.add_argument("reference", help="reference ref file")
    p.add_argument("--spec", help="dataset spec YAML (reference_policy, tolerances.compare_atol_stdev)")
    p.add_argument("--atol-stdev", type=float, help="override tolerances.compare_atol_stdev")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--all", action="store_true", help="list identical items too")
    p.add_argument("--strict", action="store_true", help="fail when a section is in only one of the files")
    return guarded("hive compare", _hive_compare)(p.parse_args(argv))


def _ref_sections(path: Path) -> list[str]:
    """The sections of the ref file `path`; ValueError for anything else, e.g. a zarr."""
    if path.is_dir():
        raise ValueError(f"{path}: a dataset, not a ref file: write its dump with `hive {path} --out DIR` and "
                         "compare DIR/ref.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    if file_format(path) != REF_FORMAT:
        raise ValueError(f"{path}: not a ref file of format {REF_FORMAT!r} (write one with `hive ... --out`)")
    return [key for key in read_file(path, REF_FORMAT, "ref file") if key != "format"]


def _hive_compare(args: Namespace) -> int:
    dump, reference = Path(args.dump), Path(args.reference)
    in_dump, in_reference, spec = _ref_sections(dump), _ref_sections(reference), load_spec(args.spec)
    report, tables = Report(dataset=f"{dump} vs {reference}"), []
    for section in sorted(set(in_dump) | set(in_reference),
                          key=lambda s: (s not in BEES, BEES.index(s) if s in BEES else 0, s)):
        if not has_dump(section):
            report.add(section, "compare", [skipped(f"no comparison for section `{section}`")])
        elif section not in in_dump or section not in in_reference:
            missing = dump if section not in in_dump else reference
            report.add(section, "compare", [skipped(f"no `{section}` section in {missing}")])
        else:
            # The bee loads its own section, checking its format.
            findings, table = _bee_cli(section).compare_section(dump, reference, spec, args)
            report.add(section, "compare", findings)
            if table:
                tables.append(table)
    _print_report(report, args, "\n".join(tables))
    return report.exit_code(Severity[args.fail_on.upper()], strict=args.strict)


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
    return _bee_main("meta", "Check metadata and structure of an anemoi dataset. See docs/meta-bee.md.", argv)


def stats_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("stats", "Check values and statistics of an anemoi dataset, and calibrate tolerances. "
                     "See docs/stats-bee.md.",
                     argv,
                     extra=_subcommands("stats"))


def spot_bee_main(argv: list[str] | None = None) -> int:
    return _bee_main("spot", "Spot-check an anemoi dataset against its source.", argv)
