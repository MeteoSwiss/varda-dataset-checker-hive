# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""meta-bee's extra subcommands: `meta-bee dump` writes a meta file, `meta-bee compare` compares two
datasets or meta files. `meta-bee check` is the generic one, see varda_hive/cli.py."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from ...honeycomb import Dataset, Report, Severity, Spec, load_spec
from ...honeycomb.report import to_json, to_markdown
from .metafile import (IDENTICAL, IGNORED, MetaComparison, compare_meta, fingerprint, load_reference, save_meta,
                       shorten)


def _add_dump_args(p: ArgumentParser) -> None:
    p.description = ("Write the meta file of an anemoi dataset: its attributes, and the shape, dtype and a hash "
                     "of every array. Commit it next to the spec as the reference of the accepted version.")
    p.epilog = """examples:
  meta-bee dump data.zarr -o checks/data.meta.yaml      # then set `reference: data.meta.yaml` in the spec

`hive dump` writes the meta and stats references into one ref file.

See docs/meta-bee.md."""
    p.add_argument("dataset", help="path to the anemoi zarr dataset")
    p.add_argument("-o",
                   "--output",
                   default="-",
                   help="output file, YAML for .yaml/.yml, otherwise JSON (default: JSON to stdout)")


def dump_section(dataset: Dataset, spec: Spec, dates_from: Path | None) -> dict[str, Any]:  # pylint: disable=unused-argument
    """meta-bee's section of a ref file (`hive dump`): the meta file."""
    return fingerprint(dataset)


def _dump(args: Namespace) -> int:
    save_meta(fingerprint(Dataset(args.dataset)), args.output)
    return 0


def _add_compare_args(p: ArgumentParser) -> None:
    p.description = ("Compare the metadata and structure of a dataset with a reference and show what is "
                     "identical and what differs. Each can be a zarr dataset or a meta file.")
    p.epilog = """examples:
  meta-bee compare new.zarr checks/data.meta.yaml
  meta-bee compare new.zarr checks/data.ref.yaml        # the meta section of a ref file (`hive dump`)
  meta-bee compare new.zarr published.zarr --all
  meta-bee compare new.zarr checks/data.meta.yaml --spec checks/data.yaml --format markdown

Differences are ERROR (structure: variables, shapes, dates, grid), WARNING (other metadata) or INFO,
and churn (uuid, timestamps, provenance, recipe) is ignored; the spec's `reference_policy` overrides
this. Exit code 1 if a difference reaches --fail-on. See docs/meta-bee.md."""
    p.add_argument("dataset", help="dataset (zarr), meta file or ref file to check")
    p.add_argument("reference", help="reference dataset (zarr), meta file or ref file")
    p.add_argument("--spec", help="dataset spec YAML (reference_policy)")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    p.add_argument("--fail-on", choices=("error", "warning"), default="error")
    p.add_argument("--all", action="store_true", help="list identical and ignored items too")


HEADER = ("group", "item", "reference", "dataset", "status")


def _cell(value: object) -> str:
    return "" if value is None else shorten(value, 48)


def _rows(comparison: MetaComparison, show_all: bool) -> list[tuple[str, ...]]:
    """One row per item of the compare table (differences only, unless `show_all`), as `HEADER`."""
    items = sorted((i for i in comparison.items if show_all or i.status not in (IDENTICAL, IGNORED)),
                   key=lambda i: (i.group, i.path))
    return [(i.group, i.path, _cell(i.reference), _cell(i.note or i.actual),
             i.status + (f" ({i.severity.name})" if i.severity is not None else "")) for i in items]


def _table(comparison: MetaComparison, show_all: bool) -> str:
    if not (rows := _rows(comparison, show_all)):
        return ""
    widths = [max(len(r[k]) for r in [HEADER, *rows]) for k in range(len(HEADER))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in [HEADER, *rows]]
    return "\n".join(lines) + "\n"


def _markdown_table(comparison: MetaComparison, show_all: bool) -> str:
    if not (rows := _rows(comparison, show_all)):
        return ""
    lines = ["| " + " | ".join(HEADER) + " |", "|---" * len(HEADER) + "|"]
    for group, path, *cells in rows:
        lines.append("| " + " | ".join([group, f"`{path}`", *(c.replace("|", "\\|") for c in cells)]) + " |")
    return "\n".join(lines) + "\n"


def _compare(args: Namespace) -> int:
    actual, reference = load_reference(args.dataset), load_reference(args.reference)
    comparison = compare_meta(actual, reference, load_spec(args.spec).reference_policy)
    report = Report(dataset=f"{actual['dataset']['name']} vs {reference['dataset']['name']}")
    report.add("meta", "compare", comparison.findings)
    if args.format == "json":
        print(to_json(report))
    elif args.format == "markdown":
        table = _markdown_table(comparison, args.all)
        print(to_markdown(report, details=False) + ("\n" + table if table else ""))
    else:
        print(report.dataset)
        print(_table(comparison, args.all))
        counts = ", ".join(f"{n} {status}" for status, n in comparison.counts.items() if n)
        severities = ", ".join(f"{report.count(sev)} {sev.name.lower()}" for sev in (Severity.ERROR, Severity.WARNING))
        print(f"{counts}; differences: {severities}")
    return report.exit_code(Severity[args.fail_on.upper()])


# (name, help, add_args, handler), see `Subcommand` in varda_hive/cli.py.
SUBCOMMANDS = [
    ("dump", "write the meta file of a dataset", _add_dump_args, _dump),
    ("compare", "compare metadata and structure with a reference", _add_compare_args, _compare),
]
