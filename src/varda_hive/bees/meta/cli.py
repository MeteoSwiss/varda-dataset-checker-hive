# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""meta-bee's part of `hive compare`: compare the meta sections of two ref files and render the
table of differences. `meta-bee check` is the generic one, see varda_hive/cli.py; meta-bee has no
other subcommands (its dump is written by `--out`)."""
from argparse import Namespace
from pathlib import Path

from ...honeycomb import Finding, Spec
from .metafile import IDENTICAL, IGNORED, MetaComparison, compare_meta, load_meta, shorten

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


def compare_section(dump: Path, reference: Path, spec: Spec, args: Namespace) -> tuple[list[Finding], str]:
    """`hive compare` of the meta sections of two ref files: the findings, and the table of differences
    (all items with `args.all`) in `args.format`."""
    comparison = compare_meta(load_meta(dump), load_meta(reference), spec.reference_policy)
    if args.format == "markdown":
        table = _markdown_table(comparison, args.all)
    elif args.format == "text":
        counts = ", ".join(f"{n} {status}" for status, n in comparison.counts.items() if n)
        table = _table(comparison, args.all) + f"meta: {counts}\n"
    else:
        table = ""
    return comparison.findings, table
