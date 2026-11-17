# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Render a Report as text (for humans), Markdown (CI job summaries, merge request comments) or JSON
(for machines)."""
import json

from .findings import Finding, Report, Severity

_MARK = {Severity.ERROR: "ERROR  ", Severity.WARNING: "WARNING", Severity.INFO: "info   "}


def to_text(report: Report, min_severity: Severity = Severity.INFO) -> str:
    lines = [f"dataset: {report.dataset}"]
    for f in report.findings:
        if f.severity < min_severity:
            continue
        where = ", ".join(x for x in (f.variable, f.time) if x)
        lines.append(f"{_MARK[f.severity]} {f.bee}.{f.check}: {f.message}" + (f" [{where}]" if where else ""))
    summary = ", ".join(f"{report.count(s)} {s.name.lower()}" for s in reversed(Severity))
    lines.append(f"{len(report.checks_run)} checks run: {summary}")
    return "\n".join(lines)


def to_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, default=str)


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(findings: list[Finding]) -> list[str]:
    lines = ["| severity | check | variable | time | message |", "|---|---|---|---|---|"]
    for f in findings:
        cells = (f.severity.name, f"`{f.bee}.{f.check}`", _md(f.variable or ""), _md(f.time or ""), _md(f.message))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(report: Report, details: bool = True) -> str:
    """A summary line, a table of the errors and warnings, and the info findings in a collapsed
    `<details>` block (left out with `details=False`)."""
    errors, warnings_ = report.count(Severity.ERROR), report.count(Severity.WARNING)
    verdict = "failed" if errors else ("passed with warnings" if warnings_ else "passed")
    lines = [
        f"### varda-hive: {verdict}", "", f"`{report.dataset}`", "",
        f"{len(report.checks_run)} checks run: **{errors} error**, **{warnings_} warning**, "
        f"{report.count(Severity.INFO)} info", ""
    ]
    serious = [f for f in report.findings if f.severity >= Severity.WARNING]
    if serious:
        lines += [*_md_table(serious), ""]
    rest = [f for f in report.findings if f.severity < Severity.WARNING]
    if details and rest:
        lines += [f"<details><summary>{len(rest)} info findings</summary>", "", *_md_table(rest), "", "</details>", ""]
    return "\n".join(lines)
