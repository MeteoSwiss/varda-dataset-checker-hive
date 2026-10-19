"""Render a Report as text (for humans) or JSON (for machines/CI)."""
import json

from .findings import Report, Severity

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
