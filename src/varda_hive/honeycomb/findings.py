"""What a check reports: `Finding`s, collected into a `Report`.

Checks never raise or assert to signal a problem; they yield findings with a
severity. That keeps one check usable from the CLI, CI, pytest or a notebook,
and lets expected noise (e.g. run-to-run ICON differences) be a WARNING
instead of a failure.
"""
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any, Iterable


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class Finding:
    severity: Severity
    message: str
    variable: str | None = None
    time: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    # Filled in by the runner, so checks don't have to repeat their own name.
    bee: str = ""
    check: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


def error(message: str, **kwargs) -> Finding:
    return Finding(Severity.ERROR, message, **kwargs)


def warning(message: str, **kwargs) -> Finding:
    return Finding(Severity.WARNING, message, **kwargs)


def info(message: str, **kwargs) -> Finding:
    return Finding(Severity.INFO, message, **kwargs)


@dataclass
class Report:
    dataset: str
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def add(self, bee: str, check: str, findings: Iterable[Finding]) -> None:
        self.checks_run.append(f"{bee}.{check}")
        self.findings.extend(replace(f, bee=bee, check=check) for f in findings)

    def count(self, severity: Severity) -> int:
        return sum(f.severity == severity for f in self.findings)

    def exit_code(self, fail_on: Severity = Severity.ERROR) -> int:
        return int(any(f.severity >= fail_on for f in self.findings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "checks_run": self.checks_run,
            "summary": {s.name: self.count(s) for s in Severity},
            "findings": [f.to_dict() for f in self.findings],
        }
