"""Check registration and the runner shared by all bees.

A check is a function `fn(ctx: CheckContext) -> Iterable[Finding]`,
registered with `@check("meta")`. The runner fills in bee/check names on the
findings and turns a crashing check into an ERROR finding, so one broken
check never hides the results of the others.
"""
from dataclasses import dataclass, field
import importlib
import traceback
from typing import Callable, Iterable

from .dataset import Dataset
from .findings import Finding, Report, error
from .spec import Spec

BEES = ("meta", "stats", "spot")


@dataclass
class CheckContext:
    dataset: Dataset
    spec: Spec
    reference: Dataset | None = None
    cache: dict = field(default_factory=dict)  # shared between checks of one run, e.g. a loaded source


CheckFn = Callable[[CheckContext], Iterable[Finding]]


@dataclass(frozen=True)
class CheckInfo:
    bee: str
    name: str
    fn: CheckFn
    description: str


_REGISTRY: dict[str, dict[str, CheckInfo]] = {bee: {} for bee in BEES}


def check(bee: str, name: str | None = None) -> Callable[[CheckFn], CheckFn]:
    if bee not in _REGISTRY:
        raise ValueError(f"unknown bee {bee!r}, expected one of {BEES}")

    def register(fn: CheckFn) -> CheckFn:
        check_name = name or fn.__name__
        doc = (fn.__doc__ or "").strip()
        _REGISTRY[bee][check_name] = CheckInfo(bee, check_name, fn, doc.splitlines()[0] if doc else "")
        return fn

    return register


def checks_for(bee: str) -> list[CheckInfo]:
    _load_bees()
    return list(_REGISTRY[bee].values())


def run(ctx: CheckContext, bees: Iterable[str] = BEES, select: Iterable[str] | None = None) -> Report:
    """Run all checks of `bees` (optionally only the check names in `select`)."""
    selected = set(select) if select else None
    report = Report(dataset=str(ctx.dataset.path))
    for bee in bees:
        for info in checks_for(bee):
            if selected is not None and info.name not in selected:
                continue
            try:
                findings = list(info.fn(ctx))
            # Reported, not swallowed.
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                findings = [error(f"check crashed: {type(e).__name__}: {e}",
                                  details={"traceback": traceback.format_exc()})]
            report.add(bee, info.name, findings)
    return report


def _load_bees() -> None:
    # Importing the bee packages registers their checks. Imported by name, at call time, because
    # the bees import the honeycomb themselves.
    importlib.import_module("..bees", __package__)
