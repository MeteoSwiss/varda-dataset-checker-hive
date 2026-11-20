# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check registration and the runner shared by all bees.

A check is a function `fn(ctx: CheckContext) -> Iterable[Finding]`,
registered with `@check("meta")`. The runner fills in bee/check names on the
findings and turns a crashing check into an ERROR finding, so one broken
check never hides the results of the others.

A bee's dump is its section of a ref file: the facts about the dataset that it compares with the
reference (meta-bee: metadata and structure; stats-bee: statistics on the sample). It is a function
`fn(ctx) -> dict` registered with `@dumps("meta")`, built once per run (`bee_dump`, cached in
`ctx.cache`): the reference checks compare it with the reference, and `--out` writes it as
`ref.yaml` (`collect_dumps`). A check is a dump plus a comparison with the spec and the reference.
"""
from dataclasses import dataclass, field, replace
import importlib
from pathlib import Path
import traceback
from typing import Callable, Iterable

from .dataset import Dataset
from .findings import Finding, Report, Severity, error
from .spec import Spec

BEES = ("meta", "stats", "spot")


@dataclass
class CheckContext:
    dataset: Dataset
    spec: Spec
    reference: Path | None = None  # `--reference`, overrides the spec's `reference`
    cache: dict = field(default_factory=dict)  # shared between checks of one run, e.g. a loaded source

    @property
    def reference_path(self) -> Path | None:
        """The reference to compare with: `--reference`, else the spec's `reference`."""
        return self.reference or self.spec.reference

    def reference_missing(self) -> str | None:
        """Why there is no reference to compare with, or None if there is one. A spec `reference` that
        does not exist yet is not an error: it is the first reference, written by `--out`."""
        path = self.reference_path
        if path is None:
            return "no reference: give `--reference` or `reference` in the spec"
        if not path.exists():
            return f"reference {path} not found (write it with `--out` and commit `ref.yaml`)"
        return None


CheckFn = Callable[[CheckContext], Iterable[Finding]]
DumpFn = Callable[[CheckContext], dict]


@dataclass(frozen=True)
class CheckInfo:
    bee: str
    name: str
    fn: CheckFn
    description: str


_REGISTRY: dict[str, dict[str, CheckInfo]] = {bee: {} for bee in BEES}
_DUMPERS: dict[str, DumpFn] = {}


def check(bee: str, name: str | None = None) -> Callable[[CheckFn], CheckFn]:
    if bee not in _REGISTRY:
        raise ValueError(f"unknown bee {bee!r}, expected one of {BEES}")

    def register(fn: CheckFn) -> CheckFn:
        check_name = name or fn.__name__
        doc = (fn.__doc__ or "").strip()
        _REGISTRY[bee][check_name] = CheckInfo(bee, check_name, fn, doc.splitlines()[0] if doc else "")
        return fn

    return register


def dumps(bee: str) -> Callable[[DumpFn], DumpFn]:
    """Register `bee`'s dump: its section of a ref file."""
    if bee not in _REGISTRY:
        raise ValueError(f"unknown bee {bee!r}, expected one of {BEES}")

    def register(fn: DumpFn) -> DumpFn:
        _DUMPERS[bee] = fn
        return fn

    return register


def has_dump(bee: str) -> bool:
    """Whether `bee` has a dump, i.e. a section in a ref file."""
    if bee not in _REGISTRY:
        return False
    _load_bee(bee)
    return bee in _DUMPERS


def bee_dump(ctx: CheckContext, bee: str) -> dict:
    """`bee`'s dump of `ctx.dataset`, built once per run. A failure is kept too, so a dump that failed
    in one check is not built again by the next one or by `collect_dumps`."""
    key = ("dump", bee)
    if key not in ctx.cache:
        _load_bee(bee)
        try:
            ctx.cache[key] = _DUMPERS[bee](ctx)
        except Exception as e:
            ctx.cache[key] = e
            raise
    if isinstance(found := ctx.cache[key], Exception):
        raise found
    return found


def collect_dumps(ctx: CheckContext, bees: Iterable[str], report: Report) -> dict[str, dict]:
    """The dumps of `bees` (those that have one), as the sections of a ref file. A dump that fails is
    left out and reported as an ERROR finding of check `dump`."""
    sections = {}
    for bee in bees:
        if not has_dump(bee):
            continue
        try:
            sections[bee] = bee_dump(ctx, bee)
        except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            report.add(bee, "dump", [
                error(f"dump failed, no `{bee}` section in ref.yaml: {type(e).__name__}: {e}",
                      details={"traceback": traceback.format_exc()})
            ])
    return sections


def checks_for(bee: str) -> list[CheckInfo]:
    _load_bee(bee)
    return list(_REGISTRY[bee].values())


def run(ctx: CheckContext, bees: Iterable[str] = BEES, select: Iterable[str] | None = None) -> Report:
    """Run all checks of `bees`, or only those in `select` ("check" or "bee.check").

    Raises ValueError for a name in `select` that matches no check of `bees`: a typo must not turn
    into a run of zero checks that passes.
    """
    bees = list(bees)
    selected = _resolve_select(bees, select)
    report = Report(dataset=str(ctx.dataset.path))
    for bee in bees:
        for info in checks_for(bee):
            if selected is not None and (bee, info.name) not in selected:
                continue
            try:
                findings = [_apply_ignore(f, ctx.spec.ignore_warnings.get(info.name)) for f in info.fn(ctx)]
            # Reported, not swallowed.
            except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                findings = [
                    error(f"check crashed: {type(e).__name__}: {e}", details={"traceback": traceback.format_exc()})
                ]
            report.add(bee, info.name, findings)
    return report


def _resolve_select(bees: list[str], select: Iterable[str] | None) -> set[tuple[str, str]] | None:
    if not select:
        return None
    available = {(bee, c.name) for bee in bees for c in checks_for(bee)}
    selected, unknown = set(), []
    for name in select:
        bee, _, check_name = name.rpartition(".")
        matches = {(b, c) for b, c in available if c == check_name and bee in ("", b)}
        if not matches:
            unknown.append(name)
        selected |= matches
    if unknown:
        valid = ", ".join(sorted(f"{b}.{c}" for b, c in available))
        raise ValueError(f"unknown check(s) {', '.join(unknown)}; available: {valid}")
    return selected


def _apply_ignore(finding: Finding, variables: list[str] | None) -> Finding:
    """Demote a WARNING the spec lists in `ignore_warnings` to INFO."""
    if finding.severity != Severity.WARNING or not variables:
        return finding
    if "*" in variables or finding.variable in variables:
        return replace(finding, severity=Severity.INFO, message=f"{finding.message} (ignored by spec)")
    return finding


def _load_bee(bee: str) -> None:
    # Importing a bee's checks module registers its checks. Only that bee is imported, so the bees stay
    # independent. Imported by name, at call time, because the bees import the honeycomb themselves.
    importlib.import_module(f"..bees.{bee}.checks", __package__)
