# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared core of the hive: findings, dataset access, spec, registry, sampling, sources."""
from .dataset import Dataset
from .findings import Finding, Report, Severity, error, info, skipped, warning
from .registry import BEES, CheckContext, bee_dump, check, checks_for, collect_dumps, dumps, has_dump, run
from .sources import NotComparable, Source, load_source
from .spec import BBox, PhysicalRange, ReferencePolicy, Spec, VariableSpec, load_spec

__all__ = [
    "BBox",
    "BEES",
    "CheckContext",
    "Dataset",
    "Finding",
    "NotComparable",
    "PhysicalRange",
    "ReferencePolicy",
    "Report",
    "Severity",
    "Source",
    "Spec",
    "VariableSpec",
    "bee_dump",
    "check",
    "checks_for",
    "collect_dumps",
    "dumps",
    "has_dump",
    "error",
    "info",
    "load_source",
    "load_spec",
    "run",
    "skipped",
    "warning",
]
