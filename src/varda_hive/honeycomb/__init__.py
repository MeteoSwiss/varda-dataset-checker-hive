# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared core of the hive: findings, dataset access, spec, registry, sampling, sources."""
from .dataset import Dataset
from .findings import Finding, Report, Severity, error, info, skipped, warning
from .registry import BEES, CheckContext, check, checks_for, run
from .sources import Source, load_source
from .spec import BBox, PhysicalRange, ReferencePolicy, Spec, load_ranges, load_spec, load_variables_file

__all__ = [
    "BBox",
    "BEES",
    "CheckContext",
    "Dataset",
    "Finding",
    "PhysicalRange",
    "ReferencePolicy",
    "Report",
    "Severity",
    "Source",
    "Spec",
    "check",
    "checks_for",
    "error",
    "info",
    "load_ranges",
    "load_source",
    "load_spec",
    "load_variables_file",
    "run",
    "skipped",
    "warning",
]
