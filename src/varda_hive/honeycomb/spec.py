# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The per-dataset spec: what a dataset is expected to look like and where
its source data lives. Specs live in the dataset's own repo; see
templates/spec.example.yaml.

Every field is optional, so a spec can start small. Checks that need an
unset field skip themselves with an INFO finding.
"""
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from .dataset import parse_frequency


class _Strict(BaseModel):
    # Typos in a spec should fail loudly, not silently disable a check.
    model_config = ConfigDict(extra="forbid")


class Sampling(_Strict):
    seed: int = 0
    # Random time indices, or "all" for a full pass (exact statistics, but reads every step).
    n_dates: int | Literal["all"] = 5
    # Grid points per date, the same points on every date, or "all" for the full domain. stats-bee
    # computes mean/stdev on them (min/max and NaN/inf always cover the full domain); spot-bee
    # compares them with the source.
    n_points: int | Literal["all"] = 1000
    # Time indices always included in the sample, in addition to the random ones.
    include_dates: list[int] = Field(default_factory=list)


class Tolerances(_Strict):
    # Recomputed vs stored statistics: allowed |diff| = z_threshold * SE + statistics_rtol * (|stored| + stdev),
    # where SE is the sampling error of the recomputed value (0 for a full pass).
    statistics_rtol: float = 1e-5
    z_threshold: float = 5.0
    # Stats file vs reference stats file: |diff| <= compare_atol_stdev * stdev of the variable is
    # reported as "close" instead of "differs". 0 means only bitwise identical values pass.
    compare_atol_stdev: float = 0.0
    # spot-bee: dataset vs source values agree if |diff| <= source_atol + source_rtol * |source|.
    source_rtol: float = 1e-4
    source_atol: float = 1e-4
    # Per-variable overrides of source_rtol/source_atol, e.g. {TOT_PREC_1H: {atol: 1.0e-6}}.
    source_per_variable: dict[str, "SourceTolerance"] = Field(default_factory=dict)

    def source_tolerance(self, variable: str) -> tuple[float, float]:
        """(rtol, atol) for comparing `variable` with the source."""
        override = self.source_per_variable.get(variable)
        rtol = override.rtol if override and override.rtol is not None else self.source_rtol
        atol = override.atol if override and override.atol is not None else self.source_atol
        return rtol, atol


class SourceTolerance(_Strict):
    rtol: float | None = None
    atol: float | None = None


Tolerances.model_rebuild()


class SourceSpec(_Strict):
    # "package.module:Class" or "path/to/file.py:Class" (relative to the spec file)
    plugin: str
    args: dict[str, Any] = Field(default_factory=dict)


class PhysicalRange(_Strict):
    """One entry of a `ranges_file`."""
    min: float = -math.inf
    max: float = math.inf
    units: str = ""


def load_ranges(path: Path) -> dict[str, PhysicalRange]:
    """Read and validate a `ranges_file`: {variable: {min, max, units}}, unknown keys rejected."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of variable -> {{min, max, units}}")
    return {str(name): PhysicalRange(**(bounds or {})) for name, bounds in raw.items()}


class BBox(_Strict):
    """Expected extent of the grid in degrees: the min/max of the latitudes and longitudes."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Allowed |difference| in degrees for each of the four bounds.
    tolerance: float = 0.01


class ReferencePolicy(_Strict):
    """Overrides of the default severity of a difference to the reference (meta-bee). Each entry is
    a glob on the dotted path of the item, e.g. `variables_metadata.*.mars.date` or `arrays.sums.*`.
    Checked in this order (ignore, error, warning, info) before the defaults."""
    ignore: list[str] = Field(default_factory=list)
    error: list[str] = Field(default_factory=list)
    warning: list[str] = Field(default_factory=list)
    info: list[str] = Field(default_factory=list)


Scalar = str | int | float | bool | None


def load_variables_file(path: Path) -> dict[str, dict[str, Scalar]]:
    """Read and validate a `variables_file`: {variable: {key: value}}, e.g. {T_850: {param: T,
    levtype: pl, levelist: 850}}. Keys are free (they follow the dataset's variables_metadata), values
    must be scalars."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of variable -> {{key: value}}")
    out: dict[str, dict[str, Scalar]] = {}
    for name, entry in raw.items():
        entry = entry or {}
        if not isinstance(entry, dict) or any(isinstance(v, (dict, list)) for v in entry.values()):
            raise ValueError(f"{path}: {name}: expected a mapping of key -> scalar value")
        out[str(name)] = {str(k): v for k, v in entry.items()}
    return out


class Spec(_Strict):
    name: str | None = None
    variables: list[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    frequency: str | None = None
    grid_points: int | None = None
    resolution: str | None = None
    bbox: BBox | None = None
    allow_nans: bool | None = None
    # Expected `missing_dates` attribute, usually [].
    missing_dates: list[datetime] | None = None
    # Expected per-variable metadata (mars param, levtype, levelist, ...), see load_variables_file.
    variables_file: Path | None = None
    constant_fields: list[str] | None = None
    ranges_file: Path | None = None
    # Stats file (written by `stats-bee dump`) to compare against, e.g. committed in the dataset repo.
    reference_stats: Path | None = None
    # Tolerance file (written by `stats-bee calibrate`) for statistics_consistent.
    calibrated_tolerances: Path | None = None
    # meta-bee: reference to compare the structure with, a meta file (`meta-bee dump`) or a zarr.
    reference_meta: Path | None = None
    reference_policy: ReferencePolicy = Field(default_factory=ReferencePolicy)
    sampling: Sampling = Field(default_factory=Sampling)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    source: SourceSpec | None = None
    # Known, accepted warnings, demoted to INFO: {check name: [variables]}, "*" for all variables.
    # E.g. {statistics_consistent: [insolation]} for a diurnal field the sample cannot resolve.
    ignore_warnings: dict[str, list[str]] = Field(default_factory=dict)
    # Directory of the spec file; relative paths in the spec resolve against it.
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @field_validator("frequency")
    @classmethod
    def _valid_frequency(cls, value: str | None) -> str | None:
        if value is not None:
            parse_frequency(value)  # ValueError -> ValidationError
        return value


def load_spec(path: str | Path | None) -> Spec:
    """Load and validate a spec. Files it names must exist, and the ranges file must be valid, so a
    misconfigured spec fails here (CLI exit code 2) instead of in every check."""
    if path is None:
        return Spec()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spec = Spec(**raw, base_dir=path.resolve().parent)
    for key in ("ranges_file", "reference_stats", "calibrated_tolerances", "variables_file", "reference_meta"):
        value = getattr(spec, key)
        if value is None:
            continue
        if not value.is_absolute():
            value = spec.base_dir / value
            setattr(spec, key, value)
        # reference_meta may be a zarr directory
        if not (value.exists() if key == "reference_meta" else value.is_file()):
            raise FileNotFoundError(f"{path}: {key} not found: {value}")
    if spec.ranges_file is not None:
        load_ranges(spec.ranges_file)
    if spec.variables_file is not None:
        load_variables_file(spec.variables_file)
    return spec
