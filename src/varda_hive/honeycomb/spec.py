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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from .dataset import parse_frequency


class _Strict(BaseModel):
    # Typos in a spec should fail loudly, not silently disable a check.
    model_config = ConfigDict(extra="forbid")


class Sampling(_Strict):
    seed: int = 0
    # Random dates on the period, or "all" for a full pass (exact statistics, but reads every step).
    n_dates: int | Literal["all"] = 5
    # Grid points per date, the same points on every date, or "all" for the full domain. stats-bee
    # computes mean/stdev on them (min/max and NaN/inf always cover the full domain); spot-bee
    # compares them with the source.
    n_points: int | Literal["all"] = 1000
    # Dates always included in the sample, in addition to the random ones.
    include_dates: list[datetime] = Field(default_factory=list)

    @field_validator("include_dates", mode="before")
    @classmethod
    def _dates_not_indices(cls, value: Any) -> Any:
        # pydantic would read an int as a unix timestamp; include_dates used to be time indices
        if isinstance(value, list) and any(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise ValueError("`include_dates` are dates (e.g. 2020-03-01T06:00:00), not time indices")
        return value


class Tolerances(_Strict):
    # Recomputed vs stored statistics: allowed |diff| = z_threshold * SE + statistics_rtol * (|stored| + stdev),
    # where SE is the sampling error of the recomputed value (0 for a full pass).
    statistics_rtol: float = 1e-5
    z_threshold: float = 5.0
    # Stats file vs reference stats file: |diff| <= compare_atol_stdev * stdev of the variable is
    # reported as "close" instead of "differs". 0 means only bitwise identical values pass.
    compare_atol_stdev: float = 0.0
    # spot-bee: dataset vs source values agree if |diff| <= source_atol + source_rtol * |source|.
    # atol 0: an absolute default would hide faults in small-valued fields (kg/kg, m); set it per variable.
    source_rtol: float = 1e-4
    source_atol: float = 0.0
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
    """The physical range of a variable, from its `min`/`max` in the spec."""
    min: float = -math.inf
    max: float = math.inf
    units: str = ""


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


class VariableSpec(BaseModel):
    """One entry of the spec's `variables` mapping. The typed keys are the hive's own; any other key
    (param, levtype, levelist, process, constant_in_time, ...) is expected in the dataset's
    `variables_metadata.<variable>`, at its top level or in its `mars` section."""
    model_config = ConfigDict(extra="allow")
    # Physical range (stats-bee physical_ranges) and units (meta-bee variables_metadata_match_spec).
    min: float | None = None
    max: float | None = None
    units: str | None = None
    # NaNs are allowed in this variable (the dataset's `allow_nans` lists it).
    allow_nans: bool = False
    # The variable never changes in time (the dataset's `constant_fields` lists it).
    constant: bool = False

    @model_validator(mode="after")
    def _valid(self) -> "VariableSpec":
        for key, value in (self.model_extra or {}).items():
            if isinstance(value, (dict, list)):
                raise ValueError(f"`{key}`: expected a scalar value, got {value!r}")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min} > max {self.max}")
        return self

    @property
    def metadata(self) -> dict[str, Scalar]:
        """The keys expected in the dataset's variables_metadata entry."""
        return dict(self.model_extra or {})


class Spec(_Strict):  # pylint: disable=too-many-instance-attributes
    name: str | None = None
    # A list of names, or a mapping {name: VariableSpec or null} with per-variable expectations.
    variables: list[str] | dict[str, VariableSpec] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    frequency: str | None = None
    grid_points: int | None = None
    resolution: str | None = None
    bbox: BBox | None = None
    # true/false, or the variables that may contain NaNs (as anemoi writes it). With a `variables`
    # mapping, leave it out and set `allow_nans: true` on the variables instead.
    allow_nans: bool | list[str] | None = None
    # Expected `missing_dates` attribute, usually [].
    missing_dates: list[datetime] | None = None
    # Tolerance file (written by `stats-bee calibrate`) for statistics_consistent.
    calibrated_tolerances: Path | None = None
    # Reference to compare with: a ref file (`ref.yaml` of `--out`, meta and stats sections) or a
    # zarr (meta-bee only). May not exist yet: the reference checks then skip (the first reference).
    reference: Path | None = None
    reference_policy: ReferencePolicy = Field(default_factory=ReferencePolicy)
    sampling: Sampling = Field(default_factory=Sampling)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    source: SourceSpec | None = None
    # Known, accepted warnings, demoted to INFO: {check name: [variables]}, "*" for all variables.
    # E.g. {statistics_consistent: [insolation]} for a diurnal field the sample cannot resolve.
    ignore_warnings: dict[str, list[str]] = Field(default_factory=dict)
    # Not read by the hive: a place for YAML anchors shared by several variables, e.g.
    # `anchors: {wind: &wind {min: -80, max: 80, units: m s-1}}`, then `U_10M: *wind`.
    anchors: Any = None
    # Directory of the spec file; relative paths in the spec resolve against it.
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @field_validator("frequency")
    @classmethod
    def _valid_frequency(cls, value: str | None) -> str | None:
        if value is not None:
            parse_frequency(value)  # ValueError -> ValidationError
        return value

    @field_validator("variables", mode="before")
    @classmethod
    def _empty_entries(cls, value: Any) -> Any:
        # `T_2M:` with nothing after it is an entry without expectations.
        if isinstance(value, dict):
            return {name: {} if entry is None else entry for name, entry in value.items()}
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "Spec":
        entries = self._entries()
        if self.allow_nans is not None and any(e.allow_nans for e in entries.values()):
            raise ValueError("`allow_nans` is set both for the spec and for variables: use one of them")
        names = self.variable_names()
        if isinstance(self.allow_nans, list) and names is not None:
            unknown = sorted(set(self.allow_nans) - set(names))
            if unknown:
                raise ValueError(f"`allow_nans` lists variables not in `variables`: {unknown}")
        return self

    # --- per-variable views, the same for a list and a mapping of variables ---

    def _entries(self) -> dict[str, VariableSpec]:
        return self.variables if isinstance(self.variables, dict) else {}

    def variable_names(self) -> list[str] | None:
        """The expected variables, None if the spec lists none."""
        return None if self.variables is None else list(self.variables)

    def ranges(self) -> dict[str, PhysicalRange]:
        """Physical ranges of the variables with a `min` or `max`."""
        return {
            name:
            PhysicalRange(min=-math.inf if e.min is None else e.min,
                          max=math.inf if e.max is None else e.max,
                          units=e.units or "")
            for name, e in self._entries().items() if e.min is not None or e.max is not None
        }

    def units(self) -> dict[str, str]:
        """Expected units of the variables that give them."""
        return {name: e.units for name, e in self._entries().items() if e.units}

    def expected_metadata(self) -> dict[str, dict[str, Scalar]]:
        """Expected variables_metadata keys of the variables that give any."""
        return {name: e.metadata for name, e in self._entries().items() if e.metadata}

    def constant_field_names(self) -> list[str] | None:
        """The variables marked `constant`, None without a `variables` mapping."""
        if not isinstance(self.variables, dict):
            return None
        return [name for name, e in self.variables.items() if e.constant]

    def nan_variables(self) -> bool | set[str] | None:
        """Where NaNs are allowed: everywhere (True), nowhere (False), in a set of variables, or None
        if the spec doesn't say. A `variables` mapping always says: the variables with `allow_nans`."""
        if isinstance(self.allow_nans, list):
            return set(self.allow_nans)
        if self.allow_nans is not None:
            return self.allow_nans
        if isinstance(self.variables, dict):
            return {name for name, e in self.variables.items() if e.allow_nans}
        return None


def load_spec(path: str | Path | None) -> Spec:
    """Load and validate a spec. `calibrated_tolerances` must exist, so a misconfigured spec fails here
    (CLI exit code 2) instead of in every check. `reference` may not exist yet (the first reference)."""
    if path is None:
        return Spec()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spec = Spec(**raw, base_dir=path.resolve().parent)
    for key in ("reference", "calibrated_tolerances"):
        value = getattr(spec, key)
        if value is None:
            continue
        if not value.is_absolute():
            value = spec.base_dir / value
            setattr(spec, key, value)
        # A missing reference is not an error: the reference checks skip, and `--out` writes it.
        if key == "calibrated_tolerances" and not value.is_file():
            raise FileNotFoundError(f"{path}: {key} not found: {value}")
    return spec
