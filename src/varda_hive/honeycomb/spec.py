"""The per-dataset spec: what a dataset is expected to look like and where
its source data lives. Specs live in the dataset's own repo; see
templates/spec.example.yaml.

Every field is optional, so a spec can start small. Checks that need an
unset field skip themselves with an INFO finding.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class _Strict(BaseModel):
    # Typos in a spec should fail loudly, not silently disable a check.
    model_config = ConfigDict(extra="forbid")


class Sampling(_Strict):
    seed: int = 0
    # Random time indices, or "all" for a full pass (exact statistics, but reads every step).
    n_dates: int | Literal["all"] = 5
    n_points: int = 1000
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
    source_rtol: float = 1e-4
    source_atol: float = 1e-4


class SourceSpec(_Strict):
    # "package.module:Class" or "path/to/file.py:Class" (relative to the spec file)
    plugin: str
    args: dict[str, Any] = Field(default_factory=dict)


class Spec(_Strict):
    name: str | None = None
    variables: list[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    frequency: str | None = None
    grid_points: int | None = None
    constant_fields: list[str] | None = None
    ranges_file: Path | None = None
    # Stats file (written by `stats-bee dump`) to compare against, e.g. committed in the dataset repo.
    reference_stats: Path | None = None
    sampling: Sampling = Field(default_factory=Sampling)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    source: SourceSpec | None = None
    # Directory of the spec file; relative paths in the spec resolve against it.
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)


def load_spec(path: str | Path | None) -> Spec:
    if path is None:
        return Spec()
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    spec = Spec(**raw, base_dir=path.resolve().parent)
    for key in ("ranges_file", "reference_stats"):
        value = getattr(spec, key)
        if value is not None and not value.is_absolute():
            setattr(spec, key, spec.base_dir / value)
    return spec
