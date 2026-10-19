"""The per-dataset spec: what a dataset is expected to look like and where
its source data lives. Specs live in the dataset's own repo; see
templates/spec.example.yaml.

Every field is optional, so a spec can start small. Checks that need an
unset field skip themselves with an INFO finding.
"""
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import yaml


class _Strict(BaseModel):
    # Typos in a spec should fail loudly, not silently disable a check.
    model_config = ConfigDict(extra="forbid")


class Sampling(_Strict):
    seed: int = 0
    n_dates: int = 5
    n_points: int = 1000
    # Time indices always included in the sample, in addition to the random ones.
    include_dates: list[int] = Field(default_factory=list)


class Tolerances(_Strict):
    statistics_rtol: float = 1e-3
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
    if spec.ranges_file is not None and not spec.ranges_file.is_absolute():
        spec.ranges_file = spec.base_dir / spec.ranges_file
    return spec
