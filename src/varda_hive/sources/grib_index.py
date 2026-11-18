# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee source reading GRIB messages through anemoi `grib-index` SQLite databases.

An anemoi `grib-index` DB (`anemoi-datasets grib-index --index x.db ...`) has two tables:
`paths(id, path)` and `grib_index(_id, _path_id, _offset, _length, valid_datetime, <keys>...)`, every
value stored as text. This source finds the message of a (variable, date) by `valid_datetime` and a
set of key = value matches, reads its bytes and decodes them with eccodes (the `grib` extra).

Which message belongs to a variable comes from the dataset's `variables_metadata[var].mars` by default:

- `levtype: pl`: shortName = param, level = levelist, typeOfLevel = isobaricInhPa
- otherwise: shortName = param

and from the spec args where that is not enough (renamed or rescaled fields, other indexes):

    source:
      plugin: varda_hive.sources.grib_index:GribIndexSource
      args:
        root: /path/to/dataset-repo        # GRIB paths in the DBs and the DB paths are relative to it
        indexes: {ana: ml_test/processed/grib-index-ana.db, fg: [day1.db, day2.db]}
        default_index: ana
        variables:
          W_SO: {index: fg, match: {shortName: W_SO, "level:d": "0.01"}}
          TOT_PREC_1H: {index: fg, match: {shortName: TOT_PREC}, scale: 0.001}
        skip: {TOT_PREC_6H: "6 h accumulation of TOT_PREC"}

Variables that cannot be compared one-to-one raise `NotComparable`: those in `skip`, and, unless
listed in `variables`, computed forcings, derived fields (metadata `process`, e.g. accumulations),
constant fields (they come from a separate file) and variables without mars metadata.
"""
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from ..honeycomb import Dataset, NotComparable


@dataclass(frozen=True)
class _Target:
    """Where a variable is found: an index, key = value matches and a factor applied to the values."""
    index: str
    match: dict[str, str]
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class _Index:
    dbs: list[Path]
    columns: set[str] = field(default_factory=set)


class GribIndexSource:  # pylint: disable=too-many-instance-attributes
    """Values of dataset variables from GRIB files, located through anemoi grib-index DBs."""

    def __init__(  # pylint: disable=too-many-arguments
            self,
            indexes: dict[str, str | list[str]],
            *,
            root: str | None = None,
            default_index: str | None = None,
            variables: dict[str, dict[str, Any]] | None = None,
            skip: dict[str, str] | None = None,
            pl_type_of_level: str = "isobaricInhPa",
            base_dir: str | Path | None = None):
        if not indexes:
            raise ValueError("GribIndexSource needs at least one entry in `indexes`")
        self.root = Path(base_dir or ".") / Path(root or ".")
        self.indexes = {
            name: _Index([self.root / p for p in ([dbs] if isinstance(dbs, str) else dbs)])
            for name, dbs in indexes.items()
        }
        self.default_index = default_index or next(iter(indexes))
        if self.default_index not in self.indexes:
            raise ValueError(f"default_index {self.default_index!r} is not one of {sorted(self.indexes)}")
        self.overrides = variables or {}
        self.skip = skip or {}
        self.pl_type_of_level = pl_type_of_level
        for db in (db for index in self.indexes.values() for db in index.dbs):
            if not db.is_file():
                raise FileNotFoundError(f"grib index {db} not found")
        self.targets: dict[str, _Target | str] = {}  # variable -> target, or the NotComparable reason
        self.n_members = 1
        self.n_grid: int | None = None
        self._rows: dict[tuple[str, str], list[dict[str, str]]] = {}  # (index, date) -> rows, latest date only

    def bind(self, dataset: Dataset) -> None:
        """Map every dataset variable to its GRIB messages (or a reason not to compare it)."""
        for index in self.indexes.values():
            index.columns = set.intersection(*(_columns(db) for db in index.dbs))
        self.n_members, self.n_grid = dataset.data.shape[2], dataset.data.shape[-1]
        metadata = dataset.attrs.get("variables_metadata", {})
        self.targets = {name: self._target(name, metadata.get(name, {})) for name in dataset.variables}
        if self.n_members > 1 and "number" not in set.union(*(i.columns for i in self.indexes.values())):
            raise ValueError("dataset has several ensemble members but the grib index has no `number` column")

    def _target(self, name: str, meta: dict[str, Any]) -> _Target | str:
        if name in self.skip:
            return str(self.skip[name])
        if name in self.overrides:
            override = dict(self.overrides[name])
            unknown = set(override) - {"index", "match", "scale", "offset"}
            if unknown:
                raise ValueError(f"source variable {name}: unknown keys {sorted(unknown)}")
            target = _Target(index=str(override.get("index", self.default_index)),
                             match={
                                 str(k): _text(v)
                                 for k, v in (override.get("match") or {}).items()
                             },
                             scale=float(override.get("scale", 1.0)),
                             offset=float(override.get("offset", 0.0)))
        else:
            reason = _not_comparable(meta)
            if reason:
                return reason
            mars = meta["mars"]
            match = {"shortName": _text(mars["param"])}
            if mars.get("levtype") == "pl":
                match |= {"level": _text(mars["levelist"]), "typeOfLevel": self.pl_type_of_level}
            target = _Target(index=self.default_index, match=match)
        if target.index not in self.indexes:
            raise ValueError(f"source variable {name}: index {target.index!r} is not one of {sorted(self.indexes)}")
        if not target.match:
            raise ValueError(f"source variable {name}: empty `match`")
        missing = set(target.match) - self.indexes[target.index].columns
        if missing:
            raise ValueError(f"source variable {name}: index {target.index!r} has no column {sorted(missing)}")
        return target

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        target = self.targets.get(variable)
        if target is None:
            raise NotComparable("unknown to the source (bind was not called or variable not in dataset)")
        if isinstance(target, str):
            raise NotComparable(target)
        rows = self._rows_at(target.index, str(np.datetime64(date, "s")))
        fields = []
        for member in range(self.n_members):
            match = target.match | ({"number": str(member)} if self.n_members > 1 else {})
            found = {(r["path"], r["_offset"], r["_length"]) for r in rows if all(r[k] == v for k, v in match.items())}
            if not found:
                return None
            if len(found) > 1:
                raise LookupError(f"{len(found)} GRIB messages match {match} in index {target.index!r}")
            fields.append(self._decode(*found.pop()))
        values = np.stack(fields) if self.n_members > 1 else fields[0]
        return values * target.scale + target.offset

    def _rows_at(self, index: str, date: str) -> list[dict[str, str]]:
        """All index rows valid at `date`, one query per (index, date): a sample has ~100 variables per date."""
        key = (index, date)
        if key not in self._rows:
            self._rows = {k: v for k, v in self._rows.items() if k[1] == date}
            rows: list[dict[str, str]] = []
            for db in self.indexes[index].dbs:
                with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
                    con.row_factory = sqlite3.Row
                    cur = con.execute(
                        "SELECT g.*, p.path FROM grib_index g JOIN paths p ON p.id = g._path_id "
                        "WHERE g.valid_datetime = ?", (date, ))
                    rows += [{k: _text(r[k]) for k in r.keys()} for r in cur]
            self._rows[key] = rows
        return self._rows[key]

    def _decode(self, path: str, offset: str, length: str) -> np.ndarray:
        try:
            import eccodes  # pylint: disable=import-outside-toplevel
        except (ImportError, RuntimeError) as exc:
            raise ImportError("GribIndexSource needs eccodes: install varda-dataset-checker-hive[grib]") from exc
        with open(self.root / path, "rb") as f:
            f.seek(int(offset))
            message = f.read(int(length))
        handle = eccodes.codes_new_from_message(message)
        try:
            n_points = eccodes.codes_get(handle, "numberOfDataPoints")
            if self.n_grid is not None and n_points != self.n_grid:
                raise ValueError(f"GRIB message in {path} has {n_points} points, dataset grid has {self.n_grid}")
            values = eccodes.codes_get_values(handle).astype(np.float64)
            if eccodes.codes_get(handle, "bitmapPresent"):
                values[values == eccodes.codes_get(handle, "missingValue")] = np.nan
        finally:
            eccodes.codes_release(handle)
        return values


def _not_comparable(meta: dict[str, Any]) -> str | None:
    """Why a variable has no one-to-one GRIB message, from its variables_metadata entry."""
    if meta.get("computed_forcing"):
        return "computed forcing"
    if meta.get("process"):
        period = meta.get("period")
        return f"derived field ({meta['process']}" + (f" over {period[0]}..{period[1]})" if period else ")")
    if meta.get("constant_in_time"):
        return "constant field, not in the time-indexed source"
    if not isinstance(meta.get("mars"), dict) or "param" not in meta["mars"]:
        return "no mars metadata"
    return None


def _columns(db: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(grib_index)")}
    if not columns:
        raise ValueError(f"{db} has no grib_index table")
    return columns


def _text(value: Any) -> str:
    """Index values are text; numbers in the spec or metadata (850, 0.01) compare as the index writes them."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if value is None else str(value)
