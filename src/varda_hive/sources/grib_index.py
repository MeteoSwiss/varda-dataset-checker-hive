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
- otherwise: shortName = param, and level = levelist if the metadata has a levelist and the index a
  `level` column

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
        members: [1, 2, 3]                 # GRIB `number` of each dataset member, default 0, 1, ... (none for
                                           # a single member); also for one member, e.g. [5]

A configuration error of one variable (unknown override key or index, a `match` column the index
lacks, ensemble data without a `number` column) is raised by `fetch` for that variable only, so
spot-bee reports it as an ERROR of that variable and still compares the others.

GRIB values are returned in message order. For grids whose coordinates eccodes can compute
(regular and reduced lat/lon and Gaussian grids), the coordinates of each distinct grid are compared
once with the dataset's latitudes/longitudes, so a source in another order is an ERROR of the
variable rather than "differ" everywhere. Unstructured (ICON) and rotated grids are not checked.

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

from ..honeycomb import Dataset
from ._grib import Decoder, dataset_coords, load_eccodes, map_targets, member_numbers, not_comparable, resolve
from ._grib import text as _text


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
            members: list[int] | None = None,
            check_grid: bool = True,
            base_dir: str | Path | None = None):
        if not indexes:
            raise ValueError("GribIndexSource needs at least one entry in `indexes`")
        self.root = Path(base_dir or ".") / Path(root or ".")
        self.indexes = {
            name: _Index([self.root / p for p in ([dbs] if isinstance(dbs, str) else dbs)])
            for name, dbs in indexes.items()
        }
        empty = sorted(name for name, index in self.indexes.items() if not index.dbs)
        if empty:
            raise ValueError(f"grib index {empty} lists no DB")
        self.default_index = default_index or next(iter(indexes))
        if self.default_index not in self.indexes:
            raise ValueError(f"default_index {self.default_index!r} is not one of {sorted(self.indexes)}")
        self.overrides = variables or {}
        self.skip = skip or {}
        self.pl_type_of_level = pl_type_of_level
        self.members = None if members is None else [int(m) for m in members]
        self.check_grid = check_grid
        for db in (db for index in self.indexes.values() for db in index.dbs):
            if not db.is_file():
                raise FileNotFoundError(f"grib index {db} not found")
        # variable -> target, the NotComparable reason, or the configuration error
        self.targets: dict[str, _Target | str] = {}
        self.numbers: list[str] = []  # GRIB `number` of each dataset member, empty for a single member
        self.n_members = 1
        self.n_grid: int | None = None
        self._rows: dict[tuple[str, str], list[dict[str, str]]] = {}  # (index, date) -> rows, latest date only
        self._decoder: Decoder | None = None

    def bind(self, dataset: Dataset) -> None:
        """Map every dataset variable to its GRIB messages (or a reason not to compare it)."""
        eccodes = load_eccodes("GribIndexSource", "grib")
        for index in self.indexes.values():
            index.columns = set.intersection(*(_columns(db) for db in index.dbs))
        self.n_members, self.n_grid = dataset.data.shape[2], dataset.data.shape[-1]
        self.numbers = member_numbers(self.members, self.n_members)
        self._decoder = Decoder(eccodes, self.n_grid, dataset_coords(dataset, self.check_grid))
        self.targets = map_targets(dataset, self._target)

    def _target(self, name: str, meta: dict[str, Any]) -> _Target | str:
        if name in self.skip:
            return str(self.skip[name])
        if name in self.overrides:
            override = dict(self.overrides[name])
            unknown = set(override) - {"index", "match", "scale", "offset"}
            if unknown:
                raise ValueError(f"unknown keys {sorted(unknown)}")
            target = _Target(index=str(override.get("index", self.default_index)),
                             match={
                                 str(k): _text(v)
                                 for k, v in (override.get("match") or {}).items()
                             },
                             scale=float(override.get("scale", 1.0)),
                             offset=float(override.get("offset", 0.0)))
        else:
            reason = not_comparable(meta)
            if reason:
                return reason
            mars = meta["mars"]
            match = {"shortName": _text(mars["param"])}
            levelist = mars.get("levelist")
            if mars.get("levtype") == "pl":
                if levelist is None:
                    raise ValueError("pressure-level variable without `levelist` in its mars metadata")
                match |= {"level": _text(levelist), "typeOfLevel": self.pl_type_of_level}
            elif levelist is not None and "level" in self.indexes[self.default_index].columns:
                match["level"] = _text(levelist)
            target = _Target(index=self.default_index, match=match)
        if target.index not in self.indexes:
            raise ValueError(f"index {target.index!r} is not one of {sorted(self.indexes)}")
        if not target.match:
            raise ValueError("empty `match`")
        columns = self.indexes[target.index].columns
        missing = set(target.match) - columns
        if missing:
            raise ValueError(f"index {target.index!r} has no column {sorted(missing)}")
        if self.numbers and "number" not in columns:
            raise ValueError(f"`number` is matched ({self.n_members} dataset members) but index {target.index!r} "
                             "has no `number` column")
        return target

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        target = resolve(self.targets, variable)
        rows = self._rows_at(target.index, str(np.datetime64(date, "s")))
        fields = []
        numbers: list[str | None] = [*self.numbers] or [None]
        for number in numbers:
            match = target.match | ({"number": number} if number is not None else {})
            found = {(r["_hive_path"], r["_offset"], r["_length"])
                     for r in rows if all(r[k] == v for k, v in match.items())}
            if not found:
                return None
            if len(found) > 1:
                raise LookupError(f"{len(found)} GRIB messages match {match} in index {target.index!r}")
            fields.append(self._decode(*found.pop()))
        values = np.stack(fields) if self.numbers else fields[0]
        return values * target.scale + target.offset

    def _rows_at(self, index: str, date: str) -> list[dict[str, str]]:
        """All index rows valid at `date`, one query per (index, date): a sample has ~100 variables per date."""
        key = (index, date)
        if key not in self._rows:
            self._rows = {k: v for k, v in self._rows.items() if k[1] == date}
            rows: list[dict[str, str]] = []
            for db in self.indexes[index].dbs:
                with closing(_connect(db)) as con:
                    con.row_factory = sqlite3.Row
                    # `_hive_path`: grib_index may itself have a `path` key column.
                    cur = con.execute(
                        "SELECT g.*, p.path AS _hive_path FROM grib_index g JOIN paths p ON p.id = g._path_id "
                        "WHERE g.valid_datetime = ?", (date, ))
                    rows += [{k: _text(r[k]) for k in r.keys()} for r in cur]
            self._rows[key] = rows
        return self._rows[key]

    def _decode(self, path: str, offset: str, length: str) -> np.ndarray:
        if self._decoder is None:
            raise RuntimeError("GribIndexSource.bind was not called")
        with open(self.root / path, "rb") as f:
            f.seek(int(offset))
            message = f.read(int(length))
        return self._decoder.decode(message, path)


def _connect(db: Path) -> sqlite3.Connection:
    """A read-only connection; the URI form escapes `?`, `#` and `%` in the path."""
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def _columns(db: Path) -> set[str]:
    with closing(_connect(db)) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(grib_index)")}
    if not columns:
        raise ValueError(f"{db} has no grib_index table")
    return columns
