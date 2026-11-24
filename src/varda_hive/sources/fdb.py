# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""spot-bee source reading GRIB fields from an FDB with pyfdb (the `fdb` extra).

The args follow the anemoi `fdb` source of the recipe, so the spec can copy them:

    source:
      plugin: varda_hive.sources.fdb:FdbSource
      args:
        fdb_config: null           # dict as in the recipe; null: pyfdb's default (FDB5_CONFIG_FILE)
        request: {class: rd, stream: reanl, expver: r001, model: icon-rea-l-ch1, type: cf}
        offset_from_date: true     # date=YYYYMMDD, time=0000, step=offset from midnight
        step_zero_from_previous_date: [VMAX_10M]   # step 0 -> previous date, step 24 (true: all)
        variables:                 # overrides of the default request, merged on top of it
          CLCL: {request: {levtype: pl, levelist: 800}}
          VMAX_10M: {}             # compare it although its metadata says `process: maximum`
        skip: {CIN_MU: masked below -999 by apply_mask_fields}
        members: [1, 2]            # GRIB `number` of each dataset member, default 0, 1, ... (none for one member)

The request of a variable is `request` + its `variables_metadata` entry + the time keys:
`param` is `grib.paramId` (FDB indexes paramIds), else `mars.param`; `levtype` is `mars.levtype`;
`levelist` is `mars.levelist` unless null. An override's `request` is merged on top, and `scale`
and `offset` apply to the values (value * scale + offset).

Not comparable (`NotComparable`), unless the variable has an override: those in `skip`, computed
forcings, derived fields (metadata `process`), constant fields, variables without mars metadata,
and `levtype: unknown` (e.g. interpolated from model to pressure levels).

`fetch` makes one retrieve per (variable, date, member) and decodes the message with eccodes; no
data is None (missing), several messages for one request an error. GRIB values are in message
order, which is the dataset order for unstructured (ICON) grids: anemoi's `grid_definition` adds
coordinates but does not reorder. For regular and reduced lat/lon and Gaussian grids the
coordinates are compared with the dataset's, as in GribIndexSource.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..honeycomb import Dataset
from ._grib import Decoder, dataset_coords, load_eccodes, map_targets, member_numbers, not_comparable, resolve, text

_OVERRIDE_KEYS = {"request", "scale", "offset"}


@dataclass(frozen=True)
class _Target:
    """The FDB request of a variable (without time keys) and a factor applied to its values."""
    request: dict[str, str] = field(default_factory=dict)
    scale: float = 1.0
    offset: float = 0.0


class FdbSource:  # pylint: disable=too-many-instance-attributes
    """Values of dataset variables retrieved from an FDB."""

    def __init__(  # pylint: disable=too-many-arguments
            self,
            request: dict[str, Any],
            *,
            fdb_config: dict | str | None = None,
            fdb_userconfig: dict | str | None = None,
            offset_from_date: bool = False,
            step_zero_from_previous_date: bool | list[str] = False,
            variables: dict[str, dict[str, Any] | None] | None = None,
            skip: dict[str, str] | None = None,
            members: list[int] | None = None,
            check_grid: bool = True):
        if not isinstance(request, dict):
            raise ValueError("FdbSource `request` must be a mapping, e.g. {class: rd, stream: reanl}")
        self.request = {str(k): text(v) for k, v in request.items()}
        self.fdb_config = fdb_config
        self.fdb_userconfig = fdb_userconfig
        self.offset_from_date = offset_from_date
        self.step_zero_from_previous_date = step_zero_from_previous_date
        self.overrides = {name: dict(o or {}) for name, o in (variables or {}).items()}
        self.skip = skip or {}
        self.members = None if members is None else [int(m) for m in members]
        self.check_grid = check_grid
        # variable -> target, the NotComparable reason, or the configuration error
        self.targets: dict[str, _Target | str] = {}
        self.numbers: list[str] = []
        self._fdb: Any = None
        self._decoder: Decoder | None = None

    def bind(self, dataset: Dataset) -> None:
        """Open the FDB and map every dataset variable to its request (or a reason not to compare it)."""
        # eccodes before pyfdb: the other order segfaults with the FDB uenv's libraries.
        eccodes = load_eccodes("FdbSource", "fdb")
        try:
            import pyfdb  # pylint: disable=import-outside-toplevel
        except (ImportError, RuntimeError, OSError) as exc:
            raise ImportError(f"FdbSource needs pyfdb and the FDB library: install varda-dataset-checker-hive[fdb] "
                              f"and load FDB (e.g. the fdb uenv): {exc}") from exc
        self._fdb = pyfdb.FDB(config=self.fdb_config, user_config=self.fdb_userconfig)
        self.numbers = member_numbers(self.members, dataset.data.shape[2])
        self._decoder = Decoder(eccodes, dataset.data.shape[-1], dataset_coords(dataset, self.check_grid))
        self.targets = map_targets(dataset, self._target)

    def _target(self, name: str, meta: dict[str, Any]) -> _Target | str:
        if name in self.skip:
            return str(self.skip[name])
        if name not in self.overrides:
            reason = not_comparable(meta)
            if reason is None and meta["mars"].get("levtype") == "unknown":
                reason = "levtype unknown (interpolated, e.g. from model to pressure levels)"
            if reason:
                return reason
            return _Target(request=self.request | _request_from_metadata(meta))
        override = self.overrides[name]
        unknown = set(override) - _OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}, expected {sorted(_OVERRIDE_KEYS)}")
        extra = override.get("request") or {}
        if not isinstance(extra, dict):
            raise TypeError("`request` must be a mapping")
        request = self.request | (_request_from_metadata(meta) if isinstance(meta.get("mars"), dict) else {})
        request |= {str(k): text(v) for k, v in extra.items()}
        if not request.get("param"):
            raise ValueError("no `param`: give it in the override's `request`")
        return _Target(request=request,
                       scale=float(override.get("scale", 1.0)),
                       offset=float(override.get("offset", 0.0)))

    def request_for(self, variable: str, date: np.datetime64) -> dict[str, str]:
        """The full FDB request of `variable` at `date` (without the member `number`)."""
        target = resolve(self.targets, variable)
        when = np.datetime64(date, "s").astype(datetime)
        return target.request | time_request_keys(when, self.offset_from_date, self._step_zero(variable))

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        target = resolve(self.targets, variable)
        if self._fdb is None or self._decoder is None:
            raise RuntimeError("FdbSource.bind was not called")
        request = self.request_for(variable, date)
        fields = []
        numbers: list[str | None] = [*self.numbers] or [None]
        for number in numbers:
            full = request | ({"number": number} if number is not None else {})
            # The context manager opens the handle in pyfdb 5 (read() alone does in pyfdb 0.1).
            with self._fdb.retrieve(full) as handle:
                data = bytes(handle.read())
            if not data:
                return None
            where = "FDB " + ",".join(f"{k}={v}" for k, v in full.items())
            n = self._decoder.count(data)
            if n > 1:
                raise LookupError(f"{n} GRIB messages match {full}")
            fields.append(self._decoder.decode(data, where))
        values = np.stack(fields) if self.numbers else fields[0]
        return values * target.scale + target.offset

    def _step_zero(self, variable: str) -> bool:
        flag = self.step_zero_from_previous_date
        return variable in flag if isinstance(flag, list) else bool(flag)


def _request_from_metadata(meta: dict[str, Any]) -> dict[str, str]:
    """param (paramId if known), levtype and levelist of a variables_metadata entry, those it has."""
    mars = meta["mars"]
    grib: dict[str, Any] = meta["grib"] if isinstance(meta.get("grib"), dict) else {}
    param = grib.get("paramId") or mars.get("param")
    out = {"param": text(param)} if param is not None else {}
    if mars.get("levtype") is not None:
        out["levtype"] = text(mars["levtype"])
    if mars.get("levelist") is not None:
        out["levelist"] = text(mars["levelist"])
    return out


def time_request_keys(when: datetime, offset_from_date: bool, step_zero_from_previous_date: bool) -> dict[str, str]:
    """date/time/step of the field valid at `when`, as the anemoi fdb source requests it.

    With `offset_from_date`, the run of the day at 00:00 and the step since midnight; at midnight,
    with `step_zero_from_previous_date`, step 24 of the previous day's run instead (for fields
    without a step 0, e.g. a maximum over the preceding interval). Otherwise time = HHMM, no step."""
    out = {"date": when.strftime("%Y%m%d")}
    if not offset_from_date:
        out["time"] = when.strftime("%H%M")
        return out
    out["time"] = "0000"
    offset = when - when.replace(hour=0, minute=0, second=0, microsecond=0)
    out["step"] = step(offset)
    if not offset and step_zero_from_previous_date:
        out["date"] = (when - timedelta(days=1)).strftime("%Y%m%d")
        out["step"] = step(timedelta(hours=24))
    return out


def step(offset: timedelta) -> str:
    """A step as FDB indexes it: whole hours as "6", otherwise "10m" or "23h30m"."""
    seconds = offset.total_seconds()
    if seconds < 0 or seconds % 60:
        raise ValueError(f"step must be a non-negative whole number of minutes, got {offset}")
    hours, minutes = divmod(int(seconds // 60), 60)
    if minutes == 0:
        return str(hours)
    return f"{minutes}m" if hours == 0 else f"{hours}h{minutes}m"
