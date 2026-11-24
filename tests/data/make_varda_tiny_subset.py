# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Write `varda-tiny-subset.zarr`, the small real dataset that `tests/test_detection.py` corrupts, and
its GRIB source for spot-bee (`grib/`, `grib-index-*.db`).

A subset of Varda-tiny (varda-kenda-ch1-2km-2020-2020-1h-v1.0-pl13-test, MeteoSwiss, CC BY 4.0):
a few variables, 48 hourly dates around 2020-03-05T04 (time index 100, where the Varda-tiny faults
are) and every 61st grid point. The statistics window is the first 40 dates; mean, stdev, minimum
and maximum are recomputed over it as anemoi-datasets does.

The GRIB source holds the original analysis (and, for TOT_PREC, first-guess) messages of these
variables and dates, cut to the same grid points and stored as 32-bit IEEE floats, so they decode to
exactly the values of the original messages. ICON's unstructured grid (template 101) carries no
coordinates, so only the number of points changes. The two grib-index DBs have the columns of the
originals, with rows for these messages only. TD_2M is there for the `m-varmeta-param` fault.

Not run by the tests; rerun by hand when the source dataset or the subset changes:

    uv run python tests/data/make_varda_tiny_subset.py <ich1-analysis-anemoi-dataset checkout>
"""
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import eccodes
import zarr

from varda_hive.honeycomb import Dataset

VARIABLES = ["FIS", "T_2M", "T_700", "T_850", "U_10M", "V_10M", "PS", "TOT_PREC_1H", "CLCT"]
DATES = slice(72, 120)  # 2020-03-04T00 to 2020-03-05T23; index 100 of the source is 28 here
WINDOW = 40  # dates in the statistics window
POINT_STRIDE = 61
DROP_ATTRS = ("provenance_load", "recipe", "data_request", "total_number_of_files", "total_size")
DATASET = "ml_test/datasets/varda-kenda-ch1-2km-2020-2020-1h-v1.0-pl13-test.zarr"
# index -> SQL condition selecting the messages to keep
GRIB = {
    "ana": "shortName IN ('T_2M', 'TD_2M', 'U_10M', 'V_10M', 'PS', 'CLCT') "
    "OR (shortName = 'T' AND typeOfLevel = 'isobaricInhPa' AND level IN ('700', '850'))",
    "fg": "shortName = 'TOT_PREC'",
}


def main(checkout: Path, out: Path) -> None:
    write_zarr(checkout / DATASET, out / "varda-tiny-subset.zarr")
    ds = Dataset(out / "varda-tiny-subset.zarr")
    dates = [str(d) for d in np.asarray(ds.dates).astype("datetime64[s]")]
    shutil.rmtree(out / "grib", ignore_errors=True)
    (out / "grib").mkdir()
    for index, where in GRIB.items():
        write_grib(checkout, index, where, dates, out)


def write_grib(checkout: Path, index: str, where: str, dates: list[str], out: Path) -> None:
    """The messages of `index` selected by `where` at `dates`, cut to every POINT_STRIDE-th point."""
    grib, db = Path("grib") / f"{index}.grib", out / f"grib-index-{index}.db"
    db.unlink(missing_ok=True)
    with closing(sqlite3.connect(checkout / f"ml_test/processed/grib-index-{index}.db")) as src, \
            closing(sqlite3.connect(db)) as dst, open(out / grib, "wb") as f:
        src.row_factory = sqlite3.Row
        for (sql, ) in src.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_stat1'"):
            dst.execute(sql)
        dst.execute("INSERT INTO paths (id, path) VALUES (1, ?)", (str(grib), ))
        rows = src.execute(
            "SELECT g.*, p.path AS _source FROM grib_index g JOIN paths p ON p.id = g._path_id "
            f"WHERE g.valid_datetime IN ({', '.join('?' * len(dates))}) AND ({where}) "
            "ORDER BY g.valid_datetime, g.shortName, g.level", dates).fetchall()
        for i, row in enumerate(rows):
            with open(checkout / row["_source"], "rb") as g:
                g.seek(row["_offset"])
                message = cut(g.read(row["_length"]))
            values = {k: row[k] for k in row.keys() if k != "_source"}
            values |= {"_id": i, "_path_id": 1, "_offset": f.tell(), "_length": len(message)}
            dst.execute(
                f"INSERT INTO grib_index ({', '.join(f'[{k}]' for k in values)}) "
                f"VALUES ({', '.join('?' * len(values))})", list(values.values()))
            f.write(message)
        dst.commit()
    print(f"{index}: {len(rows)} messages, {(out / grib).stat().st_size / 1e6:.2f} MB")


def cut(message: bytes) -> bytes:
    handle = eccodes.codes_new_from_message(message)
    try:
        values = eccodes.codes_get_values(handle)[::POINT_STRIDE]
        eccodes.codes_set(handle, "packingType", "grid_ieee")
        eccodes.codes_set(handle, "numberOfDataPoints", len(values))
        eccodes.codes_set_values(handle, values)
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


def write_zarr(source: Path, target: Path) -> None:
    ds = Dataset(source)
    attrs = dict(ds.attrs)
    columns = [attrs["variables"].index(v) for v in VARIABLES]
    points = np.arange(0, ds.data.shape[-1], POINT_STRIDE)
    data = np.asarray(ds.data[DATES])[:, columns][..., points].astype("f4")
    dates = np.asarray(ds.dates[DATES]).astype("datetime64[s]")

    if target.exists():
        shutil.rmtree(target)
    root = zarr.open_group(str(target), mode="w", zarr_format=2)
    root.create_array("data", data=data, chunks=data.shape)
    root.create_array("dates", data=dates)
    root.create_array("latitudes", data=np.asarray(ds.latitudes)[points])
    root.create_array("longitudes", data=np.asarray(ds.longitudes)[points])
    window = data[:WINDOW].astype("f8")
    for name, values in (("mean", window.mean(axis=(0, 2, 3))), ("stdev", window.std(axis=(0, 2, 3))),
                         ("minimum", window.min(axis=(0, 2, 3))), ("maximum", window.max(axis=(0, 2, 3)))):
        root.create_array(name, data=values)

    for key in DROP_ATTRS:
        attrs.pop(key, None)
    attrs.update({
        "variables":
        VARIABLES,
        "variables_metadata": {
            v: attrs["variables_metadata"][v]
            for v in VARIABLES
        },
        "constant_fields": [v for v in attrs["constant_fields"] if v in VARIABLES],
        "shape":
        list(data.shape),
        "chunks":
        list(data.shape),
        "field_shape": [len(points)],
        "start_date":
        str(dates[0]),
        "end_date":
        str(dates[-1]),
        "last_date":
        str(dates[-1]).replace("T", " "),
        "statistics_start_date":
        str(dates[0]),
        "statistics_end_date":
        str(dates[WINDOW - 1]),
        "description":
        f"Subset of Varda-tiny ({source.name}) for the hive's tests: "
        f"{len(VARIABLES)} variables, {len(dates)} dates, every {POINT_STRIDE}th grid point. "
        "Written by tests/data/make_varda_tiny_subset.py.",
    })
    root.attrs.update(attrs)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).parent)
