# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reusable spot-bee source plug-ins, named in a spec as `varda_hive.sources.<module>:<Class>`.

- `grib_index:GribIndexSource`: GRIB files through anemoi grib-index DBs (`grib` extra)
- `fdb:FdbSource`: GRIB fields from an FDB (`fdb` extra)

They are not imported by the honeycomb or any bee, so their optional dependencies (eccodes, pyfdb)
are only needed when a spec uses them.
"""
