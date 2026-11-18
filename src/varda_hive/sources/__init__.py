# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reusable spot-bee source plug-ins, named in a spec as `varda_hive.sources.<module>:<Class>`.

They are not imported by the honeycomb or any bee, so their optional dependencies (e.g. eccodes,
the `grib` extra) are only needed when a spec uses them.
"""
