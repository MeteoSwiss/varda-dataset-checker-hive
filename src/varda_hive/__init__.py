# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""varda-dataset-checker-hive: slim bees checking anemoi datasets.

- meta-bee:  metadata and structure
- stats-bee: statistics, ranges, NaNs, constant fields
- spot-bee:  sampled values vs the original source
- hive:      runs all bees and merges their findings
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("varda-dataset-checker-hive")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
