# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The bees. Importing this package registers all their checks."""
from .meta import checks as _meta  # noqa: F401
from .spot import checks as _spot  # noqa: F401
from .stats import checks as _stats  # noqa: F401
