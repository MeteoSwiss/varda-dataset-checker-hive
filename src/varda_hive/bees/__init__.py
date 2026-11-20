# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The bees. Each bee is independent: it imports only the honeycomb and its own modules, never another
bee, and is loaded only when it runs (see `honeycomb.registry.checks_for` and `cli._bee_main`).
tests/test_independence.py enforces this."""
