# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Where spot-bee gets the "true" values from.

A source is any object with `fetch(variable, date) -> np.ndarray | None`
returning the field in dataset grid order, shape (grid,) or (ensemble, grid),
or None if the source has no data for it. `fetch` raises `NotComparable` for a
variable the source cannot reproduce one-to-one (a derived or accumulated
field, a computed forcing). An optional `bind(dataset)` is called once before
the first fetch, so a source can read the dataset's variables_metadata.

The spec names the plug-in by import path, see templates/source_plugin_example.py,
varda_hive.sources.grib_index and varda_hive.sources.fdb. A plug-in whose __init__ takes `base_dir`
gets the spec's directory, to resolve relative paths in its args.
"""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Protocol, runtime_checkable

import numpy as np

from .spec import Spec


class NotComparable(Exception):
    """Raised by `Source.fetch` for a variable that cannot be compared with the source; the message
    is the reason, e.g. "accumulated over 6 h"."""


@runtime_checkable
class Source(Protocol):

    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None:
        ...


def load_source(spec: Spec) -> Source | None:
    if spec.source is None:
        return None
    target, _, class_name = spec.source.plugin.partition(":")
    if not class_name:
        raise ValueError(f"source plugin must be 'module:Class' or 'file.py:Class', got {spec.source.plugin!r}")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_absolute():
            path = spec.base_dir / path
        # A private name, so the plug-in cannot shadow an installed module with the same file name.
        name = f"_varda_hive_source_plugin_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load source plugin from {path}")
        module = importlib.util.module_from_spec(module_spec)
        # Registered before running it: dataclasses (and pickle) look the module up in sys.modules.
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    else:
        module = importlib.import_module(target)
    cls = getattr(module, class_name)
    kwargs = dict(spec.source.args)
    if "base_dir" in inspect.signature(cls).parameters:
        kwargs.setdefault("base_dir", spec.base_dir)
    source = cls(**kwargs)
    if not isinstance(source, Source):
        raise TypeError(f"{spec.source.plugin} has no fetch(variable, date) method")
    return source
