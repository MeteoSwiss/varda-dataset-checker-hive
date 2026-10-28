"""Where spot-bee gets the "true" values from.

A source is any object with `fetch(variable, date) -> np.ndarray | None`
returning the flattened field (grid points in dataset order), or None if the
source has no data for it. Sources are dataset-specific and live in the
dataset repo; the spec names them by import path, see
templates/source_plugin_example.py.
"""
import importlib
import importlib.util
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .spec import Spec


@runtime_checkable
class Source(Protocol):
    def fetch(self, variable: str, date: np.datetime64) -> np.ndarray | None: ...


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
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load source plugin from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    source = getattr(module, class_name)(**spec.source.args)
    if not isinstance(source, Source):
        raise TypeError(f"{spec.source.plugin} has no fetch(variable, date) method")
    return source
