# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reading and writing the hive's own files (stats files, tolerance files, meta files): YAML for a
`.yaml`/`.yml` path, JSON otherwise, each with a `format` key checked on load.

A ref file (`ref.yaml`, written by `--out`) holds the dump of each bee, a section per bee:

    format: varda-hive-ref/1
    meta: {format: varda-hive-meta/1, ...}     # meta-bee's dump: metadata and structure
    stats: {format: varda-hive-stats/5, ...}   # stats-bee's dump: statistics on the sample

`read_file(..., section=...)` reads either a bee's own file or its section of a ref file, so a bee
never needs to know about the others."""
import json
from pathlib import Path

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
REF_FORMAT = "varda-hive-ref/1"
_Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def save_file(content: dict, path: str | Path | None) -> None:
    """Write YAML to a `.yaml`/`.yml` path, JSON to any other path, JSON to stdout if it is None or "-".

    YAML: dicts in block style, every list on one line, so a diff shows what changed."""
    if path not in (None, "-") and Path(path).suffix in YAML_SUFFIXES:
        text = yaml.dump(content, Dumper=_Dumper, sort_keys=False, default_flow_style=None, width=120)
    else:
        text = json.dumps(content, indent=1, allow_nan=False) + "\n"
    if path in (None, "-"):
        print(text, end="")
    else:
        Path(path).write_text(text, encoding="utf-8")


def _load(path: str | Path) -> object:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.load(text, Loader=_Loader) if Path(path).suffix in YAML_SUFFIXES else json.loads(text)


def file_format(path: str | Path) -> str | None:
    """The `format` of a hive file, None for a directory (a zarr) or a file without one."""
    if not Path(path).is_file():
        return None
    try:
        content = _load(path)
    except (ValueError, yaml.YAMLError, UnicodeDecodeError):
        return None
    found = content.get("format") if isinstance(content, dict) else None
    return found if isinstance(found, str) else None


def provides(path: str | Path, section: str, fmt: str) -> bool:
    """False if `path` is a hive file without what a bee needs: another bee's file (e.g. a stats file
    for meta-bee), or a ref file without the bee's `section`. True otherwise, including for a zarr
    or a broken file, which the bee's own loader then reports."""
    found = file_format(path)
    if found == REF_FORMAT:
        content = _load(path)
        return isinstance(content, dict) and isinstance(content.get(section), dict)
    family = fmt.rsplit("/", 1)[0] + "/"
    return not (found is not None and found.startswith("varda-hive-") and not found.startswith(family))


def save_ref(sections: dict[str, dict], path: str | Path | None) -> None:
    """Write a ref file with one section per bee, e.g. {"meta": ..., "stats": ...}."""
    save_file({"format": REF_FORMAT, **sections}, path)


def read_file(path: str | Path, fmt: str, what: str, command: str = "", section: str | None = None) -> dict:
    """Load a YAML/JSON file written by `save_file` and check its `format`. `command` regenerates it.

    With `section`, the file may also be a ref file: its `section` is returned."""
    content = _load(path)
    found = content.get("format") if isinstance(content, dict) else None
    if section is not None and found == REF_FORMAT and isinstance(content, dict):
        if not isinstance(content.get(section), dict):
            raise ValueError(
                f"{path}: ref file without a `{section}` section (replace it with the `ref.yaml` of `hive ... --out`)")
        content = content[section]
        found = content.get("format")
    if found != fmt:
        family = fmt.rsplit("/", 1)[0] + "/"
        how = f" with the `ref.yaml` of `{command}`" if command else ""
        hint = f" (an older {what}: replace it{how})" if str(found).startswith(family) else ""
        raise ValueError(f"{path}: not a {what} of format {fmt!r}, found {found!r}{hint}")
    assert isinstance(content, dict)
    return content
