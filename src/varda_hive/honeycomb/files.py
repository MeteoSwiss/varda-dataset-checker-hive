# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reading and writing the hive's own files (stats files, tolerance files, meta files): YAML for a
`.yaml`/`.yml` path, JSON otherwise, each with a `format` key checked on load."""
import json
from pathlib import Path

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
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


def read_file(path: str | Path, fmt: str, what: str, command: str = "") -> dict:
    """Load a YAML/JSON file written by `save_file` and check its `format`. `command` regenerates it."""
    text = Path(path).read_text(encoding="utf-8")
    content = yaml.load(text, Loader=_Loader) if Path(path).suffix in YAML_SUFFIXES else json.loads(text)
    found = content.get("format") if isinstance(content, dict) else None
    if found != fmt:
        family = fmt.rsplit("/", 1)[0] + "/"
        how = f" with `{command}`" if command else ""
        hint = f" (an older {what}: regenerate it{how})" if str(found).startswith(family) else ""
        raise ValueError(f"{path}: not a {what} of format {fmt!r}, found {found!r}{hint}")
    return content
