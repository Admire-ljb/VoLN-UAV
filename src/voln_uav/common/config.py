from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(path_like: str | Path, base_dir: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(base_dir) / path
