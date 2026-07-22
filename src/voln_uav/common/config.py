from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    parents = cfg.pop("extends", None)
    if parents is None:
        return cfg
    parent_items = [parents] if isinstance(parents, (str, Path)) else list(parents)
    merged: dict[str, Any] = {}
    for parent in parent_items:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, _load_config(parent_path, (*stack, path)))
    return _deep_merge(merged, cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = _load_config(path, ())
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(path_like: str | Path, base_dir: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(base_dir) / path
