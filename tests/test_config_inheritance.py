from __future__ import annotations

from pathlib import Path

import pytest

from voln_uav.common.config import load_config


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_config_extends_deep_merges_nested_mappings(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "seed: 7\nmodel:\n  horizon: 8\n  lora_rank: 16\n")
    _write(tmp_path / "child.yaml", "extends: base.yaml\nmodel:\n  lora_enabled: false\n")

    config = load_config(tmp_path / "child.yaml")

    assert config["seed"] == 7
    assert config["model"] == {"horizon": 8, "lora_rank": 16, "lora_enabled": False}
    assert config["_config_path"] == str((tmp_path / "child.yaml").resolve())


def test_config_extends_rejects_cycles(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")

    with pytest.raises(ValueError, match="Cyclic config inheritance"):
        load_config(tmp_path / "a.yaml")
