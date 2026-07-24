from __future__ import annotations

import json
from pathlib import Path

import pytest

from voln_uav.cli.launch_airsim import build_launch_command
from voln_uav.common.config import load_config


def _config(tmp_path: Path, executable: Path) -> dict:
    mapping = tmp_path / "scenes.json"
    mapping.write_text(
        json.dumps({"BrushifyUrban": executable.name}),
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "port: 41451",
                "env:",
                "  root_path: simulator_environments",
                "  scene_mapping: scenes.json",
                "  executable_args: [--port, '{port}']",
            ]
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


def test_build_launch_command_uses_scene_mapping_and_port(tmp_path: Path) -> None:
    env_root = tmp_path / "downloaded_envs"
    env_root.mkdir()
    executable = env_root / "VolnEnv.exe"
    executable.write_bytes(b"placeholder")
    config = _config(tmp_path, executable)

    command = build_launch_command(config, "BrushifyUrban", 41452, env_root)

    assert command == [str(executable.resolve()), "--port", "41452"]


def test_build_launch_command_reports_missing_scene(tmp_path: Path) -> None:
    env_root = tmp_path / "downloaded_envs"
    env_root.mkdir()
    executable = env_root / "VolnEnv.exe"
    executable.write_bytes(b"placeholder")
    config = _config(tmp_path, executable)

    with pytest.raises(KeyError, match="Available scenes: BrushifyUrban"):
        build_launch_command(config, "UnknownScene", 41451, env_root)
