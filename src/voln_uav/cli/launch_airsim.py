from __future__ import annotations

import argparse
import socket
import subprocess
import time
from pathlib import Path

from voln_uav.common.config import load_config
from voln_uav.evaluation.airsim_loop import load_scene_mapping, resolve_config_path


def _port_is_open(host: str, port: int, timeout_sec: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_sec)
        return sock.connect_ex((host, port)) == 0


def build_launch_command(
    config: dict,
    scene: str,
    port: int,
    env_root: str | Path | None = None,
) -> list[str]:
    env_cfg = config.get("env", {})
    root = Path(env_root).expanduser().resolve() if env_root else resolve_config_path(
        env_cfg.get("root_path", "simulator_environments"),
        config,
    )
    mapping = load_scene_mapping(env_cfg.get("scene_mapping"), config)
    mapping.update(env_cfg.get("scene_mapping_inline", {}))
    if scene not in mapping:
        available = ", ".join(sorted(mapping)) or "(none)"
        raise KeyError(f"Unknown scene {scene!r}. Available scenes: {available}")
    executable = (root / mapping[scene]).resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"Simulator executable not found: {executable}\n"
            "Download the simulator environments and pass --env-root or set env.root_path in the config."
        )
    command = [str(executable)]
    command.extend(
        str(item).format(port=port, scene=scene)
        for item in env_cfg.get("executable_args", ["--port", "{port}"])
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one released VoLN AirSim scene.")
    parser.add_argument("--config", default="configs/eval_airsim_dataset_release.yaml")
    parser.add_argument("--scene", default="BrushifyUrban")
    parser.add_argument(
        "--env-root",
        help="Directory containing the downloaded simulator environments.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--wait-sec",
        type=float,
        default=60.0,
        help="Maximum time to wait for the AirSim RPC port (default: 60).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    port = int(args.port if args.port is not None else config.get("port", 41451))
    if _port_is_open(args.host, port):
        print(f"AirSim is already available at {args.host}:{port}.")
        return

    command = build_launch_command(config, args.scene, port, args.env_root)
    process = subprocess.Popen(command)
    print(f"Started {args.scene} (PID {process.pid}).")
    print(f"Executable: {command[0]}")

    deadline = time.monotonic() + max(args.wait_sec, 0.0)
    while time.monotonic() < deadline:
        if _port_is_open(args.host, port):
            print(f"AirSim is ready at {args.host}:{port}.")
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"Simulator process exited with code {process.returncode} before AirSim became ready."
            )
        time.sleep(1.0)
    raise TimeoutError(
        f"Simulator started, but AirSim did not become ready at {args.host}:{port} "
        f"within {args.wait_sec:g} seconds."
    )


if __name__ == "__main__":
    main()
