from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


DEFAULT_ENV_EXEC_PATHS = {
    "urban_001": "closeloop_envs/ModularEuropean.sh",
    "urban_002": "extra_envs/BrushifyUrban.sh",
    "forest_001": "closeloop_envs/ModularPark.sh",
    "forest_002": "extra_envs/BrushifyCountryRoads.sh",
    "tunnel_001": "extra_envs/Tunnel.sh",
    "tunnel_002": "extra_envs/IndustrialCorridor.sh",
}



def resolve_exec(root_path: Path, rel_exec: str) -> Path:
    path = root_path / rel_exec
    return path.resolve()



def launch_env(exec_path: Path, port: int, dry_run: bool) -> subprocess.Popen[str] | None:
    cmd = [str(exec_path), "--port", str(port)]
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return None
    return subprocess.Popen(cmd)



def main() -> None:
    parser = argparse.ArgumentParser(description="TravelUAV-style environment launcher for VoLN-UAV")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--scene", type=str, default="urban_001")
    parser.add_argument("--mapping_json", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not port_available(args.port):
        raise RuntimeError(f"Port {args.port} is not available.")

    env_exec_path_dict = dict(DEFAULT_ENV_EXEC_PATHS)
    if args.mapping_json:
        with open(args.mapping_json, "r", encoding="utf-8") as f:
            env_exec_path_dict.update(json.load(f))

    if args.scene not in env_exec_path_dict:
        raise KeyError(f"Unknown scene '{args.scene}'. Available: {sorted(env_exec_path_dict)}")

    root_path = Path(args.root_path)
    exec_path = resolve_exec(root_path, env_exec_path_dict[args.scene])
    if not exec_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Executable not found: {exec_path}")

    proc = launch_env(exec_path, args.port, args.dry_run)
    if args.dry_run:
        return
    print(f"Launched {args.scene} on port {args.port} using {exec_path}")
    try:
        while proc is not None and proc.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
