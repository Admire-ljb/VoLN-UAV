from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_STAGES = ["build", "train-adapter", "train-planner"]
STAGE_TO_MODULE = {
    "build": "voln_uav.cli.build_benchmark",
    "train-adapter": "voln_uav.cli.train_adapter",
    "train-planner": "voln_uav.cli.train_planner",
    "offline-eval": "voln_uav.cli.eval_offline",
    "airsim-eval": "voln_uav.cli.eval_airsim",
}
STAGE_TO_CONFIG = {
    "build": "configs/benchmark_dataset_release.yaml",
    "train-adapter": "configs/train_adapter_dataset_release.yaml",
    "train-planner": "configs/train_planner_dataset_release.yaml",
    "offline-eval": "configs/eval_offline_dataset_release.yaml",
    "airsim-eval": "configs/eval_airsim_dataset_release.yaml",
}


def run_command(args: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full VoLN-UAV dataset-release training pipeline.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGE_TO_MODULE),
        default=DEFAULT_STAGES,
        help="Pipeline stages to run in order.",
    )
    parser.add_argument("--device", default=None, help="Device for training/evaluation stages, e.g. cuda or cpu.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for stage in args.stages:
        cmd = [
            args.python,
            "-m",
            STAGE_TO_MODULE[stage],
            "--config",
            str(repo_root / STAGE_TO_CONFIG[stage]),
        ]
        if args.device and stage in {"train-adapter", "train-planner", "offline-eval", "airsim-eval"}:
            cmd.extend(["--device", args.device])
        run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
