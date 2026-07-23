from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


METHOD_CONFIGS = {
    "random": "dataset_release",
    "voln_mllm": "dataset_release",
    "seq2seq_vg": "seq2seq_dataset_release",
    "cma": "cma_dataset_release",
    "lag": "lag_dataset_release",
}
PAPER_SPLITS = ("validation_seen", "test_unseen")


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate manuscript methods on Validation-Seen and Test-Unseen without changing dataset files."
    )
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_CONFIGS))
    parser.add_argument("--splits", nargs="+", choices=PAPER_SPLITS, default=list(PAPER_SPLITS))
    parser.add_argument("--backend", choices=("offline", "airsim"), default="offline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="D:/VoLN_dataset/VoLN-UAV-runs")
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Optional scene IDs. Absent partial-release scenes are skipped and recorded, not scored as failures.",
    )
    parser.add_argument(
        "--strict-scenes",
        action="store_true",
        help="Fail if any requested scene is absent.",
    )
    parser.add_argument("--preflight", action="store_true", help="AirSim only: check scene readiness without loading a model.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.preflight and args.backend != "airsim":
        parser.error("--preflight is only valid with --backend airsim")
    methods = args.methods or (["random", "voln_mllm", "seq2seq_vg", "cma", "lag"] if args.backend == "airsim" else ["voln_mllm", "seq2seq_vg", "cma", "lag"])
    if args.backend == "offline" and "random" in methods:
        parser.error("The Random baseline requires --backend airsim so it shares the paper closed-loop protocol.")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    module = f"voln_uav.cli.eval_{args.backend}"

    for method in methods:
        config_name = METHOD_CONFIGS[method]
        config = repo_root / "configs" / f"eval_{args.backend}_{config_name}.yaml"
        for split in args.splits:
            work_dir = output_root / f"eval_{args.backend}_{method}_{split}_paper"
            command = [
                args.python,
                "-m",
                module,
                "--config",
                str(config),
                "--device",
                args.device,
                "--split",
                split,
                "--work-dir",
                str(work_dir),
            ]
            if args.preflight:
                command.append("--preflight")
            if args.scenes:
                command.extend(["--scenes", *args.scenes])
            if args.strict_scenes:
                command.append("--strict-scenes")
            if method == "random":
                command.extend(["--controller", "random"])
            _run(command, env, args.dry_run)


if __name__ == "__main__":
    main()
