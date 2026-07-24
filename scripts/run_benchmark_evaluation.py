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
BENCHMARK_SPLITS = ("validation_seen", "test_unseen")


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate released methods on Validation-Seen and Test-Unseen without changing dataset files."
    )
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_CONFIGS))
    parser.add_argument("--splits", nargs="+", choices=BENCHMARK_SPLITS, default=list(BENCHMARK_SPLITS))
    parser.add_argument("--backend", choices=("offline", "airsim"), default="airsim")
    parser.add_argument(
        "--allow-offline-diagnostic",
        action="store_true",
        help="Allow the route-replay diagnostic backend; it is not used for closed-loop benchmark results.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Optional scene IDs for a diagnostic subset.",
    )
    parser.add_argument(
        "--allow-scene-subset-diagnostic",
        action="store_true",
        help="Allow selected-scene diagnostics; these runs are not full-benchmark runs.",
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
    if args.backend == "offline" and not args.allow_offline_diagnostic:
        parser.error("offline route replay requires --allow-offline-diagnostic")
    if args.scenes and not args.allow_scene_subset_diagnostic:
        parser.error("--scenes requires --allow-scene-subset-diagnostic")
    methods = args.methods or (["random", "voln_mllm", "seq2seq_vg", "cma", "lag"] if args.backend == "airsim" else ["voln_mllm", "seq2seq_vg", "cma", "lag"])
    if args.backend == "offline" and "random" in methods:
        parser.error("The Random baseline requires --backend airsim so it shares the benchmark closed-loop protocol.")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    module = f"voln_uav.cli.eval_{args.backend}"

    for method in methods:
        config_name = METHOD_CONFIGS[method]
        config = repo_root / "configs" / f"eval_{args.backend}_{config_name}.yaml"
        for split in args.splits:
            run_kind = "diagnostic" if args.scenes or args.backend == "offline" else "benchmark"
            work_dir = output_root / f"eval_{args.backend}_{method}_{split}_{run_kind}"
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
                command.append("--allow-partial-diagnostic")
            if args.strict_scenes:
                command.append("--strict-scenes")
            if args.backend == "airsim":
                command.extend(["--controller", "random" if method == "random" else "policy"])
            _run(command, env, args.dry_run)


if __name__ == "__main__":
    main()
