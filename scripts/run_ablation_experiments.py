from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = ("no_align", "no_lora", "clip_input")
TRAIN_CONFIGS = {
    "no_align": [
        ("voln_uav.cli.train_adapter", "configs/train_adapter_no_align_dataset_release.yaml"),
        ("voln_uav.cli.train_planner", "configs/train_planner_no_align_dataset_release.yaml"),
    ],
    "no_lora": [("voln_uav.cli.train_planner", "configs/train_planner_no_lora_dataset_release.yaml")],
    "clip_input": [("voln_uav.cli.train_planner", "configs/train_planner_clip_input_dataset_release.yaml")],
}
EVAL_CONFIGS = {
    "offline": {
        variant: f"configs/eval_offline_{variant}_dataset_release.yaml"
        for variant in VARIANTS
    },
    "airsim": {
        variant: f"configs/eval_airsim_{variant}_dataset_release.yaml"
        for variant in VARIANTS
    },
}


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the three benchmark ablations.")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--stages", nargs="+", choices=["train", "offline", "airsim"], default=["train", "airsim"])
    parser.add_argument(
        "--allow-offline-diagnostic",
        action="store_true",
        help="Allow route-replay diagnostics in addition to AirSim benchmark evaluation.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "offline" in args.stages and not args.allow_offline_diagnostic:
        parser.error("offline route replay requires --allow-offline-diagnostic")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    main_adapter = Path("runs/adapter_voln_mllm/adapter_best.pt")
    if "train" in args.stages and "no_lora" in args.variants and not main_adapter.exists():
        _run(
            [
                args.python,
                "-m",
                "voln_uav.cli.train_adapter",
                "--config",
                str(repo_root / "configs/train_adapter_dataset_release.yaml"),
                "--device",
                args.device,
            ],
            env,
            args.dry_run,
        )

    for variant in args.variants:
        if "train" in args.stages:
            for module, relative_config in TRAIN_CONFIGS[variant]:
                _run(
                    [args.python, "-m", module, "--config", str(repo_root / relative_config), "--device", args.device],
                    env,
                    args.dry_run,
                )
        for stage in ("offline", "airsim"):
            if stage not in args.stages:
                continue
            module = "voln_uav.cli.eval_offline" if stage == "offline" else "voln_uav.cli.eval_airsim"
            _run(
                [args.python, "-m", module, "--config", str(repo_root / EVAL_CONFIGS[stage][variant]), "--device", args.device],
                env,
                args.dry_run,
            )


if __name__ == "__main__":
    main()
