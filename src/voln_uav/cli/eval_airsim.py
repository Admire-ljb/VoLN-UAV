from __future__ import annotations

import argparse
import json
from pathlib import Path

from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed
from voln_uav.common.io import read_jsonl


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoLN-UAV closed-loop evaluation in an AirSim environment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--preflight", action="store_true", help="Check AirSim dependencies and scene access without loading the model.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from voln_uav.evaluation.airsim_loop import AirSimClosedLoopEvaluator, check_airsim_readiness

    if args.preflight:
        episodes = read_jsonl(Path(cfg["benchmark_root"]) / cfg["episodes_file"])
        issues = check_airsim_readiness(cfg, episodes)
        print(json.dumps({"ok": not issues, "issues": issues}, indent=2))
        raise SystemExit(1 if issues else 0)

    set_seed(int(cfg["seed"]))
    evaluator = AirSimClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
