from __future__ import annotations

import argparse
import json

from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed


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
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    from voln_uav.evaluation.airsim_loop import AirSimClosedLoopEvaluator

    evaluator = AirSimClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
