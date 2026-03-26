from __future__ import annotations

import argparse
import json

import torch

from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed
from voln_uav.evaluation.closed_loop import ClosedLoopEvaluator



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    evaluator = ClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
