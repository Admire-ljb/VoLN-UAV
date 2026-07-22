from __future__ import annotations

import argparse
import json

import torch

from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed
from voln_uav.evaluation.closed_loop import ClosedLoopEvaluator


PAPER_SPLITS = {
    "validation_seen": "val.jsonl",
    "test_unseen": "test.jsonl",
}



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split",
        choices=sorted(PAPER_SPLITS),
        help="Paper evaluation split. validation_seen maps to val.jsonl; test_unseen maps to test.jsonl.",
    )
    split_group.add_argument("--episodes-file", help="Custom episode JSONL path relative to benchmark_root.")
    parser.add_argument("--work-dir", help="Override the output directory for metrics and trajectories.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.split is not None:
        cfg["episodes_file"] = PAPER_SPLITS[args.split]
    elif args.episodes_file is not None:
        cfg["episodes_file"] = args.episodes_file
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    set_seed(int(cfg["seed"]))
    evaluator = ClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
