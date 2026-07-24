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
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Optional scene IDs. Missing partial-release scenes are skipped and recorded in scene_coverage.json.",
    )
    parser.add_argument(
        "--strict-scenes",
        action="store_true",
        help="Fail instead of skipping when any requested scene is absent.",
    )
    parser.add_argument(
        "--allow-partial-diagnostic",
        action="store_true",
        help="Run route-replay diagnostics on an incomplete benchmark release.",
    )
    parser.add_argument("--work-dir", help="Override the output directory for metrics and trajectories.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.split is not None:
        cfg["episodes_file"] = PAPER_SPLITS[args.split]
    elif args.episodes_file is not None:
        cfg["episodes_file"] = args.episodes_file
    if args.scenes is not None:
        cfg["scene_allowlist"] = args.scenes
    if args.strict_scenes:
        cfg["strict_scenes"] = True
    if args.allow_partial_diagnostic:
        cfg["strict_paper_protocol"] = False
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    set_seed(int(cfg["seed"]))
    evaluator = ClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
