from __future__ import annotations

import argparse
import json
from pathlib import Path

from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed
from voln_uav.common.io import read_jsonl


PAPER_SPLITS = {
    "validation_seen": "val.jsonl",
    "test_unseen": "test.jsonl",
}


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
    parser.add_argument(
        "--ip",
        help="AirSim RPC host override, for example a Windows simulator host reached from Ubuntu.",
    )
    parser.add_argument("--port", type=int, help="AirSim RPC port override.")
    parser.add_argument("--controller", choices=["policy", "random", "reference"], help="Evaluation controller override.")
    parser.add_argument(
        "--beacon-mode",
        "--beacon-type",
        dest="beacon_mode",
        choices=["random", "direction", "text"],
        help="Active beacon asset style. If omitted, the shared config defaults to direction icons.",
    )
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
        help="Run a clearly labelled diagnostic on an incomplete benchmark release.",
    )
    parser.add_argument("--preflight", action="store_true", help="Check AirSim dependencies and scene access without loading the model.")
    parser.add_argument("--trials", type=int, help="Number of episodes to run after episode-index/stride filtering.")
    parser.add_argument("--episode-index", type=int, help="First episode index after split/scene/difficulty filtering.")
    parser.add_argument("--episode-stride", type=int, help="Stride between evaluated episodes after filtering.")
    parser.add_argument("--reference-stride", type=int, help="Stride used when following/logging the reference trajectory.")
    parser.add_argument(
        "--control-mode",
        choices=["move_to_position", "teleport"],
        help="How to execute each predicted waypoint in AirSim.",
    )
    parser.add_argument("--fast-reset", action="store_true", help="Reset by pose only instead of taking off before every episode.")
    parser.add_argument("--settle-sec", type=float, help="Seconds to wait after each action/reset.")
    parser.add_argument("--max-teleport-step-m", type=float, help="Maximum setVehiclePose displacement per teleport action.")
    parser.add_argument("--max-teleport-vertical-step-m", type=float, help="Maximum vertical displacement per teleport action.")
    parser.add_argument("--reference-bootstrap-steps", type=int, help="Teleport through this many reference points before policy actions.")
    parser.add_argument("--disable-teleport-keep-initial-height", action="store_true", help="Compatibility flag; teleport no longer pins altitude to the episode start height.")
    parser.add_argument("--disable-teleport-hover-after-setpose", action="store_true", help="Do not call hoverAsync after teleport setVehiclePose.")
    parser.add_argument("--disable-teleport-pause-after-setpose", action="store_true", help="Compatibility flag; teleport no longer pauses physics after setVehiclePose.")
    parser.add_argument("--disable-teleport-zero-velocity", action="store_true", help="Do not zero kinematics after teleport setVehiclePose.")
    parser.add_argument("--work-dir", help="Directory for metrics, trajectories, and beacon placement logs.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.ip is not None:
        cfg["ip"] = args.ip
    if args.port is not None:
        cfg["port"] = args.port
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
    if args.controller is not None:
        cfg["controller"] = args.controller
    if args.beacon_mode is not None:
        cfg.setdefault("beacon_placement", {})["render_mode"] = args.beacon_mode
    if args.trials is not None:
        cfg["trials"] = args.trials
    if args.episode_index is not None:
        cfg["episode_index"] = args.episode_index
    if args.episode_stride is not None:
        cfg["episode_stride"] = args.episode_stride
    if args.reference_stride is not None:
        cfg["reference_stride"] = args.reference_stride
    if args.control_mode is not None:
        cfg["control_mode"] = args.control_mode
    if args.fast_reset:
        cfg["fast_reset"] = True
    if args.settle_sec is not None:
        cfg["settle_sec"] = args.settle_sec
    if args.max_teleport_step_m is not None:
        cfg["max_teleport_step_m"] = args.max_teleport_step_m
    if args.max_teleport_vertical_step_m is not None:
        cfg["max_teleport_vertical_step_m"] = args.max_teleport_vertical_step_m
    if args.reference_bootstrap_steps is not None:
        cfg["reference_bootstrap_steps"] = args.reference_bootstrap_steps
    if args.disable_teleport_keep_initial_height:
        cfg["teleport_keep_initial_height"] = False
    if args.disable_teleport_hover_after_setpose:
        cfg["teleport_hover_after_setpose"] = False
    if args.disable_teleport_pause_after_setpose:
        cfg["teleport_pause_after_setpose"] = False
    if args.disable_teleport_zero_velocity:
        cfg["teleport_zero_velocity"] = False
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    from voln_uav.evaluation.airsim_loop import AirSimClosedLoopEvaluator, check_airsim_readiness

    if args.preflight:
        from voln_uav.evaluation.airsim_loop import filter_airsim_episodes
        from voln_uav.evaluation.paper_protocol import (
            require_paper_protocol_ready,
            select_available_episodes,
        )

        require_paper_protocol_ready(cfg["benchmark_root"], cfg)
        raw_episodes = read_jsonl(Path(cfg["benchmark_root"]) / cfg["episodes_file"])
        _available, coverage = select_available_episodes(raw_episodes, cfg)
        episodes = filter_airsim_episodes(cfg, raw_episodes)
        coverage["selected_episodes_after_filters"] = len(episodes)
        issues = check_airsim_readiness(cfg, episodes) if episodes else []
        status = "ready" if not issues and episodes else "skipped_no_available_episodes"
        if issues:
            status = "not_ready"
        print(
            json.dumps(
                {"ok": not issues, "status": status, "issues": issues, "scene_coverage": coverage},
                indent=2,
            )
        )
        raise SystemExit(1 if issues else 0)

    set_seed(int(cfg["seed"]))
    evaluator = AirSimClosedLoopEvaluator(cfg, device=args.device)
    summary = evaluator.evaluate()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
