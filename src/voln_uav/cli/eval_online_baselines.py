from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from voln_uav.common.config import load_config
from voln_uav.common.geometry import l2, path_length
from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.evaluation.airsim_loop import check_airsim_readiness
from voln_uav.evaluation.metrics import (
    aggregate_by_difficulty,
    aggregate_metrics,
    reference_travel_time,
    summarize_episode,
    validated_shortest_path_length,
)
from voln_uav.evaluation.termination import StationaryDetector
from voln_uav.simulators.airsim_env import AirSimRouteEnv


METRIC_KEYS = ("NE", "SR", "OSR", "nDTW", "SPL")


def _position(env: AirSimRouteEnv) -> list[float]:
    state = env.client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return [float(pos.x_val), float(pos.y_val), float(pos.z_val)]


def _collision(env: AirSimRouteEnv) -> bool:
    return bool(getattr(env.client.simGetCollisionInfo(), "has_collided", False))


def _reference_targets(episode: dict[str, Any], stride: int, start_index: int = 0) -> list[list[float]]:
    states = list(episode["states"])
    step = max(int(stride), 1)
    start = min(max(int(start_index), 0), max(len(states) - 1, 0))
    targets = [[float(v) for v in state["position"][:3]] for state in states[start::step]]
    final = [float(v) for v in states[-1]["position"][:3]]
    if not targets or targets[-1] != final:
        targets.append(final)
    return targets


def _random_targets(
    episode: dict[str, Any],
    rng: random.Random,
    steps: int,
    max_step_m: float,
    max_vertical_step_m: float,
    start_index: int = 0,
) -> list[list[float]]:
    ref = [[float(v) for v in state["position"][:3]] for state in episode["states"]]
    start = min(max(int(start_index), 0), max(len(ref) - 1, 0))
    pos = list(ref[start])

    targets: list[list[float]] = []
    for _ in range(max(int(steps), 1)):
        heading = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(0.0, max(float(max_step_m), 0.0))
        x = pos[0] + math.cos(heading) * distance
        y = pos[1] + math.sin(heading) * distance
        z = pos[2] + rng.uniform(-max(float(max_vertical_step_m), 0.0), max(float(max_vertical_step_m), 0.0))
        pos = [x, y, z]
        targets.append(pos)
    return targets


def _run_targets(
    env: AirSimRouteEnv,
    episode: dict[str, Any],
    targets: list[list[float]],
    control_mode: str,
    timeout_sec: float,
    path_length_limit_m: float,
    goal: list[float],
    success_radius: float,
    stationary_timeout_sec: float,
    stationary_radius_m: float,
    max_teleport_step_m: float,
    max_teleport_vertical_step_m: float,
    teleport_keep_initial_height: bool,
    paper_protocol: bool,
    random_stop_probability: float = 0.0,
    rng: random.Random | None = None,
    stop_at_end: bool = False,
    max_decisions: int | None = None,
    initial_executed_path: list[list[float]] | None = None,
    initial_path_length_m: float = 0.0,
) -> tuple[list[list[float]], list[float], int, float, str, float, float, bool]:
    executed = list(initial_executed_path) if initial_executed_path else [_position(env)]
    cycle_times: list[float] = []
    collisions = 0
    executed_path_length_m = float(initial_path_length_m)
    started_at = time.perf_counter()
    stationary_detector = StationaryDetector(timeout_sec=stationary_timeout_sec, radius_m=stationary_radius_m)
    stationary_detector.update(_position(env), started_at)
    termination_reason = "completed_targets"
    stopped = False
    decision_limit = len(targets) if max_decisions is None else max(int(max_decisions), 0)
    scheduled_targets = targets[:decision_limit]
    truncated_by_step_limit = len(scheduled_targets) < len(targets)
    for target in scheduled_targets:
        if rng is not None and rng.random() < max(min(float(random_stop_probability), 1.0), 0.0):
            termination_reason = "policy_stop"
            stopped = True
            break
        if not paper_protocol and time.perf_counter() - started_at >= timeout_sec:
            termination_reason = "timeout"
            break
        current = _position(env)
        if not paper_protocol and l2(current, goal) <= success_radius:
            termination_reason = "goal_reached"
            break
        if not paper_protocol and executed_path_length_m >= path_length_limit_m:
            termination_reason = "path_length_limit"
            break
        if not paper_protocol and stationary_detector.update(current, time.perf_counter()):
            termination_reason = "stationary_timeout"
            break

        start = time.perf_counter()
        env.move_to_waypoint(
            current,
            target,
            control_mode=control_mode,
            max_teleport_step_m=max_teleport_step_m,
            max_teleport_vertical_step_m=max_teleport_vertical_step_m,
            teleport_keep_initial_height=teleport_keep_initial_height,
        )
        cycle_times.append(time.perf_counter() - start)
        pos = _position(env)
        executed_path_length_m += l2(current, pos)
        executed.append(pos)
        if _collision(env):
            collisions += 1
        if not paper_protocol and l2(pos, goal) <= success_radius:
            termination_reason = "goal_reached"
            break
        if not paper_protocol and time.perf_counter() - started_at >= timeout_sec:
            termination_reason = "timeout"
            break
        if not paper_protocol and executed_path_length_m >= path_length_limit_m:
            termination_reason = "path_length_limit"
            break
        if not paper_protocol and stationary_detector.update(pos, time.perf_counter()):
            termination_reason = "stationary_timeout"
            break
    if termination_reason == "completed_targets" and truncated_by_step_limit:
        termination_reason = "max_steps"
    episode_elapsed_sec = time.perf_counter() - started_at
    if stop_at_end and termination_reason == "completed_targets":
        stopped = True
        termination_reason = "policy_stop"
    return executed, cycle_times, collisions, episode_elapsed_sec, termination_reason, executed_path_length_m, stationary_detector.duration_sec, stopped


def _summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_metrics([{key: float(item[key]) for key in METRIC_KEYS} for item in details])
    cycle_times = [float(v) for item in details for v in item.get("cycle_times", [])]
    sorted_ct = sorted(cycle_times)
    p95_idx = min(int(0.95 * max(len(sorted_ct) - 1, 0)), max(len(sorted_ct) - 1, 0))
    collisions = sum(int(item.get("collisions", 0)) for item in details)
    return {
        **metrics,
        "episodes": len(details),
        "CT_mean": sum(cycle_times) / max(len(cycle_times), 1),
        "CT_p95": sorted_ct[p95_idx] if sorted_ct else 0.0,
        "EER": 0.0,
        "collisions": collisions,
        "by_difficulty": aggregate_by_difficulty(details),
    }


def _paper_markdown(name: str, summary: dict[str, Any]) -> str:
    headers = ["Baseline", "Episodes", "NE", "SR", "OSR", "nDTW", "SPL", "CT_mean", "CT_p95", "EER", "collisions"]
    row = [
        name,
        str(summary.get("episodes", 0)),
        f"{float(summary.get('NE', 0.0)):.4f}",
        f"{float(summary.get('SR', 0.0)) * 100.0:.2f}",
        f"{float(summary.get('OSR', 0.0)) * 100.0:.2f}",
        f"{float(summary.get('nDTW', 0.0)) * 100.0:.2f}",
        f"{float(summary.get('SPL', 0.0)) * 100.0:.2f}",
        f"{float(summary.get('CT_mean', 0.0)):.4f}",
        f"{float(summary.get('CT_p95', 0.0)):.4f}",
        f"{float(summary.get('EER', 0.0)) * 100.0:.2f}",
        str(int(summary.get("collisions", 0))),
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            "| " + " | ".join(row) + " |",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online AirSim reference/random baselines with physical or teleport control.")
    parser.add_argument("--config", default="configs/eval_airsim_dataset_release.yaml")
    parser.add_argument("--baseline", choices=["reference", "random"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference-stride", type=int, default=1)
    parser.add_argument("--random-steps", type=int)
    parser.add_argument("--control-mode", default="move_to_position", choices=["move_to_position", "teleport"])
    parser.add_argument("--fast-reset", action="store_true", help="Reset each episode with simSetVehiclePose only, skipping takeoff/moveToPositionAsync.")
    parser.add_argument("--settle-sec", type=float, help="Override simulator settle time after each pose/action update.")
    parser.add_argument("--max-teleport-step-m", type=float, help="Maximum setVehiclePose displacement per teleport action.")
    parser.add_argument("--max-teleport-vertical-step-m", type=float, help="Maximum vertical displacement per teleport action.")
    parser.add_argument("--reference-bootstrap-steps", type=int, help="Teleport through this many reference points before policy/baseline actions.")
    parser.add_argument("--disable-teleport-keep-initial-height", action="store_true", help="Compatibility flag; teleport no longer pins altitude to the episode start height.")
    parser.add_argument("--disable-teleport-hover-after-setpose", action="store_true", help="Do not call hoverAsync after teleport setVehiclePose.")
    parser.add_argument("--disable-teleport-pause-after-setpose", action="store_true", help="Compatibility flag; teleport no longer pauses physics after setVehiclePose.")
    parser.add_argument("--disable-teleport-zero-velocity", action="store_true", help="Do not zero kinematics after teleport setVehiclePose.")
    parser.add_argument("--episode-timeout-factor", type=float, help="End an episode after this multiple of reference travel time.")
    parser.add_argument("--episode-path-length-factor", type=float, help="End an episode after this multiple of reference path length.")
    parser.add_argument("--stationary-timeout-sec", type=float, help="End an episode after the vehicle stays within stationary radius for this many seconds; <=0 disables it.")
    parser.add_argument("--stationary-radius-m", type=float, help="Radius used to decide whether the vehicle is stationary.")
    parser.add_argument("--work-dir")
    parser.add_argument("--no-beacons", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = read_jsonl(Path(cfg["benchmark_root"]) / cfg["episodes_file"])
    selected = []
    for trial in range(int(args.trials)):
        idx = int(args.episode_index) + trial * max(int(args.episode_stride), 1)
        if idx >= len(episodes):
            break
        selected.append(episodes[idx])
    issues = check_airsim_readiness(cfg, selected)
    if issues:
        raise RuntimeError("AirSim online baseline is not ready:\n" + "\n".join(f"- {issue}" for issue in issues))

    run_dir = ensure_dir(args.work_dir or (Path(cfg.get("work_dir", "work_dirs/eval_online")) / f"online_{args.baseline}_10"))
    details: list[dict[str, Any]] = []
    env = AirSimRouteEnv(
        ip=str(cfg.get("ip", "127.0.0.1")),
        port=int(cfg.get("port", 41451)),
        camera=str(cfg.get("camera", "0")),
        image_type=str(cfg.get("image_type", "Scene")),
        work_dir=run_dir,
        speed=float(cfg.get("speed", 3.0)),
        move_timeout_sec=float(cfg.get("move_timeout_sec", 15.0)),
        settle_sec=float(args.settle_sec if args.settle_sec is not None else cfg.get("settle_sec", 0.05)),
        takeoff_timeout_sec=float(cfg.get("takeoff_timeout_sec", 10.0)),
        max_teleport_step_m=float(args.max_teleport_step_m if args.max_teleport_step_m is not None else cfg.get("max_teleport_step_m", cfg.get("speed", 3.0))),
        max_teleport_vertical_step_m=float(args.max_teleport_vertical_step_m if args.max_teleport_vertical_step_m is not None else cfg.get("max_teleport_vertical_step_m", 0.5)),
        teleport_keep_initial_height=not bool(args.disable_teleport_keep_initial_height or not cfg.get("teleport_keep_initial_height", False)),
        teleport_hover_after_setpose=not bool(args.disable_teleport_hover_after_setpose or not cfg.get("teleport_hover_after_setpose", True)),
        teleport_pause_after_setpose=not bool(args.disable_teleport_pause_after_setpose or not cfg.get("teleport_pause_after_setpose", False)),
        teleport_zero_velocity=not bool(args.disable_teleport_zero_velocity or not cfg.get("teleport_zero_velocity", True)),
    )
    env.connect(timeout_sec=float(cfg.get("connect_timeout_sec", 60.0)))
    try:
        for trial, episode in enumerate(selected):
            episode_id = str(episode["episode_id"])
            fast_reset = bool(args.fast_reset or cfg.get("fast_reset", False) or str(args.control_mode) == "teleport")
            reset_stabilization = env.reset_to_episode_start(episode, ensure_flying=not fast_reset)
            beacon_cfg = dict(cfg.get("beacon_placement", {}) or {})
            if args.no_beacons:
                beacon_cfg["enabled"] = False
            placements = env.place_beacons_for_episode(episode, beacon_cfg, seed=int(cfg.get("seed", 0)))
            max_teleport_step_m = float(args.max_teleport_step_m if args.max_teleport_step_m is not None else cfg.get("max_teleport_step_m", env.max_teleport_step_m))
            max_teleport_vertical_step_m = float(args.max_teleport_vertical_step_m if args.max_teleport_vertical_step_m is not None else cfg.get("max_teleport_vertical_step_m", env.max_teleport_vertical_step_m))
            teleport_keep_initial_height = not bool(args.disable_teleport_keep_initial_height or not cfg.get("teleport_keep_initial_height", env.teleport_keep_initial_height))
            teleport_hover_after_setpose = not bool(args.disable_teleport_hover_after_setpose or not cfg.get("teleport_hover_after_setpose", env.teleport_hover_after_setpose))
            teleport_pause_after_setpose = not bool(args.disable_teleport_pause_after_setpose or not cfg.get("teleport_pause_after_setpose", env.teleport_pause_after_setpose))
            teleport_zero_velocity = not bool(args.disable_teleport_zero_velocity or not cfg.get("teleport_zero_velocity", env.teleport_zero_velocity))
            reference_bootstrap_steps = max(int(args.reference_bootstrap_steps if args.reference_bootstrap_steps is not None else cfg.get("reference_bootstrap_steps", 3)), 0)
            reference_bootstrap = env.teleport_reference_prefix(
                episode,
                count=reference_bootstrap_steps,
                max_teleport_step_m=max_teleport_step_m,
                max_teleport_vertical_step_m=max_teleport_vertical_step_m,
                teleport_keep_initial_height=teleport_keep_initial_height,
                teleport_hover_after_setpose=teleport_hover_after_setpose,
                teleport_pause_after_setpose=teleport_pause_after_setpose,
                teleport_zero_velocity=teleport_zero_velocity,
            )
            bootstrap_positions = [item["position_after"] for item in reference_bootstrap]
            bootstrap_reference_offset = max((int(item.get("reference_index", -1)) for item in reference_bootstrap), default=-1)
            bootstrap_path_length_m = sum(float(item.get("executed_distance_m", 0.0)) for item in reference_bootstrap)
            ref_path = [state["position"] for state in episode["states"]]
            reference_path_length_m = path_length(ref_path)
            reference_time_sec = reference_travel_time(ref_path, env.speed)
            timeout_factor = float(args.episode_timeout_factor if args.episode_timeout_factor is not None else cfg.get("episode_timeout_factor", 2.0))
            timeout_sec = max(reference_time_sec * timeout_factor, float(cfg.get("minimum_episode_timeout_sec", 1.0)))
            path_length_factor = float(args.episode_path_length_factor if args.episode_path_length_factor is not None else cfg.get("episode_path_length_factor", 2.0))
            path_length_limit_m = max(
                reference_path_length_m * path_length_factor,
                float(cfg.get("minimum_episode_path_length_m", 1.0)),
            )
            goal = [float(v) for v in episode["states"][-1]["position"][:3]]
            success_radius = float(cfg["success_radius"])
            stationary_timeout_sec = float(args.stationary_timeout_sec if args.stationary_timeout_sec is not None else cfg.get("stationary_timeout_sec", 10.0))
            stationary_radius_m = float(args.stationary_radius_m if args.stationary_radius_m is not None else cfg.get("stationary_radius_m", 0.5))
            paper_protocol = str(cfg.get("termination_mode", "paper")).lower() == "paper"
            if args.baseline == "reference":
                targets = _reference_targets(episode, stride=int(args.reference_stride), start_index=bootstrap_reference_offset + 1)
                random_rng = None
            else:
                random_rng = random.Random(int(args.seed) + trial * 1009)
                targets = _random_targets(
                    episode,
                    random_rng,
                    steps=int(args.random_steps if args.random_steps is not None else cfg.get("max_steps", 128)),
                    max_step_m=max_teleport_step_m,
                    max_vertical_step_m=max_teleport_vertical_step_m,
                    start_index=max(bootstrap_reference_offset, 0),
                )
            executed, cycle_times, collisions, episode_elapsed_sec, termination_reason, executed_path_length_m, stationary_duration_sec, stopped = _run_targets(
                env,
                episode,
                targets,
                control_mode=str(args.control_mode),
                timeout_sec=timeout_sec,
                path_length_limit_m=path_length_limit_m,
                goal=goal,
                success_radius=success_radius,
                stationary_timeout_sec=stationary_timeout_sec,
                stationary_radius_m=stationary_radius_m,
                max_teleport_step_m=max_teleport_step_m,
                max_teleport_vertical_step_m=max_teleport_vertical_step_m,
                teleport_keep_initial_height=teleport_keep_initial_height,
                paper_protocol=paper_protocol,
                random_stop_probability=float(cfg.get("random_stop_probability", 1.0 / max(int(cfg.get("max_steps", 128)), 1))),
                rng=random_rng,
                stop_at_end=args.baseline == "reference",
                max_decisions=int(cfg.get("max_steps", 128)) if paper_protocol else None,
                initial_executed_path=bootstrap_positions,
                initial_path_length_m=bootstrap_path_length_m,
            )
            metrics = summarize_episode(
                pred_path=executed,
                ref_path=ref_path,
                goal=goal,
                success_radius=success_radius,
                shortest_path_length=validated_shortest_path_length(episode),
                stopped=stopped,
            )
            trajectory_file = run_dir / "trajectories" / f"{args.baseline}_{trial:03d}_{episode_id}.json"
            beacon_file = run_dir / "beacons" / f"{args.baseline}_{trial:03d}_{episode_id}.json"
            write_json({"episode_id": episode_id, "placements": placements}, beacon_file)
            write_json(
                {
                    "episode_id": episode_id,
                    "baseline": args.baseline,
                    "targets": targets,
                    "executed_path": executed,
                    "reference_path": ref_path,
                    "reference_bootstrap": reference_bootstrap,
                    "reset_stabilization": reset_stabilization,
                    "reference_time_sec": reference_time_sec,
                    "timeout_sec": timeout_sec,
                    "episode_elapsed_sec": episode_elapsed_sec,
                    "reference_path_length_m": reference_path_length_m,
                    "success_radius_m": success_radius,
                    "max_teleport_step_m": max_teleport_step_m,
                    "max_teleport_vertical_step_m": max_teleport_vertical_step_m,
                    "teleport_keep_initial_height": teleport_keep_initial_height,
                    "teleport_hover_after_setpose": teleport_hover_after_setpose,
                    "teleport_pause_after_setpose": teleport_pause_after_setpose,
                    "teleport_zero_velocity": teleport_zero_velocity,
                    "reference_bootstrap_steps": reference_bootstrap_steps,
                    "reference_bootstrap_count": len(reference_bootstrap),
                    "path_length_limit_m": path_length_limit_m,
                    "executed_path_length_m": executed_path_length_m,
                    "stationary_timeout_sec": stationary_timeout_sec,
                    "stationary_radius_m": stationary_radius_m,
                    "stationary_duration_sec": stationary_duration_sec,
                    "termination_reason": termination_reason,
                    "stopped": stopped,
                    "termination_mode": "paper" if paper_protocol else "legacy",
                },
                trajectory_file,
            )
            detail = {
                "trial": trial,
                "episode_id": episode_id,
                "scene_id": episode["scene_id"],
                "difficulty": episode.get("difficulty"),
                "baseline": args.baseline,
                **metrics,
                "cycle_times": cycle_times,
                "collisions": collisions,
                "reference_time_sec": reference_time_sec,
                "timeout_sec": timeout_sec,
                "episode_elapsed_sec": episode_elapsed_sec,
                "reference_path_length_m": reference_path_length_m,
                "success_radius_m": success_radius,
                "max_teleport_step_m": max_teleport_step_m,
                "max_teleport_vertical_step_m": max_teleport_vertical_step_m,
                "teleport_keep_initial_height": teleport_keep_initial_height,
                "teleport_hover_after_setpose": teleport_hover_after_setpose,
                "teleport_pause_after_setpose": teleport_pause_after_setpose,
                "teleport_zero_velocity": teleport_zero_velocity,
                "reference_bootstrap_steps": reference_bootstrap_steps,
                "reference_bootstrap_count": len(reference_bootstrap),
                "path_length_limit_m": path_length_limit_m,
                "executed_path_length_m": executed_path_length_m,
                "stationary_timeout_sec": stationary_timeout_sec,
                "stationary_radius_m": stationary_radius_m,
                "stationary_duration_sec": stationary_duration_sec,
                "termination_reason": termination_reason,
                "stopped": stopped,
                "termination_mode": "paper" if paper_protocol else "legacy",
                "beacons_placed": sum(1 for item in placements if item.get("placed")),
                "beacons_requested": len(placements),
                "trajectory_file": str(trajectory_file),
                "beacon_file": str(beacon_file),
            }
            details.append(detail)
            print(json.dumps({"trial": trial + 1, "trials": len(selected), "termination_reason": termination_reason, **{k: detail[k] for k in METRIC_KEYS}}, ensure_ascii=False), flush=True)
    finally:
        env.close()

    write_jsonl(details, run_dir / "details.jsonl")
    summary = _summarize(details)
    summary["baseline"] = args.baseline
    summary["run_dir"] = str(run_dir)
    summary["status"] = "diagnostic_baseline_complete"
    summary["evaluation_backend"] = "airsim_online_baseline"
    write_json(summary, run_dir / "metrics.json")
    (run_dir / "paper_metrics.md").write_text(_paper_markdown(args.baseline, summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
