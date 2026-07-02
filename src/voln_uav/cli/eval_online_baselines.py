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
from voln_uav.evaluation.metrics import aggregate_by_difficulty, aggregate_metrics, summarize_episode
from voln_uav.simulators.airsim_env import AirSimRouteEnv


METRIC_KEYS = ("NE", "SR", "OSR", "nDTW", "SPL")


def _position(env: AirSimRouteEnv) -> list[float]:
    state = env.client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return [float(pos.x_val), float(pos.y_val), float(pos.z_val)]


def _collision(env: AirSimRouteEnv) -> bool:
    return bool(getattr(env.client.simGetCollisionInfo(), "has_collided", False))


def _reference_targets(episode: dict[str, Any], stride: int) -> list[list[float]]:
    states = list(episode["states"])
    step = max(int(stride), 1)
    targets = [[float(v) for v in state["position"][:3]] for state in states[::step]]
    final = [float(v) for v in states[-1]["position"][:3]]
    if not targets or targets[-1] != final:
        targets.append(final)
    return targets


def _random_targets(episode: dict[str, Any], rng: random.Random, steps: int | None, stride: int) -> list[list[float]]:
    ref = [[float(v) for v in state["position"][:3]] for state in episode["states"]]
    count = int(steps) if steps is not None else max(4, math.ceil(len(ref) / max(int(stride), 1)))
    count = max(count, 2)
    xs = [p[0] for p in ref]
    ys = [p[1] for p in ref]
    zs = [p[2] for p in ref]
    margin_xy = max(path_length(ref) * 0.08, 15.0)
    min_x, max_x = min(xs) - margin_xy, max(xs) + margin_xy
    min_y, max_y = min(ys) - margin_xy, max(ys) + margin_xy
    min_z, max_z = min(zs) - 6.0, max(zs) + 6.0

    seg_lengths = [l2(ref[i], ref[i + 1]) for i in range(len(ref) - 1)]
    mean_step = sum(seg_lengths) / max(len(seg_lengths), 1)
    step_len = max(mean_step * max(int(stride), 1), 2.0)
    max_turn = math.radians(35.0)
    pos = list(ref[0])
    if len(ref) > 1:
        heading = math.atan2(ref[1][1] - ref[0][1], ref[1][0] - ref[0][0])
    else:
        heading = rng.uniform(-math.pi, math.pi)

    targets: list[list[float]] = []
    for _ in range(count):
        heading += rng.uniform(-max_turn, max_turn)
        distance = step_len * rng.uniform(0.7, 1.3)
        x = pos[0] + math.cos(heading) * distance
        y = pos[1] + math.sin(heading) * distance
        z = pos[2] + rng.uniform(-2.0, 2.0)

        if x < min_x or x > max_x:
            heading = math.pi - heading
            x = max(min_x, min(max_x, x))
        if y < min_y or y > max_y:
            heading = -heading
            y = max(min_y, min(max_y, y))
        z = max(min_z, min(max_z, z))
        pos = [x, y, z]
        targets.append(pos)
    return targets


def _run_targets(
    env: AirSimRouteEnv,
    episode: dict[str, Any],
    targets: list[list[float]],
    control_mode: str,
) -> tuple[list[list[float]], list[float], int]:
    executed = [_position(env)]
    cycle_times: list[float] = []
    collisions = 0
    for target in targets:
        current = _position(env)
        start = time.perf_counter()
        env.move_to_waypoint(current, target, control_mode=control_mode)
        cycle_times.append(time.perf_counter() - start)
        pos = _position(env)
        executed.append(pos)
        if _collision(env):
            collisions += 1
    return executed, cycle_times, collisions


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
    parser = argparse.ArgumentParser(description="Run online AirSim reference/random baselines without teleporting between waypoints.")
    parser.add_argument("--config", default="configs/eval_airsim_dataset_release.yaml")
    parser.add_argument("--baseline", choices=["reference", "random"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference-stride", type=int, default=1)
    parser.add_argument("--random-steps", type=int)
    parser.add_argument("--control-mode", default="move_to_position", choices=["move_to_position", "teleport"])
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
        settle_sec=float(cfg.get("settle_sec", 0.05)),
        takeoff_timeout_sec=float(cfg.get("takeoff_timeout_sec", 10.0)),
    )
    env.connect(timeout_sec=float(cfg.get("connect_timeout_sec", 60.0)))
    try:
        for trial, episode in enumerate(selected):
            episode_id = str(episode["episode_id"])
            env.reset_to_episode_start(episode)
            beacon_cfg = dict(cfg.get("beacon_placement", {}) or {})
            if args.no_beacons:
                beacon_cfg["enabled"] = False
            placements = env.place_beacons_for_episode(episode, beacon_cfg, seed=int(cfg.get("seed", 0)))
            if args.baseline == "reference":
                targets = _reference_targets(episode, stride=int(args.reference_stride))
            else:
                rng = random.Random(int(args.seed) + trial * 1009)
                targets = _random_targets(episode, rng, steps=args.random_steps, stride=int(args.reference_stride))
            executed, cycle_times, collisions = _run_targets(env, episode, targets, control_mode=str(args.control_mode))
            ref_path = [state["position"] for state in episode["states"]]
            metrics = summarize_episode(
                pred_path=executed,
                ref_path=ref_path,
                goal=episode["states"][-1]["position"],
                success_radius=float(cfg["success_radius"]),
                shortest_path_length=float(episode.get("shortest_path_length", episode.get("path_length", 1.0))),
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
                "beacons_placed": sum(1 for item in placements if item.get("placed")),
                "beacons_requested": len(placements),
                "trajectory_file": str(trajectory_file),
                "beacon_file": str(beacon_file),
            }
            details.append(detail)
            print(json.dumps({"trial": trial + 1, "trials": len(selected), **{k: detail[k] for k in METRIC_KEYS}}, ensure_ascii=False), flush=True)
    finally:
        env.close()

    write_jsonl(details, run_dir / "details.jsonl")
    summary = _summarize(details)
    summary["baseline"] = args.baseline
    summary["run_dir"] = str(run_dir)
    write_json(summary, run_dir / "metrics.json")
    (run_dir / "paper_metrics.md").write_text(_paper_markdown(args.baseline, summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
