from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from voln_uav.common.config import load_config
from voln_uav.common.io import ensure_dir, read_json, read_jsonl, write_json
from voln_uav.evaluation.airsim_loop import check_airsim_readiness
from voln_uav.simulators.airsim_env import AirSimRouteEnv, yaw_to_quaternion


def _state_from_item(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        if "position" not in item:
            raise ValueError(f"trajectory state {index} is missing position")
        out = dict(item)
        out["position"] = [float(v) for v in out["position"][:3]]
        return out
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {"position": [float(item[0]), float(item[1]), float(item[2])]}
    raise ValueError(f"unsupported trajectory state {index}: {item!r}")


def load_trajectory_file(path: str | Path, scene_id: str, difficulty: str) -> dict[str, Any]:
    source = Path(path)
    payload = read_json(source)
    if isinstance(payload, dict) and "states" in payload:
        episode = dict(payload)
        episode["states"] = [_state_from_item(item, idx) for idx, item in enumerate(payload["states"])]
        episode.setdefault("episode_id", source.stem)
        episode.setdefault("scene_id", scene_id)
        episode.setdefault("difficulty", difficulty)
        return episode
    if isinstance(payload, dict):
        items = payload.get("positions", payload.get("trajectory"))
        if items is None:
            raise ValueError("trajectory file must contain states, positions, or trajectory")
        episode_id = str(payload.get("episode_id", source.stem))
        scene = str(payload.get("scene_id", scene_id))
        diff = str(payload.get("difficulty", difficulty))
    else:
        items = payload
        episode_id = source.stem
        scene = scene_id
        diff = difficulty
    if not isinstance(items, list):
        raise ValueError("trajectory states/positions must be a list")
    return {
        "episode_id": episode_id,
        "scene_id": scene,
        "difficulty": diff,
        "states": [_state_from_item(item, idx) for idx, item in enumerate(items)],
    }


def select_dataset_episode(config: dict[str, Any], episode_id: str | None, episode_index: int) -> dict[str, Any]:
    benchmark_root = Path(config["benchmark_root"])
    episodes = read_jsonl(benchmark_root / config["episodes_file"])
    if episode_id:
        for episode in episodes:
            if str(episode.get("episode_id")) == episode_id:
                return episode
        raise KeyError(f"episode_id not found: {episode_id}")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode index {episode_index} out of range for {len(episodes)} episodes")
    return episodes[episode_index]


def set_vehicle_pose(env: AirSimRouteEnv, state: dict[str, Any]) -> None:
    x, y, z = [float(v) for v in state["position"][:3]]
    yaw = float(state.get("yaw", 0.0))
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    pose = env.airsim.Pose(
        env.airsim.Vector3r(x, y, z),
        env.airsim.Quaternionr(qx, qy, qz, qw),
    )
    env.client.simSetVehiclePose(pose, True)
    env.client.enableApiControl(True)
    env.client.armDisarm(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset or JSON trajectory in AirSim using simSetVehiclePose.")
    parser.add_argument("--config", default="configs/eval_airsim_dataset_release.yaml")
    parser.add_argument("--episode-id")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--trajectory-file")
    parser.add_argument("--trajectory-scene-id", default="custom")
    parser.add_argument("--trajectory-difficulty", default="Normal")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--delay-sec", type=float, default=0.7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-beacons", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episode = (
        load_trajectory_file(args.trajectory_file, args.trajectory_scene_id, args.trajectory_difficulty)
        if args.trajectory_file
        else select_dataset_episode(cfg, args.episode_id, args.episode_index)
    )
    issues = check_airsim_readiness(cfg, [episode])
    if issues:
        raise RuntimeError("AirSim trajectory review is not ready:\n" + "\n".join(f"- {issue}" for issue in issues))

    env = AirSimRouteEnv(
        ip=str(cfg.get("ip", "127.0.0.1")),
        port=int(cfg.get("port", 41451)),
        camera=str(cfg.get("camera", "0")),
        image_type=str(cfg.get("image_type", "Scene")),
        work_dir=cfg.get("work_dir", "work_dirs/trajectory_review"),
        speed=float(cfg.get("speed", 3.0)),
        move_timeout_sec=float(cfg.get("move_timeout_sec", 15.0)),
        settle_sec=float(cfg.get("settle_sec", 0.02)),
        takeoff_timeout_sec=float(cfg.get("takeoff_timeout_sec", 10.0)),
    )
    env.connect(timeout_sec=float(cfg.get("connect_timeout_sec", 60.0)))
    env.reset_to_episode_start(episode)
    beacon_cfg = dict(cfg.get("beacon_placement", {}) or {})
    if args.no_beacons:
        beacon_cfg["enabled"] = False
    placements = env.place_beacons_for_episode(episode, beacon_cfg, seed=int(cfg.get("seed", 0)))
    out_dir = ensure_dir(Path(cfg.get("work_dir", "work_dirs/trajectory_review")) / "trajectory_review")
    write_json({"episode_id": episode.get("episode_id"), "placements": placements}, out_dir / "beacons.json")

    states = list(episode["states"])
    stride = max(int(args.stride), 1)
    delay = max(float(args.delay_sec), 0.0)
    print(
        json.dumps(
            {
                "mode": "simSetVehiclePose",
                "episode_id": episode.get("episode_id"),
                "scene_id": episode.get("scene_id"),
                "difficulty": episode.get("difficulty"),
                "states": len(states),
                "stride": stride,
                "loops": max(int(args.loops), 1),
                "delay_sec": delay,
                "beacons_requested": len(placements),
                "beacons_placed": sum(1 for item in placements if item.get("placed")),
                "beacon_file": str(out_dir / "beacons.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for loop in range(max(int(args.loops), 1)):
        for idx in range(0, len(states), stride):
            set_vehicle_pose(env, states[idx])
            if idx % max(stride * 5, 1) == 0 or idx == len(states) - 1:
                print(json.dumps({"loop": loop + 1, "step": idx, "position": states[idx]["position"]}, ensure_ascii=False), flush=True)
            time.sleep(delay)
        if (len(states) - 1) % stride != 0:
            set_vehicle_pose(env, states[-1])
            print(json.dumps({"loop": loop + 1, "step": len(states) - 1, "position": states[-1]["position"]}, ensure_ascii=False), flush=True)
            time.sleep(delay)
    print(json.dumps({"done": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
