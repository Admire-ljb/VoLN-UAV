from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2
from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.evaluation.metrics import aggregate_metrics, summarize_episode
from voln_uav.models.policy import VoLNPolicy
from voln_uav.simulators.airsim_env import AirSimRouteEnv


def resolve_config_path(path: str | Path, config: dict[str, Any]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    config_dir = Path(config.get("_config_dir", "."))
    for root in (Path.cwd(), config_dir, config_dir.parent):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (config_dir.parent / candidate).resolve()


def load_scene_mapping(path: str | Path | None, config: dict[str, Any]) -> dict[str, str]:
    if not path:
        return {}
    with resolve_config_path(path, config).open("r", encoding="utf-8") as f:
        return json.load(f)


class AirSimProcess:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.proc: subprocess.Popen[str] | None = None

    def launch(self, scene_id: str, port: int) -> None:
        env_cfg = self.cfg.get("env", {})
        if not bool(env_cfg.get("auto_launch", False)):
            return
        root_path = resolve_config_path(env_cfg["root_path"], self.cfg)
        mapping = load_scene_mapping(env_cfg.get("scene_mapping"), self.cfg)
        mapping.update(env_cfg.get("scene_mapping_inline", {}))
        if scene_id not in mapping:
            raise KeyError(f"No AirSim executable mapping for scene {scene_id}")
        exec_path = (root_path / mapping[scene_id]).resolve()
        if not exec_path.exists():
            raise FileNotFoundError(f"AirSim executable not found: {exec_path}")
        args = [str(exec_path)]
        for item in env_cfg.get("executable_args", ["--port", "{port}"]):
            args.append(str(item).format(port=port, scene=scene_id))
        self.proc = subprocess.Popen(args)
        time.sleep(float(env_cfg.get("launch_wait_sec", 20.0)))

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class AirSimClosedLoopEvaluator:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self.benchmark_root = Path(config["benchmark_root"])
        self.work_dir = ensure_dir(config["work_dir"])
        episodes = read_jsonl(self.benchmark_root / config["episodes_file"])
        self.episodes = self._filter_episodes(episodes)
        self.policy = VoLNPolicy(
            config=config,
            semantic_bank_path=self.benchmark_root / config["semantic_bank"],
            adapter_ckpt=config["adapter_ckpt"],
            planner_ckpt=config["planner_ckpt"],
            device=device,
        )

    def _filter_episodes(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scene_allowlist = set(self.cfg.get("scene_allowlist", []) or [])
        difficulty_allowlist = set(self.cfg.get("difficulty_allowlist", []) or [])
        episode_limit = self.cfg.get("episode_limit")
        filtered = []
        for episode in episodes:
            if scene_allowlist and episode["scene_id"] not in scene_allowlist:
                continue
            if difficulty_allowlist and episode.get("difficulty") not in difficulty_allowlist:
                continue
            filtered.append(episode)
            if episode_limit is not None and len(filtered) >= int(episode_limit):
                break
        return filtered

    def _history(self, live_states: list[dict[str, Any]], memory_len: int) -> list[dict[str, Any]]:
        hist = live_states[-memory_len:]
        while len(hist) < memory_len:
            hist.insert(0, hist[0])
        return hist

    def _episode_groups(self) -> list[tuple[str, list[dict[str, Any]]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for episode in self.episodes:
            grouped.setdefault(episode["scene_id"], []).append(episode)
        return list(grouped.items())

    def evaluate(self) -> dict[str, Any]:
        port = int(self.cfg.get("port", 41451))
        launcher = AirSimProcess(self.cfg)
        all_metrics = []
        details = []
        cycle_times = []
        execution_errors = 0
        collisions = 0
        for scene_id, episodes in self._episode_groups():
            launcher.launch(scene_id, port)
            env = AirSimRouteEnv(
                ip=str(self.cfg.get("ip", "127.0.0.1")),
                port=port,
                camera=str(self.cfg.get("camera", "FrontCamera")),
                image_type=str(self.cfg.get("image_type", "Scene")),
                work_dir=self.work_dir,
                speed=float(self.cfg.get("speed", 3.0)),
                move_timeout_sec=float(self.cfg.get("move_timeout_sec", 15.0)),
                settle_sec=float(self.cfg.get("settle_sec", 0.05)),
            )
            try:
                env.connect(timeout_sec=float(self.cfg.get("connect_timeout_sec", 60.0)))
                for episode in episodes:
                    result = self._evaluate_episode(env, episode)
                    all_metrics.append(result["metrics"])
                    details.append(result["detail"])
                    cycle_times.extend(result["cycle_times"])
                    execution_errors += int(result["execution_errors"])
                    collisions += int(result["collisions"])
            finally:
                env.close()
                launcher.close()

        agg = aggregate_metrics(all_metrics)
        sorted_ct = sorted(cycle_times)
        p95_idx = min(int(0.95 * max(len(sorted_ct) - 1, 0)), max(len(sorted_ct) - 1, 0))
        summary = {
            **agg,
            "episodes": len(details),
            "CT_mean": sum(cycle_times) / max(len(cycle_times), 1),
            "CT_p95": sorted_ct[p95_idx] if sorted_ct else 0.0,
            "EER": execution_errors / max(len(cycle_times), 1),
            "collisions": collisions,
            "details_file": str(self.work_dir / "details.jsonl"),
        }
        write_jsonl(details, self.work_dir / "details.jsonl")
        write_json(summary, self.work_dir / "metrics.json")
        return summary

    def _evaluate_episode(self, env: AirSimRouteEnv, episode: dict[str, Any]) -> dict[str, Any]:
        episode_id = episode["episode_id"]
        env.reset_to_episode_start(episode)
        live_states: list[dict[str, Any]] = []
        executed_path: list[list[float]] = []
        cycle_times: list[float] = []
        episode_errors = 0
        episode_collisions = 0
        previous_position: list[float] | None = None
        max_steps = int(self.cfg["max_steps"])
        memory_len = int(self.cfg["model"]["memory_len"])
        success_radius = float(self.cfg["success_radius"])
        stop_threshold = float(self.cfg.get("stop_probability", 0.7))
        min_steps_before_stop = int(self.cfg.get("min_steps_before_stop", 3))
        control_mode = str(self.cfg.get("control_mode", "move_to_position"))

        for step_idx in range(max_steps):
            obs = env.current_state(episode_id, step_idx, previous_position=previous_position)
            previous_position = obs.position
            live_states.append(obs.state)
            executed_path.append(obs.position)
            if obs.collision:
                episode_collisions += 1
            if AirSimRouteEnv.reached_goal(obs.position, episode, success_radius):
                break

            history_states = self._history(live_states, memory_len=memory_len)
            start = time.perf_counter()
            try:
                action = self.policy.act(obs.state, history_states, episode["visual_goal"])
                waypoint = action["waypoints"][0].detach().cpu().tolist()
                invalid = False
            except Exception as exc:
                action = {"error": repr(exc), "stop_prob": 0.0}
                waypoint = episode["states"][min(step_idx + 1, len(episode["states"]) - 1)]["position"]
                invalid = True
            cycle_time = time.perf_counter() - start
            cycle_times.append(cycle_time)
            if invalid or cycle_time > float(self.cfg["budget_sec"]):
                episode_errors += 1
            env.move_to_waypoint(obs.position, waypoint, control_mode=control_mode)
            if step_idx >= min_steps_before_stop and float(action.get("stop_prob", 0.0)) >= stop_threshold:
                next_dist = l2(waypoint, episode["states"][-1]["position"])
                if next_dist <= success_radius:
                    break

        metrics = summarize_episode(
            pred_path=executed_path,
            ref_path=[state["position"] for state in episode["states"]],
            goal=episode["states"][-1]["position"],
            success_radius=success_radius,
            shortest_path_length=float(episode.get("shortest_path_length", episode.get("path_length", 1.0))),
        )
        trajectory_path = self.work_dir / "trajectories" / f"{episode_id}.json"
        ensure_dir(trajectory_path.parent)
        write_json(
            {
                "episode_id": episode_id,
                "scene_id": episode["scene_id"],
                "executed_path": executed_path,
                "reference_path": [state["position"] for state in episode["states"]],
            },
            trajectory_path,
        )
        detail = {
            "episode_id": episode_id,
            "scene_id": episode["scene_id"],
            "difficulty": episode.get("difficulty"),
            **metrics,
            "cycle_errors": episode_errors,
            "collisions": episode_collisions,
            "num_steps": len(executed_path),
            "trajectory_file": str(trajectory_path),
        }
        return {
            "metrics": metrics,
            "detail": detail,
            "cycle_times": cycle_times,
            "execution_errors": episode_errors,
            "collisions": episode_collisions,
        }
