from __future__ import annotations

import importlib.util
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2, path_length
from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.baselines.random_policy import RandomPolicy
from voln_uav.evaluation.metrics import METRIC_KEYS, aggregate_by_difficulty, aggregate_metrics, reference_travel_time, summarize_episode
from voln_uav.evaluation.paper_protocol import select_available_episodes
from voln_uav.evaluation.termination import StationaryDetector
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


def _port_is_open(ip: str, port: int, timeout_sec: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(float(timeout_sec))
        return sock.connect_ex((ip, int(port))) == 0


def filter_airsim_episodes(config: dict[str, Any], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the exact episode selection used by AirSim evaluation and preflight."""
    episodes, _coverage = select_available_episodes(episodes, config)
    difficulty_allowlist = set(config.get("difficulty_allowlist", []) or [])
    episode_index = max(int(config.get("episode_index", 0)), 0)
    episode_stride = max(int(config.get("episode_stride", 1)), 1)
    trials = config.get("trials", config.get("episode_limit"))
    filtered = [
        episode
        for episode in episodes
        if not difficulty_allowlist or episode.get("difficulty") in difficulty_allowlist
    ]
    selected = filtered[episode_index::episode_stride]
    if trials is not None:
        selected = selected[: max(int(trials), 0)]
    return selected


def check_airsim_readiness(config: dict[str, Any], episodes: list[dict[str, Any]] | None = None) -> list[str]:
    issues: list[str] = []
    if importlib.util.find_spec("airsim") is None:
        issues.append("Install the AirSim Python package, for example: pip install -e .[real]")

    env_cfg = config.get("env", {})
    auto_launch = bool(env_cfg.get("auto_launch", False))
    if auto_launch:
        root_path = resolve_config_path(env_cfg.get("root_path", ""), config)
        if not root_path.exists():
            issues.append(f"AirSim env root does not exist: {root_path}")
        mapping_path = env_cfg.get("scene_mapping")
        mapping: dict[str, str] = {}
        if mapping_path:
            resolved_mapping = resolve_config_path(mapping_path, config)
            if resolved_mapping.exists():
                mapping = load_scene_mapping(mapping_path, config)
            else:
                issues.append(f"AirSim scene mapping does not exist: {resolved_mapping}")
        mapping.update(env_cfg.get("scene_mapping_inline", {}))
        scene_ids = sorted({episode["scene_id"] for episode in episodes or []})
        for scene_id in scene_ids:
            rel_exec = mapping.get(scene_id)
            if rel_exec is None:
                issues.append(f"No AirSim executable mapping for scene {scene_id}")
                continue
            exec_path = (root_path / rel_exec).resolve()
            if not exec_path.exists():
                issues.append(f"AirSim executable not found for scene {scene_id}: {exec_path}")
    else:
        scene_ids = sorted({episode["scene_id"] for episode in episodes or []})
        if len(scene_ids) > 1:
            issues.append(
                "Manual AirSim mode (env.auto_launch: false) can evaluate only one scene at a time; "
                f"the selected episodes span {len(scene_ids)} scenes. Set scene_allowlist to the open scene "
                "or use env.auto_launch: true for a full-split run."
            )
        ip = str(config.get("ip", "127.0.0.1"))
        port = int(config.get("port", 41451))
        if not _port_is_open(ip, port):
            issues.append(
                f"No AirSim server is listening at {ip}:{port}. Start a scene first or set env.auto_launch: true."
            )
    return issues


def raise_for_airsim_readiness(config: dict[str, Any], episodes: list[dict[str, Any]] | None = None) -> None:
    issues = check_airsim_readiness(config, episodes)
    if not issues:
        return
    detail = "\n".join(f"- {issue}" for issue in issues)
    raise RuntimeError(f"AirSim evaluation is not ready:\n{detail}")


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
        raw_episodes = read_jsonl(self.benchmark_root / config["episodes_file"])
        _available, self.scene_coverage = select_available_episodes(raw_episodes, config)
        self.episodes = filter_airsim_episodes(config, raw_episodes)
        self.scene_coverage["selected_episodes_after_filters"] = len(self.episodes)
        if self.episodes:
            raise_for_airsim_readiness(config, self.episodes)
        self.controller = str(config.get("controller", "policy")).lower()
        if self.controller not in {"policy", "reference", "random"}:
            raise ValueError(f"Unsupported AirSim controller: {self.controller}")
        self.policy = None
        if self.controller == "policy" and self.episodes:
            self.policy = VoLNPolicy(
                config=config,
                semantic_bank_path=self.benchmark_root / config["semantic_bank"],
                adapter_ckpt=config["adapter_ckpt"],
                planner_ckpt=config["planner_ckpt"],
                device=device,
            )

    def _filter_episodes(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return filter_airsim_episodes(self.cfg, episodes)

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

    def _load_completed_details(self) -> list[dict[str, Any]]:
        details_file = self.work_dir / "details.jsonl"
        if not bool(self.cfg.get("resume", True)) or not details_file.exists():
            return []
        return read_jsonl(details_file)

    def _metrics_from_detail(self, detail: dict[str, Any]) -> dict[str, float]:
        return {key: float(detail.get(key, 0.0)) for key in METRIC_KEYS}

    def _summary(
        self,
        details: list[dict[str, Any]],
        cycle_times: list[float],
        execution_errors: int,
        collisions: int,
    ) -> dict[str, Any]:
        agg = aggregate_metrics([self._metrics_from_detail(detail) for detail in details])
        sorted_ct = sorted(cycle_times)
        p95_idx = min(int(0.95 * max(len(sorted_ct) - 1, 0)), max(len(sorted_ct) - 1, 0))
        return {
            **agg,
            "status": "complete",
            "episodes": len(details),
            "total_episodes": len(self.episodes),
            "CT_mean": sum(cycle_times) / max(len(cycle_times), 1),
            "CT_p95": sorted_ct[p95_idx] if sorted_ct else 0.0,
            "EER": execution_errors / max(len(cycle_times), 1),
            "collisions": collisions,
            "by_difficulty": aggregate_by_difficulty(details),
            "scene_coverage": self.scene_coverage,
            "details_file": str(self.work_dir / "details.jsonl"),
            "progress_file": str(self.work_dir / "progress.json"),
        }

    def _append_detail(self, detail: dict[str, Any]) -> None:
        details_file = self.work_dir / "details.jsonl"
        ensure_dir(details_file.parent)
        with details_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    def evaluate(self) -> dict[str, Any]:
        write_json(self.scene_coverage, self.work_dir / "scene_coverage.json")
        if not self.episodes:
            summary = {
                "status": "skipped_no_available_episodes",
                "episodes": 0,
                "total_episodes": 0,
                "scene_coverage": self.scene_coverage,
            }
            write_json(summary, self.work_dir / "metrics.json")
            return summary
        port = int(self.cfg.get("port", 41451))
        launcher = AirSimProcess(self.cfg)
        details = self._load_completed_details()
        completed_ids = {str(detail["episode_id"]) for detail in details}
        cycle_times: list[float] = []
        execution_errors = 0
        collisions = 0
        for detail in details:
            cycle_times.extend(float(v) for v in detail.get("cycle_times", []))
            execution_errors += int(detail.get("cycle_errors", 0))
            collisions += int(detail.get("collisions", 0))

        log_every = max(int(self.cfg.get("log_every", 1)), 1)
        for scene_id, episodes in self._episode_groups():
            pending = [episode for episode in episodes if str(episode["episode_id"]) not in completed_ids]
            if not pending:
                continue
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
                takeoff_timeout_sec=float(self.cfg.get("takeoff_timeout_sec", 10.0)),
                max_teleport_step_m=float(self.cfg.get("max_teleport_step_m", self.cfg.get("speed", 3.0))),
                max_teleport_vertical_step_m=float(self.cfg.get("max_teleport_vertical_step_m", 0.5)),
                teleport_keep_initial_height=bool(self.cfg.get("teleport_keep_initial_height", False)),
                teleport_hover_after_setpose=bool(self.cfg.get("teleport_hover_after_setpose", True)),
                teleport_pause_after_setpose=bool(self.cfg.get("teleport_pause_after_setpose", False)),
                teleport_zero_velocity=bool(self.cfg.get("teleport_zero_velocity", True)),
            )
            try:
                env.connect(timeout_sec=float(self.cfg.get("connect_timeout_sec", 60.0)))
                for episode in pending:
                    result = self._evaluate_episode(env, episode)
                    detail = result["detail"]
                    detail["cycle_times"] = [float(v) for v in result["cycle_times"]]
                    details.append(detail)
                    completed_ids.add(str(episode["episode_id"]))
                    self._append_detail(detail)
                    cycle_times.extend(result["cycle_times"])
                    execution_errors += int(result["execution_errors"])
                    collisions += int(result["collisions"])
                    summary = self._summary(details, cycle_times, execution_errors, collisions)
                    write_json(summary, self.work_dir / "progress.json")
                    if len(details) % log_every == 0 or len(details) == len(self.episodes):
                        print(f"[eval_airsim] {len(details)}/{len(self.episodes)} episodes complete", flush=True)
            finally:
                env.close()
                launcher.close()

        summary = self._summary(details, cycle_times, execution_errors, collisions)
        write_jsonl(details, self.work_dir / "details.jsonl")
        write_json(summary, self.work_dir / "metrics.json")
        return summary

    def _evaluate_episode(self, env: AirSimRouteEnv, episode: dict[str, Any]) -> dict[str, Any]:
        episode_id = episode["episode_id"]
        random_policy = None
        if self.controller == "random":
            episode_hash = int(hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest()[:8], 16)
            random_policy = RandomPolicy(
                horizon=int(self.cfg["model"]["horizon"]),
                step_scale=float(self.cfg.get("max_teleport_step_m", self.cfg.get("speed", 3.0))),
                vertical_step_scale=float(self.cfg.get("max_teleport_vertical_step_m", 0.5)),
                stop_probability=float(
                    self.cfg.get("random_stop_probability", 1.0 / max(int(self.cfg["max_steps"]), 1))
                ),
                seed=int(self.cfg.get("seed", 7)) + episode_hash,
            )
        control_mode = str(self.cfg.get("control_mode", "move_to_position"))
        max_teleport_step_m = float(self.cfg.get("max_teleport_step_m", env.max_teleport_step_m))
        max_teleport_vertical_step_m = float(self.cfg.get("max_teleport_vertical_step_m", env.max_teleport_vertical_step_m))
        teleport_keep_initial_height = bool(self.cfg.get("teleport_keep_initial_height", env.teleport_keep_initial_height))
        teleport_hover_after_setpose = bool(self.cfg.get("teleport_hover_after_setpose", env.teleport_hover_after_setpose))
        teleport_pause_after_setpose = bool(self.cfg.get("teleport_pause_after_setpose", env.teleport_pause_after_setpose))
        teleport_zero_velocity = bool(self.cfg.get("teleport_zero_velocity", env.teleport_zero_velocity))
        reference_bootstrap_steps = max(int(self.cfg.get("reference_bootstrap_steps", 3)), 0)
        fast_reset = bool(self.cfg.get("fast_reset", control_mode == "teleport"))
        reset_stabilization = env.reset_to_episode_start(episode, ensure_flying=not fast_reset)
        beacon_placements = env.place_beacons_for_episode(
            episode,
            config=self.cfg.get("beacon_placement", {}),
            seed=int(self.cfg.get("seed", 0)),
        )
        beacon_path = self.work_dir / "beacons" / f"{episode_id}.json"
        ensure_dir(beacon_path.parent)
        write_json({"episode_id": episode_id, "placements": beacon_placements}, beacon_path)
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
        live_states: list[dict[str, Any]] = []
        executed_path: list[list[float]] = list(bootstrap_positions)
        action_trace: list[dict[str, Any]] = []
        cycle_times: list[float] = []
        episode_errors = 0
        episode_collisions = 0
        previous_position: list[float] | None = bootstrap_positions[-1] if bootstrap_positions else None
        max_steps = int(self.cfg["max_steps"])
        memory_len = int(self.cfg["model"]["memory_len"])
        success_radius = float(self.cfg["success_radius"])
        configured_stop_threshold = self.cfg.get("stop_probability")
        if configured_stop_threshold is not None:
            stop_threshold = float(configured_stop_threshold)
        elif self.policy is not None:
            stop_threshold = float(self.policy.stop_threshold)
        else:
            stop_threshold = 0.5
        min_steps_before_stop = int(self.cfg.get("min_steps_before_stop", 0))
        paper_protocol = str(self.cfg.get("termination_mode", "paper")).lower() == "paper"
        reference_stride = max(int(self.cfg.get("reference_stride", 1)), 1)
        ref_path = [state["position"] for state in episode["states"]]
        reference_path_length_m = path_length(ref_path)
        reference_time_sec = reference_travel_time(ref_path, env.speed)
        timeout_factor = float(self.cfg.get("episode_timeout_factor", 2.0))
        timeout_sec = max(reference_time_sec * timeout_factor, float(self.cfg.get("minimum_episode_timeout_sec", 1.0)))
        path_length_factor = float(self.cfg.get("episode_path_length_factor", 2.0))
        path_length_limit_m = max(
            reference_path_length_m * path_length_factor,
            float(self.cfg.get("minimum_episode_path_length_m", 1.0)),
        )
        executed_path_length_m = sum(float(item.get("executed_distance_m", 0.0)) for item in reference_bootstrap)
        stationary_timeout_sec = float(self.cfg.get("stationary_timeout_sec", 10.0))
        stationary_radius_m = float(self.cfg.get("stationary_radius_m", 0.5))
        stationary_detector = StationaryDetector(timeout_sec=stationary_timeout_sec, radius_m=stationary_radius_m)
        episode_started_at = time.perf_counter()
        termination_reason = "max_steps"
        policy_stopped = False


        for step_idx in range(max_steps):
            if not paper_protocol and time.perf_counter() - episode_started_at >= timeout_sec:
                termination_reason = "timeout"
                break
            obs = env.current_state(episode_id, step_idx, previous_position=previous_position)
            previous_position = obs.position
            live_states.append(obs.state)
            if not executed_path or l2(executed_path[-1], obs.position) > 1e-6:
                executed_path.append(obs.position)
            if obs.collision:
                episode_collisions += 1
            if not paper_protocol and AirSimRouteEnv.reached_goal(obs.position, episode, success_radius):
                termination_reason = "goal_reached"
                break
            if not paper_protocol and executed_path_length_m >= path_length_limit_m:
                termination_reason = "path_length_limit"
                break
            if not paper_protocol and stationary_detector.update(obs.position, time.perf_counter()):
                termination_reason = "stationary_timeout"
                break

            history_states = self._history(live_states, memory_len=memory_len)
            start = time.perf_counter()
            reference_idx = min(max(bootstrap_reference_offset, 0) + (step_idx + 1) * reference_stride, len(episode["states"]) - 1)
            reference_next = episode["states"][reference_idx]["position"]
            if self.controller == "reference":
                reference_at_goal = reference_idx >= len(episode["states"]) - 1 and l2(obs.position, episode["states"][-1]["position"]) <= success_radius
                action = {"stop_prob": 1.0 if reference_at_goal else 0.0, "controller": "reference"}
                waypoint_segment = [reference_next]
                invalid = False
            elif self.controller == "random":
                assert random_policy is not None
                action = random_policy.act(obs.state)
                waypoint_segment = action["waypoints"].detach().cpu().tolist()
                invalid = not waypoint_segment or any(len(point) < 3 for point in waypoint_segment)
            else:
                try:
                    assert self.policy is not None
                    action = self.policy.act(obs.state, history_states, episode["visual_goal"])
                    waypoint_segment = action["waypoints"].detach().cpu().tolist()
                    invalid = not waypoint_segment or any(len(point) < 3 for point in waypoint_segment)
                except Exception as exc:
                    action = {"error": repr(exc), "stop_prob": 0.0, "controller": "policy_error_hold"}
                    waypoint_segment = [obs.position]
                    invalid = True
            waypoint = waypoint_segment[-1]
            cycle_time = time.perf_counter() - start
            cycle_times.append(cycle_time)
            if invalid or cycle_time > float(self.cfg["budget_sec"]):
                episode_errors += 1
            if step_idx >= min_steps_before_stop and float(action.get("stop_prob", 0.0)) >= stop_threshold:
                policy_stopped = True
                termination_reason = "policy_stop"
                action_trace.append(
                    {
                        "step": step_idx,
                        "position": obs.position,
                        "waypoint": [float(v) for v in obs.position[:3]],
                        "requested_waypoints": [[float(v) for v in point[:3]] for point in waypoint_segment],
                        "executed_waypoint": [float(v) for v in obs.position[:3]],
                        "reference_next": reference_next,
                        "controller": str(action.get("controller", self.controller)),
                        "stop_prob": float(action.get("stop_prob", 0.0)),
                        "cycle_time": cycle_time,
                        "invalid": invalid,
                        "error": action.get("error"),
                        "stop_executed": True,
                    }
                )
                break
            movement_steps: list[dict[str, Any]] = []
            segment_position = obs.position
            for segment_index, segment_waypoint in enumerate(waypoint_segment):
                movement = env.move_to_waypoint(
                    segment_position,
                    segment_waypoint,
                    control_mode=control_mode,
                    max_teleport_step_m=max_teleport_step_m,
                    max_teleport_vertical_step_m=max_teleport_vertical_step_m,
                    teleport_keep_initial_height=teleport_keep_initial_height,
                    teleport_hover_after_setpose=teleport_hover_after_setpose,
                    teleport_pause_after_setpose=teleport_pause_after_setpose,
                    teleport_zero_velocity=teleport_zero_velocity,
                )
                next_position = env.current_position()
                executed_path_length_m += l2(segment_position, next_position)
                if not executed_path or l2(executed_path[-1], next_position) > 1e-6:
                    executed_path.append(next_position)
                movement_steps.append(
                    {
                        "segment_index": segment_index,
                        "requested_waypoint": [float(v) for v in segment_waypoint[:3]],
                        "position_after": [float(v) for v in next_position[:3]],
                        **movement,
                    }
                )
                segment_position = next_position
            post_move_position = segment_position
            movement = movement_steps[-1] if movement_steps else {}
            action_trace.append(
                {
                    "step": step_idx,
                    "position": obs.position,
                    "waypoint": [float(v) for v in waypoint[:3]],
                    "requested_waypoints": [[float(v) for v in point[:3]] for point in waypoint_segment],
                    "executed_waypoints": [[float(v) for v in item["position_after"][:3]] for item in movement_steps],
                    "segment_movements": movement_steps,
                    "executed_waypoint": [float(v) for v in movement.get("executed_waypoint", waypoint)[:3]],
                    "height_limited_waypoint": [float(v) for v in movement.get("height_limited_waypoint", waypoint)[:3]],
                    "reference_next": reference_next,
                    "controller": str(action.get("controller", self.controller)),
                    "stop_prob": float(action.get("stop_prob", 0.0)),
                    "cycle_time": cycle_time,
                    "invalid": invalid,
                    "error": action.get("error"),
                    "teleport_clipped": bool(movement.get("teleport_clipped", False)),
                    "requested_distance_m": float(movement.get("requested_distance_m", 0.0)),
                    "executed_distance_m": float(movement.get("executed_distance_m", 0.0)),
                    "max_teleport_step_m": movement.get("max_teleport_step_m"),
                    "max_teleport_vertical_step_m": movement.get("max_teleport_vertical_step_m"),
                    "teleport_vertical_clipped": bool(movement.get("teleport_vertical_clipped", False)),
                    "teleport_height_locked": bool(movement.get("teleport_height_locked", False)),
                    "teleport_height_adjusted": bool(movement.get("teleport_height_adjusted", False)),
                    "teleport_stabilization": movement.get("teleport_stabilization", {}),
                    "requested_vertical_delta_m": float(movement.get("requested_vertical_delta_m", 0.0)),
                    "executed_vertical_delta_m": float(movement.get("executed_vertical_delta_m", 0.0)),
                }
            )
            if not paper_protocol and AirSimRouteEnv.reached_goal(post_move_position, episode, success_radius):
                termination_reason = "goal_reached"
                break
            if not paper_protocol and time.perf_counter() - episode_started_at >= timeout_sec:
                termination_reason = "timeout"
                break
            if not paper_protocol and executed_path_length_m >= path_length_limit_m:
                termination_reason = "path_length_limit"
                break
            if not paper_protocol and stationary_detector.update(post_move_position, time.perf_counter()):
                termination_reason = "stationary_timeout"
                break

        episode_elapsed_sec = time.perf_counter() - episode_started_at

        metrics = summarize_episode(
            pred_path=executed_path,
            ref_path=ref_path,
            goal=episode["states"][-1]["position"],
            success_radius=success_radius,
            shortest_path_length=float(episode.get("shortest_path_length", episode.get("path_length", 1.0))),
            stopped=policy_stopped,
        )
        trajectory_path = self.work_dir / "trajectories" / f"{episode_id}.json"
        ensure_dir(trajectory_path.parent)
        write_json(
            {
                "episode_id": episode_id,
                "scene_id": episode["scene_id"],
                "executed_path": executed_path,
                "reference_path": ref_path,
                "action_trace": action_trace,
                "reference_bootstrap": reference_bootstrap,
                "reset_stabilization": reset_stabilization,
                "beacon_placements": beacon_placements,
                "control_mode": control_mode,
                "max_teleport_step_m": max_teleport_step_m,
                "max_teleport_vertical_step_m": max_teleport_vertical_step_m,
                "teleport_keep_initial_height": teleport_keep_initial_height,
                "teleport_hover_after_setpose": teleport_hover_after_setpose,
                "teleport_pause_after_setpose": teleport_pause_after_setpose,
                "teleport_zero_velocity": teleport_zero_velocity,
                "reference_bootstrap_steps": reference_bootstrap_steps,
                "fast_reset": fast_reset,
                "reference_time_sec": reference_time_sec,
                "timeout_sec": timeout_sec,
                "episode_elapsed_sec": episode_elapsed_sec,
                "reference_path_length_m": reference_path_length_m,
                "success_radius_m": success_radius,
                "stop_threshold": stop_threshold,
                "path_length_limit_m": path_length_limit_m,
                "executed_path_length_m": executed_path_length_m,
                "stationary_timeout_sec": stationary_timeout_sec,
                "stationary_radius_m": stationary_radius_m,
                "stationary_duration_sec": stationary_detector.duration_sec,
                "termination_reason": termination_reason,
                "stopped": policy_stopped,
                "termination_mode": "paper" if paper_protocol else "legacy",
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
            "beacon_file": str(beacon_path),
            "beacons_placed": sum(1 for item in beacon_placements if item.get("placed")),
            "beacons_requested": len(beacon_placements),
            "reference_time_sec": reference_time_sec,
            "max_teleport_step_m": max_teleport_step_m,
            "max_teleport_vertical_step_m": max_teleport_vertical_step_m,
            "teleport_keep_initial_height": teleport_keep_initial_height,
            "teleport_hover_after_setpose": teleport_hover_after_setpose,
            "teleport_pause_after_setpose": teleport_pause_after_setpose,
            "teleport_zero_velocity": teleport_zero_velocity,
            "reference_bootstrap_steps": reference_bootstrap_steps,
            "reference_bootstrap_count": len(reference_bootstrap),
            "timeout_sec": timeout_sec,
            "episode_elapsed_sec": episode_elapsed_sec,
            "reference_path_length_m": reference_path_length_m,
            "success_radius_m": success_radius,
            "stop_threshold": stop_threshold,
            "path_length_limit_m": path_length_limit_m,
            "executed_path_length_m": executed_path_length_m,
            "stationary_timeout_sec": stationary_timeout_sec,
            "stationary_radius_m": stationary_radius_m,
            "stationary_duration_sec": stationary_detector.duration_sec,
            "termination_reason": termination_reason,
            "stopped": policy_stopped,
            "termination_mode": "paper" if paper_protocol else "legacy",
        }
        return {
            "metrics": metrics,
            "detail": detail,
            "cycle_times": cycle_times,
            "execution_errors": episode_errors,
            "collisions": episode_collisions,
        }
