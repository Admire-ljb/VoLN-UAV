from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from voln_uav.common.geometry import l2
from voln_uav.common.io import ensure_dir, read_jsonl, write_json
from voln_uav.evaluation.metrics import aggregate_by_difficulty, aggregate_metrics, summarize_episode
from voln_uav.evaluation.paper_protocol import select_available_episodes
from voln_uav.models.policy import VoLNPolicy
from voln_uav.simulators.offline_env import RouteReplayEnv


class ClosedLoopEvaluator:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self.benchmark_root = Path(config["benchmark_root"])
        raw_episodes = read_jsonl(self.benchmark_root / config["episodes_file"])
        self.episodes, self.scene_coverage = select_available_episodes(raw_episodes, config)
        episode_limit = config.get("episode_limit")
        if episode_limit is not None:
            self.episodes = self.episodes[: int(episode_limit)]
        self.scene_coverage["selected_episodes_after_limit"] = len(self.episodes)
        self.policy = None
        if self.episodes:
            self.policy = VoLNPolicy(
                config=config,
                semantic_bank_path=self.benchmark_root / config["semantic_bank"],
                adapter_ckpt=config["adapter_ckpt"],
                planner_ckpt=config["planner_ckpt"],
                device=device,
            )

    @staticmethod
    def _execute_waypoint_segment(env: RouteReplayEnv, waypoints: torch.Tensor | None) -> tuple[dict[str, Any], bool]:
        """Execute one H-waypoint action while counting it as one decision."""
        if waypoints is None or not torch.isfinite(waypoints).all():
            env.execution_errors += 1
            env.collisions += 1
            waypoint_list: list[list[float]] = []
        else:
            waypoint_list = waypoints.detach().cpu().tolist()

        for waypoint in waypoint_list:
            candidates = list(range(env.current_idx + 1, min(env.current_idx + 6, len(env.states))))
            if not candidates:
                break
            best_idx = min(candidates, key=lambda index: l2(waypoint, env.states[index]["position"]))
            if l2(waypoint, env.states[best_idx]["position"]) > env.success_radius * 1.5:
                env.collisions += 1
                break
            env.current_idx = best_idx
            env.visited_indices.append(best_idx)

        env.steps_taken += 1
        replay_end = env.current_idx >= len(env.states) - 1
        env.done = replay_end or env.steps_taken >= env.max_steps
        return env.current_state(), env.done

    def evaluate(self) -> dict[str, Any]:
        work_dir = ensure_dir(self.cfg["work_dir"])
        write_json(self.scene_coverage, work_dir / "scene_coverage.json")
        if not self.episodes:
            summary = {
                "status": "skipped_no_available_episodes",
                "episodes": 0,
                "scene_coverage": self.scene_coverage,
            }
            write_json(summary, work_dir / "metrics.json")
            return summary
        assert self.policy is not None
        details_path = work_dir / "details.jsonl"
        progress_path = work_dir / "progress.json"
        details_path.write_text("", encoding="utf-8")
        episode_metrics = []
        cycle_times = []
        execution_errors = 0
        details = []
        total_episodes = len(self.episodes)
        log_every = int(self.cfg.get("log_every", 10))
        configured_stop_threshold = self.cfg.get("stop_probability")
        stop_threshold = (
            float(configured_stop_threshold)
            if configured_stop_threshold is not None
            else float(self.policy.stop_threshold)
        )
        for ep_idx, episode in enumerate(self.episodes, start=1):
            env = RouteReplayEnv(
                episode,
                success_radius=float(self.cfg["success_radius"]),
                max_steps=int(self.cfg["max_steps"]),
            )
            state = env.reset()
            done = False
            policy_stopped = False
            termination_reason = "max_steps"
            local_errors = 0
            while not done:
                history_states = env.history_states(memory_len=int(self.cfg["model"]["memory_len"]))
                start = time.perf_counter()
                try:
                    out = self.policy.act(state, history_states, episode["visual_goal"])
                    action = out["waypoints"]
                    stop_prob = float(out["stop_prob"])
                    invalid = False
                except Exception:
                    action = None
                    stop_prob = 0.0
                    invalid = True
                ct = time.perf_counter() - start
                cycle_times.append(ct)
                if invalid or ct > float(self.cfg["budget_sec"]):
                    local_errors += 1
                    execution_errors += 1
                if not invalid and stop_prob >= stop_threshold:
                    policy_stopped = True
                    termination_reason = "policy_stop"
                    break
                state, done = self._execute_waypoint_segment(env, action)
                if done:
                    termination_reason = "replay_end"

            # RouteReplayEnv ends when it first reaches the goal. Give the policy
            # the corresponding decision at that state so SR still requires its
            # learned stop head rather than automatic success on entry.
            if done and not policy_stopped:
                history_states = env.history_states(memory_len=int(self.cfg["model"]["memory_len"]))
                start = time.perf_counter()
                try:
                    final_out = self.policy.act(state, history_states, episode["visual_goal"])
                    final_stop_prob = float(final_out["stop_prob"])
                    invalid = False
                except Exception:
                    final_stop_prob = 0.0
                    invalid = True
                ct = time.perf_counter() - start
                cycle_times.append(ct)
                if invalid or ct > float(self.cfg["budget_sec"]):
                    local_errors += 1
                    execution_errors += 1
                if not invalid and final_stop_prob >= stop_threshold:
                    policy_stopped = True
                    termination_reason = "policy_stop"
            pred_path = env.executed_path()
            ref_path = env.reference_path()
            metrics = summarize_episode(
                pred_path=pred_path,
                ref_path=ref_path,
                goal=episode["states"][-1]["position"],
                success_radius=float(self.cfg["success_radius"]),
                shortest_path_length=float(episode.get("shortest_path_length", episode.get("path_length", 1.0))),
                stopped=policy_stopped,
            )
            episode_metrics.append(metrics)
            detail = {
                "episode_id": episode["episode_id"],
                "scene_id": episode["scene_id"],
                "difficulty": episode.get("difficulty"),
                **metrics,
                "cycle_errors": local_errors,
                "num_cycles": len(pred_path),
                "stopped": policy_stopped,
                "termination_reason": termination_reason,
                "stop_threshold": stop_threshold,
            }
            details.append(detail)
            with details_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            partial = {
                "completed": ep_idx,
                "episodes": total_episodes,
                "last_episode_id": episode["episode_id"],
                "partial_metrics": aggregate_metrics(episode_metrics),
                "by_difficulty": aggregate_by_difficulty(details),
            }
            write_json(partial, progress_path)
            if ep_idx == 1 or ep_idx % log_every == 0 or ep_idx == total_episodes:
                print(f"[eval_offline] {ep_idx}/{total_episodes} episodes complete", flush=True)
        agg = aggregate_metrics(episode_metrics)
        ct_mean = sum(cycle_times) / max(len(cycle_times), 1)
        sorted_ct = sorted(cycle_times)
        p95_idx = min(int(0.95 * max(len(sorted_ct) - 1, 0)), max(len(sorted_ct) - 1, 0))
        ct_p95 = sorted_ct[p95_idx] if sorted_ct else 0.0
        eer = execution_errors / max(len(cycle_times), 1)
        summary = {
            **agg,
            "status": "complete",
            "CT_mean": ct_mean,
            "CT_p95": ct_p95,
            "EER": eer,
            "episodes": len(self.episodes),
            "by_difficulty": aggregate_by_difficulty(details),
            "scene_coverage": self.scene_coverage,
            "details": details,
        }
        write_json(summary, work_dir / "metrics.json")
        return summary
