from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.models.policy import VoLNPolicy
from voln_uav.simulators.offline_env import RouteReplayEnv


class DAggerCollector:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        benchmark_root = Path(config["benchmark_root"])
        self.episodes = read_jsonl(benchmark_root / config["episodes_file"])
        self.output_dir = ensure_dir(config["output_dir"])
        self.policy = VoLNPolicy(
            config=config,
            semantic_bank_path=benchmark_root / config["semantic_bank"],
            adapter_ckpt=config["adapter_ckpt"],
            planner_ckpt=config["planner_ckpt"],
            device=device,
        )
        self.rng = random.Random(int(config["seed"]))

    def collect(self) -> dict[str, Any]:
        beta = float(self.cfg["beta"])
        horizon = int(self.cfg["model"]["horizon"])
        collected = []
        episode_logs = []
        for round_idx in range(int(self.cfg["collect_rounds"])):
            for episode in self.episodes:
                env = RouteReplayEnv(episode, success_radius=float(self.cfg["success_radius"]), max_steps=int(self.cfg["max_steps"]))
                state = env.reset()
                done = False
                while not done:
                    history_states = env.history_states(memory_len=int(self.cfg["model"]["memory_len"]))
                    policy_out = self.policy.act(state, history_states, episode["visual_goal"])
                    teacher_waypoints = env.expert_waypoints(horizon=horizon)
                    use_teacher = self.rng.random() < beta
                    action = teacher_waypoints if use_teacher else policy_out["waypoints"]
                    step_result = env.step(action)
                    collected.append(
                        {
                            "episode_id": episode["episode_id"],
                            "step": env.current_idx,
                            "visited_state_image": state["image"],
                            "expert_future_waypoints": teacher_waypoints.tolist(),
                            "used_teacher": use_teacher,
                            "semantic_names": policy_out["semantic_names"],
                        }
                    )
                    state = step_result.state
                    done = step_result.done
                episode_logs.append(
                    {
                        "episode_id": episode["episode_id"],
                        "success": step_result.info["success"],
                        "oracle_success": step_result.info["oracle_success"],
                        "final_distance": step_result.info["final_distance"],
                    }
                )
        write_jsonl(collected, self.output_dir / "dagger_records.jsonl")
        summary = {
            "num_records": len(collected),
            "num_episodes": len(episode_logs),
            "episodes": episode_logs,
        }
        write_json(summary, self.output_dir / "summary.json")
        return summary
