from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from voln_uav.common.io import read_jsonl, write_json
from voln_uav.evaluation.metrics import aggregate_metrics, summarize_episode
from voln_uav.models.policy import VoLNPolicy
from voln_uav.simulators.offline_env import RouteReplayEnv


class ClosedLoopEvaluator:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = device
        self.benchmark_root = Path(config["benchmark_root"])
        self.episodes = read_jsonl(self.benchmark_root / config["episodes_file"])
        self.policy = VoLNPolicy(
            config=config,
            semantic_bank_path=self.benchmark_root / config["semantic_bank"],
            adapter_ckpt=config["adapter_ckpt"],
            planner_ckpt=config["planner_ckpt"],
            device=device,
        )

    def evaluate(self) -> dict[str, Any]:
        episode_metrics = []
        cycle_times = []
        execution_errors = 0
        details = []
        for episode in self.episodes:
            env = RouteReplayEnv(
                episode,
                success_radius=float(self.cfg["success_radius"]),
                max_steps=int(self.cfg["max_steps"]),
            )
            state = env.reset()
            done = False
            local_errors = 0
            while not done:
                history_states = env.history_states(memory_len=int(self.cfg["model"]["memory_len"]))
                start = time.perf_counter()
                try:
                    out = self.policy.act(state, history_states, episode["visual_goal"])
                    action = out["waypoints"]
                    invalid = False
                except Exception:
                    action = None
                    invalid = True
                ct = time.perf_counter() - start
                cycle_times.append(ct)
                if invalid or ct > float(self.cfg["budget_sec"]):
                    local_errors += 1
                    execution_errors += 1
                step_result = env.step(action)
                state = step_result.state
                done = step_result.done
            pred_path = env.executed_path()
            ref_path = env.reference_path()
            metrics = summarize_episode(
                pred_path=pred_path,
                ref_path=ref_path,
                goal=episode["states"][-1]["position"],
                success_radius=float(self.cfg["success_radius"]),
                shortest_path_length=float(episode.get("shortest_path_length", episode.get("path_length", 1.0))),
            )
            episode_metrics.append(metrics)
            details.append(
                {
                    "episode_id": episode["episode_id"],
                    **metrics,
                    "cycle_errors": local_errors,
                    "num_cycles": len(pred_path),
                }
            )
        agg = aggregate_metrics(episode_metrics)
        ct_mean = sum(cycle_times) / max(len(cycle_times), 1)
        sorted_ct = sorted(cycle_times)
        p95_idx = min(int(0.95 * max(len(sorted_ct) - 1, 0)), max(len(sorted_ct) - 1, 0))
        ct_p95 = sorted_ct[p95_idx] if sorted_ct else 0.0
        eer = execution_errors / max(len(cycle_times), 1)
        summary = {
            **agg,
            "CT_mean": ct_mean,
            "CT_p95": ct_p95,
            "EER": eer,
            "episodes": len(self.episodes),
            "details": details,
        }
        write_json(summary, Path(self.cfg["work_dir"]) / "metrics.json")
        return summary
