from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from voln_uav.common.geometry import l2


@dataclass
class StepResult:
    state: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class RouteReplayEnv:
    def __init__(self, episode: dict[str, Any], success_radius: float = 20.0, max_steps: int | None = None) -> None:
        self.episode = episode
        self.states = episode["states"]
        self.goal = self.states[-1]["position"]
        self.success_radius = float(success_radius)
        self.max_steps = max_steps or len(self.states)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.current_idx = 0
        self.steps_taken = 0
        self.done = False
        self.execution_errors = 0
        self.collisions = 0
        self.visited_indices = [0]
        return self.current_state()

    def current_state(self) -> dict[str, Any]:
        return self.states[self.current_idx]

    def history_states(self, memory_len: int) -> list[dict[str, Any]]:
        start = max(0, self.current_idx - memory_len + 1)
        hist = self.states[start : self.current_idx + 1]
        while len(hist) < memory_len:
            hist.insert(0, hist[0])
        return hist

    def expert_waypoints(self, horizon: int) -> torch.Tensor:
        pts = []
        for off in range(1, horizon + 1):
            idx = min(self.current_idx + off, len(self.states) - 1)
            pts.append(self.states[idx]["position"])
        return torch.tensor(pts, dtype=torch.float32)

    def _oracle_success(self) -> bool:
        return any(l2(self.states[idx]["position"], self.goal) <= self.success_radius for idx in self.visited_indices)

    def step(self, action_waypoints: torch.Tensor | None) -> StepResult:
        if self.done:
            return StepResult(state=self.current_state(), reward=0.0, done=True, info={"reason": "already_done"})

        if action_waypoints is None or not torch.isfinite(action_waypoints).all():
            self.execution_errors += 1
            self.collisions += 1
            next_idx = min(self.current_idx + 1, len(self.states) - 1)
        else:
            target = action_waypoints[0].detach().cpu().tolist()
            future_candidates = list(range(self.current_idx + 1, min(self.current_idx + 6, len(self.states))))
            if not future_candidates:
                future_candidates = [self.current_idx]
            best_idx = min(future_candidates, key=lambda i: l2(target, self.states[i]["position"]))
            best_dist = l2(target, self.states[best_idx]["position"])
            if best_dist > self.success_radius * 1.5:
                self.collisions += 1
                next_idx = min(self.current_idx + 1, len(self.states) - 1)
            else:
                next_idx = best_idx

        self.current_idx = next_idx
        self.steps_taken += 1
        self.visited_indices.append(self.current_idx)
        state = self.current_state()
        final_dist = l2(state["position"], self.goal)
        success = final_dist <= self.success_radius
        timeout = self.steps_taken >= self.max_steps or self.current_idx >= len(self.states) - 1
        self.done = success or timeout
        reward = 1.0 if success else -final_dist / max(self.episode.get("path_length", 1.0), 1.0)
        info = {
            "success": success,
            "oracle_success": self._oracle_success(),
            "final_distance": final_dist,
            "current_idx": self.current_idx,
            "collisions": self.collisions,
            "execution_errors": self.execution_errors,
        }
        return StepResult(state=state, reward=reward, done=self.done, info=info)

    def executed_path(self) -> list[list[float]]:
        return [self.states[idx]["position"] for idx in self.visited_indices]

    def reference_path(self) -> list[list[float]]:
        return [st["position"] for st in self.states]
