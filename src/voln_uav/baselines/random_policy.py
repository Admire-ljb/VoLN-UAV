from __future__ import annotations

import random
from typing import Any

import torch


class RandomPolicy:
    def __init__(self, horizon: int = 5, step_scale: float = 2.0, seed: int = 7) -> None:
        self.horizon = horizon
        self.step_scale = step_scale
        self.rng = random.Random(seed)

    def act(self, state: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        base = state["position"]
        pts = []
        x, y, z = base
        for _ in range(self.horizon):
            x += self.rng.uniform(0.5, self.step_scale)
            y += self.rng.uniform(-self.step_scale, self.step_scale)
            z += self.rng.uniform(-0.5, 0.8)
            pts.append([x, y, z])
        return {
            "waypoints": torch.tensor(pts, dtype=torch.float32),
            "anchor": torch.tensor(pts[-1], dtype=torch.float32),
            "stop_prob": 0.0,
            "semantic_names": [],
        }
