from __future__ import annotations

import math
import random
from typing import Any

import torch


class RandomPolicy:
    def __init__(
        self,
        horizon: int = 8,
        step_scale: float = 10.0,
        vertical_step_scale: float = 0.5,
        stop_probability: float = 1.0 / 128.0,
        seed: int = 7,
    ) -> None:
        self.horizon = horizon
        self.step_scale = step_scale
        self.vertical_step_scale = vertical_step_scale
        self.stop_probability = stop_probability
        self.rng = random.Random(seed)

    def act(self, state: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        base = state["position"]
        pts = []
        x, y, z = base
        for _ in range(self.horizon):
            heading = self.rng.uniform(-3.141592653589793, 3.141592653589793)
            distance = self.rng.uniform(0.0, self.step_scale)
            x += distance * math.cos(heading)
            y += distance * math.sin(heading)
            z += self.rng.uniform(-self.vertical_step_scale, self.vertical_step_scale)
            pts.append([x, y, z])
        return {
            "waypoints": torch.tensor(pts, dtype=torch.float32),
            "anchor": torch.tensor(pts[-1], dtype=torch.float32),
            "stop_prob": float(self.rng.random() < self.stop_probability),
            "semantic_names": [],
        }
