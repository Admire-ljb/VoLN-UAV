from __future__ import annotations

import torch

from voln_uav.simulators.offline_env import RouteReplayEnv


def _episode() -> dict:
    states = [
        {"position": [0.0, 0.0, 0.0], "image": "a.png"},
        {"position": [10.0, 0.0, 0.0], "image": "b.png"},
        {"position": [20.0, 0.0, 0.0], "image": "c.png"},
    ]
    return {"episode_id": "toy", "states": states, "path_length": 20.0}


def test_invalid_action_holds_position() -> None:
    env = RouteReplayEnv(_episode(), success_radius=2.0, max_steps=3)
    start = env.current_idx

    result = env.step(None)

    assert env.current_idx == start
    assert result.info["execution_errors"] == 1
    assert result.info["collisions"] == 1


def test_far_action_holds_position() -> None:
    env = RouteReplayEnv(_episode(), success_radius=2.0, max_steps=3)
    start = env.current_idx

    result = env.step(torch.tensor([[999.0, 999.0, 999.0]], dtype=torch.float32))

    assert env.current_idx == start
    assert result.info["execution_errors"] == 0
    assert result.info["collisions"] == 1
