import torch

from voln_uav.evaluation.closed_loop import ClosedLoopEvaluator
from voln_uav.simulators.offline_env import RouteReplayEnv


def test_full_waypoint_segment_counts_as_one_decision():
    states = [
        {"position": [float(index), 0.0, 0.0], "image": f"{index}.png"}
        for index in range(10)
    ]
    env = RouteReplayEnv(
        {"episode_id": "segment", "states": states, "path_length": 9.0},
        success_radius=1.0,
        max_steps=128,
    )
    waypoints = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

    state, done = ClosedLoopEvaluator._execute_waypoint_segment(env, waypoints)

    assert state["position"] == [6.0, 0.0, 0.0]
    assert env.steps_taken == 1
    assert env.visited_indices == [0, 1, 3, 6]
    assert done is False
