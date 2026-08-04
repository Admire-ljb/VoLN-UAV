from __future__ import annotations

import math

from voln_uav.cli.eval_online_baselines import (
    _interpolate_reference_state,
    _online_shortest_path_length,
    _resample_path,
    _run_targets,
)
from voln_uav.simulators.airsim_env import split_waypoints_by_heading


class _Position:
    def __init__(self, xyz: list[float]) -> None:
        self.x_val, self.y_val, self.z_val = xyz


class _Kinematics:
    def __init__(self, xyz: list[float]) -> None:
        self.position = _Position(xyz)


class _State:
    def __init__(self, xyz: list[float]) -> None:
        self.kinematics_estimated = _Kinematics(xyz)


class _Collision:
    has_collided = False


class _Client:
    def __init__(self) -> None:
        self.position = [0.0, 0.0, 0.0]

    def getMultirotorState(self) -> _State:
        return _State(self.position)

    def simGetCollisionInfo(self) -> _Collision:
        return _Collision()


class _Env:
    def __init__(self) -> None:
        self.client = _Client()
        self.hover_calls = 0
        self.reference_pose_calls: list[dict[str, object]] = []

    def move_to_waypoint(self, _current: list[float], target: list[float], **_kwargs: object) -> None:
        self.client.position = list(target)

    def set_reference_pose(self, state: dict[str, object], **_kwargs: object) -> None:
        self.reference_pose_calls.append(state)
        self.client.position = list(state["position"])

    def move_on_path(self, targets: list[list[float]], **_kwargs: object) -> dict[str, object]:
        start = list(self.client.position)
        if targets:
            self.client.position = list(targets[-1])
        return {
            "telemetry_path": [start, *[list(target) for target in targets]],
            "collision_samples": 0,
        }

    def hover(self, _position: list[float] | None = None) -> bool:
        self.hover_calls += 1
        return True


def test_reference_baseline_respects_paper_decision_limit() -> None:
    env = _Env()
    result = _run_targets(
        env=env,
        episode={},
        targets=[[float(i), 0.0, 0.0] for i in range(1, 6)],
        control_mode="teleport",
        timeout_sec=100.0,
        path_length_limit_m=100.0,
        goal=[5.0, 0.0, 0.0],
        success_radius=0.5,
        stationary_timeout_sec=0.0,
        stationary_radius_m=0.1,
        max_teleport_step_m=10.0,
        max_teleport_vertical_step_m=10.0,
        teleport_keep_initial_height=False,
        paper_protocol=True,
        stop_at_end=True,
        max_decisions=3,
    )

    executed, _cycle_times, _collisions, _elapsed, reason, _length, _stationary, stopped = result
    assert len(executed) == 4
    assert executed[-1] == [3.0, 0.0, 0.0]
    assert reason == "max_steps"
    assert stopped is False


def test_online_replay_accepts_released_shortest_path_without_provenance() -> None:
    assert (
        _online_shortest_path_length(
            {"shortest_path_length": 7.5},
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )
        == 7.5
    )


def test_online_replay_falls_back_to_route_length() -> None:
    assert (
        _online_shortest_path_length(
            {},
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        )
        == 5.0
    )


def test_resample_path_matches_requested_count_and_endpoints() -> None:
    sampled = _resample_path(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        5,
    )
    assert len(sampled) == 5
    assert sampled[0] == [0.0, 0.0, 0.0]
    assert sampled[-1] == [2.0, 2.0, 0.0]
    assert sampled[2] == [2.0, 0.0, 0.0]



def test_reference_pose_interpolation_smooths_position_and_shortest_yaw_arc() -> None:
    state = _interpolate_reference_state(
        {"position": [0.0, 0.0, 0.0], "yaw": math.radians(170.0)},
        {"position": [6.0, 3.0, -3.0], "yaw": math.radians(-170.0)},
        0.5,
    )
    assert state["position"] == [3.0, 1.5, -1.5]
    assert abs(abs(float(state["yaw"])) - math.pi) < 1e-9

def test_split_waypoints_by_heading_creates_local_yaw_segments() -> None:
    chunks = split_waypoints_by_heading(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        turn_threshold_deg=15.0,
    )
    assert len(chunks) == 2
    assert chunks[0][-1] == [10.0, 0.0, 0.0]
    assert chunks[1][0] == [10.0, 0.0, 0.0]


def test_completed_route_enters_hover() -> None:
    env = _Env()
    result = _run_targets(
        env=env,
        episode={},
        targets=[[1.0, 0.0, 0.0]],
        control_mode="move_to_position",
        timeout_sec=100.0,
        path_length_limit_m=100.0,
        goal=[1.0, 0.0, 0.0],
        success_radius=0.5,
        stationary_timeout_sec=0.0,
        stationary_radius_m=0.1,
        max_teleport_step_m=10.0,
        max_teleport_vertical_step_m=10.0,
        teleport_keep_initial_height=False,
        paper_protocol=True,
        stop_at_end=True,
    )
    assert result[4] == "policy_stop"
    assert env.hover_calls == 1


def test_continuous_reference_path_uses_one_motion_command_and_hovers() -> None:
    env = _Env()
    result = _run_targets(
        env=env,
        episode={},
        targets=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        control_mode="move_on_path",
        timeout_sec=100.0,
        path_length_limit_m=100.0,
        goal=[3.0, 0.0, 0.0],
        success_radius=0.5,
        stationary_timeout_sec=0.0,
        stationary_radius_m=0.1,
        max_teleport_step_m=10.0,
        max_teleport_vertical_step_m=10.0,
        teleport_keep_initial_height=False,
        paper_protocol=True,
        stop_at_end=True,
    )
    assert result[0][-1] == [3.0, 0.0, 0.0]
    assert result[4] == "policy_stop"
    assert env.hover_calls == 1
    assert len(result[1]) == 1


def test_reference_setpose_replay_uses_recorded_states_without_step_clipping() -> None:
    env = _Env()
    states = [
        {
            "position": [20.0, 0.0, -4.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "position": [45.0, 10.0, -9.0],
            "orientation": [0.0, 0.0, 0.7071068, 0.7071068],
        },
    ]
    result = _run_targets(
        env=env,
        episode={},
        targets=[list(state["position"]) for state in states],
        target_states=states,
        control_mode="setpose_replay",
        reference_pose_interval_sec=0.0,
        timeout_sec=100.0,
        path_length_limit_m=100.0,
        goal=[45.0, 10.0, -9.0],
        success_radius=0.5,
        stationary_timeout_sec=0.0,
        stationary_radius_m=0.1,
        max_teleport_step_m=10.0,
        max_teleport_vertical_step_m=0.5,
        teleport_keep_initial_height=False,
        paper_protocol=True,
        stop_at_end=True,
    )

    assert env.reference_pose_calls == states
    assert result[0][-1] == [45.0, 10.0, -9.0]
    assert result[4] == "policy_stop"
    assert env.hover_calls == 1
