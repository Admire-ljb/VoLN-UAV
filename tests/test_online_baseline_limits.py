from __future__ import annotations

from voln_uav.cli.eval_online_baselines import _run_targets


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

    def move_to_waypoint(self, _current: list[float], target: list[float], **_kwargs: object) -> None:
        self.client.position = list(target)


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
