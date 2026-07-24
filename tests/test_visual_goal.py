import pytest

from voln_uav.benchmark.visual_goal import build_visual_goal_interface


def test_visual_goal_uses_final_three_consecutive_frames() -> None:
    route = {
        "trajectory_id": "route",
        "states": [{"image": f"frame_{index}.png"} for index in range(8)],
    }
    interface = build_visual_goal_interface(
        route,
        task_beacons=[],
        num_terminal_views=3,
        num_subgoals=2,
        num_beacons=0,
    )

    assert interface["goal_indices"] == [5, 6, 7]
    assert interface["V_goal"] == ["frame_5.png", "frame_6.png", "frame_7.png"]


def test_visual_goal_rejects_route_shorter_than_goal_window() -> None:
    route = {
        "trajectory_id": "short",
        "states": [{"image": "frame_0.png"}, {"image": "frame_1.png"}],
    }
    with pytest.raises(ValueError, match="requires 3 terminal views"):
        build_visual_goal_interface(
            route,
            task_beacons=[],
            num_terminal_views=3,
            num_subgoals=0,
            num_beacons=0,
        )
