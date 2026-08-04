from __future__ import annotations

import math

import pytest

from voln_uav.simulators.beacon_placement import (
    TARGET_TAG,
    plan_route_beacons,
    route_motion_features,
)


def _episode() -> dict:
    states = []
    points = [
        (0.0, 0.0, -10.0, 0.0),
        (10.0, 0.0, -12.0, 0.0),
        (20.0, 0.0, -14.0, 0.0),
        (20.0, 10.0, -14.0, 1.5708),
        (20.0, 20.0, -14.0, 1.5708),
        (10.0, 20.0, -16.0, 3.1416),
        (0.0, 20.0, -16.0, 3.1416),
    ]
    for x, y, z, yaw in points:
        states.append({"position": [x, y, z], "yaw": yaw})
    return {"episode_id": "unit_route", "difficulty": "Normal", "states": states}


def test_plan_route_beacons_is_reproducible_and_adds_target() -> None:
    cfg = {"count": 3, "include_target": True, "random_seed": 9, "min_gap_steps": 1}
    first = plan_route_beacons(_episode(), cfg, base_seed=7)
    second = plan_route_beacons(_episode(), cfg, base_seed=7)

    assert first == second
    assert len(first) == 4
    assert first[-1]["tag"] == TARGET_TAG
    assert first[-1]["kind"] == "target"
    assert first[-1]["position"] == [0.0, 20.0, -16.0]
    assert all("position" in item and len(item["position"]) == 3 for item in first)
    assert all(item["kind"] in {"route_beacon", "target"} for item in first)


def test_plan_route_beacons_respects_direction_icon_tags() -> None:
    placements = plan_route_beacons(
        _episode(),
        {
            "count": 3,
            "include_target": False,
            "allowed_tags": ["left_yaw", "right_yaw", "up"],
            "fallback_tags": ["left_yaw", "right_yaw", "up"],
            "min_gap_steps": 1,
        },
        base_seed=7,
    )
    assert placements
    assert all(item["tag"] in {"left_yaw", "right_yaw", "up"} for item in placements)


def test_route_generated_beacons_prioritize_turning_points() -> None:
    placements = plan_route_beacons(
        _episode(),
        {
            "count": 3,
            "include_target": False,
            "allowed_tags": [
                "left_yaw", "left_turn", "left90",
                "right_yaw", "right_turn", "right90", "up",
            ],
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )

    turning = [
        item for item in placements if item["tag"].startswith(("left", "right"))
    ]
    assert len(turning) >= 2
    assert all(item["tag"] == "right90" for item in turning)
    assert all(
        item["index"] <= item["turn_pivot_index"] <= item["turn_event_end_index"]
        for item in turning
    )
    assert all(item["turn_warning_distance_m"] == 30.0 for item in turning)
    assert all(
        item["turn_direction_reference_index"] == max(item["index"] - 1, 0)
        for item in turning
    )
    assert all(item["forward_m"] == 0.0 for item in turning)


    states = _episode()["states"]
    for item in turning:
        turn_position = states[item["turn_pivot_index"]]["position"]
        expected_yaw = math.radians(item["turn_direction_yaw_deg"] + 90.0)
        assert item["yaw_rad"] == pytest.approx(expected_yaw)
        anchor_position = item["route_anchor_position"]
        horizontal_distance = math.hypot(
            anchor_position[0] - turn_position[0],
            anchor_position[1] - turn_position[1],
        )
        assert horizontal_distance == pytest.approx(30.0)


def test_turn_warning_uses_exact_distance_with_irregular_sampling() -> None:
    states = [
        {"position": [0.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [3.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [11.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 5.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 10.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 20.0, -10.0], "yaw": 0.0},
    ]
    placements = plan_route_beacons(
        {"episode_id": "irregular_right_turn", "difficulty": "Easy", "states": states},
        {
            "count": 1,
            "include_target": False,
            "allowed_tags": ["right_yaw", "right_turn", "right90"],
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
            "turn_warning_distance_m": 15.0,
            "preset_override": {"lateral": [0.0, 0.0], "vertical_ned": 0.0},
        },
        base_seed=7,
    )

    assert len(placements) == 1
    beacon = placements[0]
    assert beacon["tag"] == "right90"
    assert beacon["index"] == 3
    assert beacon["turn_event_end_index"] == 3
    assert beacon["ref_index"] == 3
    assert beacon["route_anchor_position"] == pytest.approx([35.0, 0.0, -10.0])


@pytest.mark.parametrize(
    ("final_y", "expected_tag"),
    [(10.0, "right90"), (-10.0, "left90")],
)
def test_turn_direction_follows_reference_geometry(
    final_y: float,
    expected_tag: str,
) -> None:
    states = [
        {"position": [0.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [10.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, final_y, -10.0], "yaw": 0.0},
    ]
    placements = plan_route_beacons(
        {"episode_id": expected_tag, "difficulty": "Easy", "states": states},
        {
            "count": 1,
            "include_target": False,
            "allowed_tags": ["left90", "right90"],
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )

    assert len(placements) == 1
    assert placements[0]["tag"] == expected_tag
    assert placements[0]["route_anchor_position"] == pytest.approx([50.0, 0.0, -10.0])


def test_turn_warning_distance_ignores_altitude_change() -> None:
    states = [
        {"position": [0.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [10.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 5.0, 0.0], "yaw": 0.0},
    ]
    placements = plan_route_beacons(
        {"episode_id": "vertical_right_turn", "difficulty": "Easy", "states": states},
        {
            "count": 1,
            "include_target": False,
            "allowed_tags": ["right90"],
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "turn_warning_distance_m": 15.0,
            "preset_override": {"lateral": [0.0, 0.0], "vertical_ned": 0.0},
        },
        base_seed=7,
    )

    assert len(placements) == 1
    beacon = placements[0]
    turn_position = states[beacon["index"]]["position"]
    anchor = beacon["route_anchor_position"]
    assert math.hypot(
        anchor[0] - turn_position[0],
        anchor[1] - turn_position[1],
    ) == pytest.approx(15.0)
    assert anchor == pytest.approx([35.0, 0.0, -10.0])


def test_plan_route_beacons_uses_scene_ground_height() -> None:
    episode = {**_episode(), "scene_id": "BrushifyUrban"}
    placements = plan_route_beacons(
        episode,
        {
            "count": 2,
            "include_target": True,
            "ground_z_ned_by_scene": {"BrushifyUrban": -1.0},
            "target_distance_m": 40.0,
            "target_vertical_ned_m": 0.0,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )
    route_placements = [item for item in placements if item["tag"] != "target_people"]
    target_placements = [item for item in placements if item["tag"] == "target_people"]
    assert route_placements
    assert all(item["position"][2] == -1.0 for item in route_placements)
    assert len(target_placements) == 1
    assert target_placements[0]["position"] == pytest.approx([-40.0, 20.0, -16.0], abs=1e-3)


def test_plan_route_beacons_uses_scene_vertical_offset() -> None:
    episode = {**_episode(), "scene_id": "BrushifyCountryRoads"}
    placements = plan_route_beacons(
        episode,
        {
            "count": 2,
            "include_target": False,
            "vertical_ned_offset_by_scene": {"BrushifyCountryRoads": 12.0},
            "min_gap_steps": 1,
        },
        base_seed=7,
    )
    assert placements
    assert all(item["vertical_ned_m"] == 12.0 for item in placements)
    assert all(
        item["position"][2] == pytest.approx(item["route_anchor_position"][2] + 12.0)
        for item in placements
    )


def test_straight_route_ignores_recorded_camera_yaw_and_uses_straight_icons() -> None:
    states = [
        {"position": [0.0, 0.0, -10.0], "yaw": math.radians(-150.0)},
        {"position": [5.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [10.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [15.0, 0.0, -10.0], "yaw": 0.0},
        {"position": [20.0, 0.0, -10.0], "yaw": 0.0},
    ]
    placements = plan_route_beacons(
        {"episode_id": "straight", "difficulty": "Easy", "states": states},
        {
            "count": 3,
            "include_target": False,
            "allowed_tags": ["left_yaw", "right_yaw", "here"],
            "straight_tag": "here",
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )
    assert len(placements) == 1
    assert placements[0]["index"] == len(states) - 1
    assert placements[0]["ref_index"] == len(states) - 1
    assert placements[0]["forward_m"] == 0.0
    assert placements[0]["tag"] == "here"
    assert placements[0]["reason"] == "end of straight route"


def test_gradual_terminal_turn_is_detected_from_cumulative_heading_change() -> None:
    headings = [0.0, -10.0, -20.0, -30.0, -30.0, -30.0]
    points = [[0.0, 0.0, -10.0]]
    for heading in headings:
        angle = math.radians(heading)
        previous = points[-1]
        points.append(
            [
                previous[0] + 10.0 * math.cos(angle),
                previous[1] + 10.0 * math.sin(angle),
                previous[2],
            ]
        )
    states = [{"position": point, "yaw": 0.0} for point in points]
    placements = plan_route_beacons(
        {"episode_id": "gradual_turn", "difficulty": "Easy", "states": states},
        {
            "count": 3,
            "include_target": False,
            "allowed_tags": ["left_yaw", "left_turn", "left90", "here"],
            "straight_tag": "here",
            "start_margin_steps": 0,
            "end_margin_steps": 2,
            "min_gap_steps": 2,
            "turn_window_steps": 3,
        },
        base_seed=7,
    )

    assert placements
    assert all(item["tag"].startswith("left") for item in placements)
    assert all("over" in item["reason"] for item in placements)


def test_terminal_continuous_turn_uses_total_yaw_for_sign_class() -> None:
    headings = [0.0, -20.0, -40.0, -70.0, -70.0]
    points = [[0.0, 0.0, -10.0]]
    for heading in headings:
        angle = math.radians(heading)
        previous = points[-1]
        points.append(
            [
                previous[0] + 10.0 * math.cos(angle),
                previous[1] + 10.0 * math.sin(angle),
                previous[2],
            ]
        )
    states = [{"position": point, "yaw": 0.0} for point in points]
    placements = plan_route_beacons(
        {"episode_id": "terminal_total_turn", "difficulty": "Easy", "states": states},
        {
            "count": 1,
            "include_target": False,
            "allowed_tags": ["left_yaw", "left_turn", "left90"],
            "start_margin_steps": 0,
            "end_margin_steps": 2,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )

    assert len(placements) == 1
    assert placements[0]["tag"] == "left90"
    assert placements[0]["turn_total_yaw_deg"] == pytest.approx(-70.0)
    assert "over route indices" in placements[0]["reason"]


def test_terminal_reversing_turn_uses_signed_net_yaw_for_one_sign() -> None:
    # Segment headings turn left 30, correct right 40, then turn left 60.
    # The terminal fragment therefore has net yaw -50 degrees and must emit
    # one left_turn sign rather than three contradictory signs.
    headings = [0.0, -30.0, 10.0, -50.0, -50.0]
    points = [[0.0, 0.0, -10.0]]
    for heading in headings:
        angle = math.radians(heading)
        previous = points[-1]
        points.append(
            [
                previous[0] + 10.0 * math.cos(angle),
                previous[1] + 10.0 * math.sin(angle),
                previous[2],
            ]
        )
    states = [{"position": point, "yaw": 0.0} for point in points]
    placements = plan_route_beacons(
        {"episode_id": "terminal_reversing_turn", "difficulty": "Easy", "states": states},
        {
            "count": 3,
            "include_target": False,
            "allowed_tags": [
                "left_yaw", "left_turn", "left90",
                "right_yaw", "right_turn", "right90",
            ],
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
            "terminal_reverse_turn_window_steps": 12,
            "terminal_reverse_turn_max_straight_gap_m": 30.0,
        },
        base_seed=7,
    )

    assert len(placements) == 1
    assert placements[0]["tag"] == "left_turn"
    assert placements[0]["turn_total_yaw_deg"] == pytest.approx(-50.0)
    assert "net over reversing terminal" in placements[0]["reason"]


def test_online_placement_uses_exact_episode_task_beacons() -> None:
    episode = {
        **_episode(),
        "path_length": 350.0,
        "task_beacons": [
            {
                "beacon_id": f"task_{order}",
                "route_index": route_index,
                "visible_at": route_index,
                "semantic_type": semantic_type,
            }
            for order, (route_index, semantic_type) in enumerate(
                [
                    (1, "road-sign"),
                    (2, "turn-right"),
                    (4, "road-sign"),
                    (5, "turn-left"),
                ]
            )
        ],
    }
    placements = plan_route_beacons(
        episode,
        {
            "source": "episode_task_beacons",
            "include_target": True,
            "distance_jitter_m": 0.0,
        },
        base_seed=7,
    )

    active = [item for item in placements if item["kind"] == "route_beacon"]
    assert [item["task_beacon_id"] for item in active] == [
        "task_0",
        "task_1",
        "task_2",
        "task_3",
    ]
    assert [item["index"] for item in active] == [1, 2, 4, 5]
    assert all(item["source"] == "episode_task_beacons" for item in active)
    assert len([item for item in placements if item["kind"] == "target"]) == 1


def test_online_placement_rejects_missing_episode_task_beacons() -> None:
    with pytest.raises(ValueError, match="missing list-valued task_beacons"):
        plan_route_beacons(
            {**_episode(), "path_length": 250.0},
            {"source": "episode_task_beacons"},
            base_seed=7,
        )



def test_target_is_placed_in_front_of_the_recorded_final_view() -> None:
    episode = {
        "episode_id": "target_final_view",
        "difficulty": "Easy",
        "states": [
            {"position": [0.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [10.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [20.0, 0.0, -5.0], "yaw": math.pi / 2.0},
        ],
    }
    placements = plan_route_beacons(
        episode,
        {
            "count": 0,
            "include_target": True,
            "target_distance_m": 10.0,
            "target_lateral_m": 0.0,
            "target_vertical_ned_m": 0.0,
        },
        base_seed=7,
    )
    target = next(item for item in placements if item["kind"] == "target")
    assert target["position"] == pytest.approx([20.0, 10.0, -5.0])
    assert target["target_direction_source"] == "recorded_final_yaw"
    assert target["target_direction_yaw_deg"] == pytest.approx(90.0)
    assert target["yaw_rad"] == pytest.approx(3.0 * math.pi / 2.0)


def test_ned_yaw_sign_maps_positive_to_right_and_negative_to_left() -> None:
    right = route_motion_features(
        [
            {"position": [0.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [10.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [10.0, 10.0, -5.0], "yaw": math.pi / 2.0},
        ]
    )
    left = route_motion_features(
        [
            {"position": [0.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [10.0, 0.0, -5.0], "yaw": 0.0},
            {"position": [10.0, -10.0, -5.0], "yaw": -math.pi / 2.0},
        ]
    )
    assert right[1]["d_yaw_deg"] == pytest.approx(90.0)
    assert right[1]["tag"] == "right90"
    assert left[1]["d_yaw_deg"] == pytest.approx(-90.0)
    assert left[1]["tag"] == "left90"
