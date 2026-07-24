from __future__ import annotations

import math

import pytest

from voln_uav.simulators.beacon_placement import TARGET_TAG, plan_route_beacons


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
    assert target_placements[0]["position"] == pytest.approx([-40.0, 20.0, -16.0])


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
            "allowed_tags": ["left_yaw", "right_yaw", "up"],
            "straight_tag": "up",
            "start_margin_steps": 0,
            "end_margin_steps": 0,
            "min_gap_steps": 1,
        },
        base_seed=7,
    )
    assert len(placements) == 3
    assert all(item["tag"] == "up" for item in placements)
    assert all(item["reason"] == "straight route" for item in placements)
