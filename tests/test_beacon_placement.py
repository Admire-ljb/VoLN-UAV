from __future__ import annotations

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
