from __future__ import annotations

import pytest

from voln_uav.benchmark.beacon_protocol import (
    task_beacon_count_for_path_length,
    validate_episode_task_beacons,
)


@pytest.mark.parametrize(
    ("path_length_m", "expected"),
    [
        (0.0, 3),
        (299.999, 3),
        (300.0, 4),
        (449.999, 4),
        (450.0, 5),
        (900.0, 5),
    ],
)
def test_task_beacon_count_is_fixed_by_reference_path_length(
    path_length_m: float,
    expected: int,
) -> None:
    assert task_beacon_count_for_path_length(path_length_m) == expected


def _episode(path_length_m: float, count: int) -> dict:
    states = [
        {"position": [float(index), 0.0, -10.0], "yaw": 0.0}
        for index in range(8)
    ]
    return {
        "episode_id": "protocol_episode",
        "path_length": path_length_m,
        "states": states,
        "task_beacons": [
            {
                "beacon_id": f"task_{index}",
                "route_index": index + 1,
                "visible_at": index + 1,
                "semantic_type": "road-sign",
            }
            for index in range(count)
        ],
    }


def test_episode_task_beacons_require_exact_length_dependent_count() -> None:
    episode = _episode(350.0, 4)
    beacons, expected_count, path_length_m = validate_episode_task_beacons(episode)

    assert beacons == episode["task_beacons"]
    assert expected_count == 4
    assert path_length_m == 350.0


def test_episode_task_beacons_reject_method_specific_count() -> None:
    with pytest.raises(ValueError, match="has 3 task_beacons; expected 4"):
        validate_episode_task_beacons(_episode(350.0, 3))


def test_episode_task_beacons_are_required() -> None:
    episode = _episode(250.0, 3)
    del episode["task_beacons"]

    with pytest.raises(ValueError, match="missing list-valued task_beacons"):
        validate_episode_task_beacons(episode)
