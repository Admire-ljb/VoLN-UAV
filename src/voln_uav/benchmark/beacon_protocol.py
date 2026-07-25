from __future__ import annotations

import math
from typing import Any


DEFAULT_COUNT_BY_PATH_LENGTH = {
    "easy_lt_m": 300.0,
    "normal_lt_m": 450.0,
    "easy": 3,
    "normal": 4,
    "hard": 5,
}


def reference_path_length_m(states: list[dict[str, Any]]) -> float:
    """Compute the 3D reference-path length stored by the benchmark builder."""
    total = 0.0
    for previous, current in zip(states, states[1:]):
        a = previous.get("position")
        b = current.get("position")
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            raise ValueError("Every episode state must contain a position")
        if len(a) < 3 or len(b) < 3:
            raise ValueError("Episode positions must contain x, y, and z")
        total += math.dist(
            [float(value) for value in a[:3]],
            [float(value) for value in b[:3]],
        )
    return total


def task_beacon_count_for_path_length(
    path_length_m: float,
    *,
    easy_lt_m: float = 300.0,
    normal_lt_m: float = 450.0,
    easy_count: int = 3,
    normal_count: int = 4,
    hard_count: int = 5,
) -> int:
    """Map reference-path length to the fixed Easy/Normal/Hard beacon budget."""
    path_length_m = float(path_length_m)
    easy_lt_m = float(easy_lt_m)
    normal_lt_m = float(normal_lt_m)
    counts = (int(easy_count), int(normal_count), int(hard_count))
    if not math.isfinite(path_length_m) or path_length_m < 0.0:
        raise ValueError(f"Invalid reference path length: {path_length_m!r}")
    if not 0.0 < easy_lt_m < normal_lt_m:
        raise ValueError("Beacon path-length thresholds must satisfy 0 < easy < normal")
    if counts != (3, 4, 5):
        raise ValueError(
            "The release active-beacon protocol requires exactly 3/4/5 "
            f"beacons for Easy/Normal/Hard routes, got {counts}"
        )
    if path_length_m < easy_lt_m:
        return counts[0]
    if path_length_m < normal_lt_m:
        return counts[1]
    return counts[2]


def count_protocol_from_config(config: dict[str, Any] | None) -> dict[str, float | int]:
    raw = dict((config or {}).get("count_by_path_length", {}) or {})
    protocol = dict(DEFAULT_COUNT_BY_PATH_LENGTH)
    protocol.update(raw)
    return {
        "easy_lt_m": float(protocol["easy_lt_m"]),
        "normal_lt_m": float(protocol["normal_lt_m"]),
        "easy_count": int(protocol["easy"]),
        "normal_count": int(protocol["normal"]),
        "hard_count": int(protocol["hard"]),
    }


def episode_reference_path_length_m(episode: dict[str, Any]) -> float:
    stored = episode.get("path_length")
    if stored is not None:
        value = float(stored)
        if math.isfinite(value) and value >= 0.0:
            return value
    return reference_path_length_m(list(episode.get("states", [])))


def task_beacon_route_index(beacon: dict[str, Any]) -> int:
    value = beacon.get("route_index", beacon.get("visible_at"))
    if value is None:
        raise ValueError("task_beacon is missing route_index/visible_at")
    return int(value)


def validate_episode_task_beacons(
    episode: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int, float]:
    """Validate and return the immutable episode-level active-beacon list."""
    episode_id = str(episode.get("episode_id", "episode"))
    raw = episode.get("task_beacons")
    if not isinstance(raw, list):
        raise ValueError(f"Episode {episode_id!r} is missing list-valued task_beacons")

    path_length_m = episode_reference_path_length_m(episode)
    expected_count = task_beacon_count_for_path_length(
        path_length_m,
        **count_protocol_from_config(config),
    )
    if len(raw) != expected_count:
        raise ValueError(
            f"Episode {episode_id!r} has {len(raw)} task_beacons; expected "
            f"{expected_count} for reference path length {path_length_m:.3f} m"
        )

    states = list(episode.get("states", []))
    if not states:
        raise ValueError(f"Episode {episode_id!r} has no states")
    beacon_ids: set[str] = set()
    route_indices: set[int] = set()
    previous_route_index = -1
    validated: list[dict[str, Any]] = []
    for order, beacon in enumerate(raw):
        if not isinstance(beacon, dict):
            raise ValueError(f"Episode {episode_id!r} task_beacons[{order}] is not an object")
        beacon_id = str(beacon.get("beacon_id", "")).strip()
        if not beacon_id:
            raise ValueError(f"Episode {episode_id!r} task_beacons[{order}] has no beacon_id")
        if beacon_id in beacon_ids:
            raise ValueError(f"Episode {episode_id!r} has duplicate task beacon {beacon_id!r}")
        route_index = task_beacon_route_index(beacon)
        if route_index < 0 or route_index >= len(states):
            raise ValueError(
                f"Episode {episode_id!r} task beacon {beacon_id!r} has out-of-range "
                f"route index {route_index}"
            )
        if route_index in route_indices:
            raise ValueError(
                f"Episode {episode_id!r} has multiple task beacons at route index "
                f"{route_index}"
            )
        if route_index <= previous_route_index:
            raise ValueError(
                f"Episode {episode_id!r} task_beacons must be ordered by route index"
            )
        semantic_type = str(beacon.get("semantic_type", "")).strip()
        if not semantic_type:
            raise ValueError(
                f"Episode {episode_id!r} task beacon {beacon_id!r} has no semantic_type"
            )
        beacon_ids.add(beacon_id)
        route_indices.add(route_index)
        previous_route_index = route_index
        validated.append(beacon)
    return validated, expected_count, path_length_m
