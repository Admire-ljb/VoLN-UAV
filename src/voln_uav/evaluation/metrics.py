from __future__ import annotations

import math
from typing import Any, Sequence

from voln_uav.common.geometry import l2, path_length


Vec3 = Sequence[float]
METRIC_KEYS = ("NE", "SR", "OSR", "nDTW", "SPL")
DIFFICULTY_ORDER = ("Easy", "Normal", "Hard")


def validated_shortest_path_length(episode: dict[str, Any]) -> float:
    value = episode.get("shortest_path_length")
    provenance = episode.get("shortest_path_provenance")
    if value is None or not isinstance(provenance, dict):
        raise ValueError(
            f"Episode {episode.get('episode_id', '<unknown>')} lacks an independently "
            "computed shortest path and provenance required for SPL"
        )
    method = str(provenance.get("method", "")).strip().casefold()
    if not method or method in {"reference_path", "reference_path_length", "demonstration"}:
        raise ValueError(
            f"Episode {episode.get('episode_id', '<unknown>')} uses invalid SPL "
            f"shortest-path provenance: {method or '<missing>'}"
        )
    length = float(value)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("shortest_path_length must be finite and positive")
    return length



def navigation_error(pred_path: Sequence[Vec3], goal: Vec3) -> float:
    if not pred_path:
        return float("inf")
    return l2(pred_path[-1], goal)



def success(pred_path: Sequence[Vec3], goal: Vec3, radius: float, stopped: bool = True) -> bool:
    """SR requires an explicit stop inside the three-dimensional goal region."""
    return bool(stopped and pred_path) and l2(pred_path[-1], goal) <= radius



def oracle_success(pred_path: Sequence[Vec3], goal: Vec3, radius: float) -> bool:
    if any(l2(point, goal) <= radius for point in pred_path):
        return True
    return any(
        _point_to_segment_distance(goal, pred_path[index - 1], pred_path[index]) <= radius
        for index in range(1, len(pred_path))
    )


def _point_to_segment_distance(point: Vec3, start: Vec3, end: Vec3) -> float:
    segment = [float(end[index]) - float(start[index]) for index in range(3)]
    offset = [float(point[index]) - float(start[index]) for index in range(3)]
    denominator = sum(value * value for value in segment)
    if denominator <= 1e-12:
        return l2(point, start)
    fraction = max(
        0.0,
        min(1.0, sum(offset[index] * segment[index] for index in range(3)) / denominator),
    )
    closest = [float(start[index]) + fraction * segment[index] for index in range(3)]
    return l2(point, closest)



def dtw_distance(path_a: Sequence[Vec3], path_b: Sequence[Vec3]) -> float:
    n, m = len(path_a), len(path_b)
    if n == 0 or m == 0:
        return float("inf")
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = l2(path_a[i - 1], path_b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]



def ndtw(pred_path: Sequence[Vec3], ref_path: Sequence[Vec3], success_radius: float) -> float:
    if not pred_path or not ref_path:
        return 0.0
    dist = dtw_distance(pred_path, ref_path)
    # Standard nDTW normalizes accumulated DTW error by the number of points
    # in the reference trajectory and the task success threshold.
    return math.exp(-dist / (max(float(success_radius), 1e-6) * len(ref_path)))



def spl(
    pred_path: Sequence[Vec3],
    goal: Vec3,
    success_radius: float,
    shortest_path_length: float,
    stopped: bool = True,
) -> float:
    if not math.isfinite(float(shortest_path_length)) or float(shortest_path_length) <= 0.0:
        raise ValueError(
            "SPL requires a positive independently computed shortest_path_length; "
            "the reference trajectory length is not a valid fallback"
        )
    succ = 1.0 if success(pred_path, goal, success_radius, stopped=stopped) else 0.0
    actual = max(path_length(pred_path), 1e-6)
    optimal = max(float(shortest_path_length), 1e-6)
    return succ * optimal / max(actual, optimal)



def reference_travel_time(ref_path: Sequence[Vec3], speed_mps: float) -> float:
    """Return the reproducible expert travel time at the configured simulator speed."""
    speed = max(float(speed_mps), 1e-6)
    return path_length(ref_path) / speed


def summarize_episode(
    pred_path: Sequence[Vec3],
    ref_path: Sequence[Vec3],
    goal: Vec3,
    success_radius: float,
    shortest_path_length: float,
    stopped: bool = True,
) -> dict[str, float]:
    return {
        "NE": navigation_error(pred_path, goal),
        "SR": float(success(pred_path, goal, success_radius, stopped=stopped)),
        "OSR": float(oracle_success(pred_path, goal, success_radius)),
        "nDTW": ndtw(pred_path, ref_path, success_radius),
        "SPL": spl(pred_path, goal, success_radius, shortest_path_length, stopped=stopped),
    }



def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {key: 0.0 for key in METRIC_KEYS}
    keys = [key for key in METRIC_KEYS if key in items[0]]
    return {k: sum(x[k] for x in items) / len(items) for k in keys}


def aggregate_by_difficulty(items: list[dict[str, float | str | None]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | str | None]]] = {}
    for item in items:
        difficulty = str(item.get("difficulty") or "Unknown")
        grouped.setdefault(difficulty, []).append(item)

    ordered = [name for name in DIFFICULTY_ORDER if name in grouped]
    ordered.extend(name for name in sorted(grouped) if name not in DIFFICULTY_ORDER)

    summary: dict[str, dict[str, float | int]] = {}
    for difficulty in ordered:
        group = grouped[difficulty]
        metrics = aggregate_metrics([{key: float(item[key]) for key in METRIC_KEYS} for item in group])
        summary[difficulty] = {"episodes": len(group), **metrics}
    return summary
