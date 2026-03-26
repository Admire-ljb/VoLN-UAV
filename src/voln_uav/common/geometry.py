from __future__ import annotations

import math
from typing import Iterable, Sequence


Vec3 = Sequence[float]


def l2(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def path_length(points: Sequence[Vec3]) -> float:
    if len(points) <= 1:
        return 0.0
    return sum(l2(points[i - 1], points[i]) for i in range(1, len(points)))


def cumulative_lengths(points: Sequence[Vec3]) -> list[float]:
    out = [0.0]
    for i in range(1, len(points)):
        out.append(out[-1] + l2(points[i - 1], points[i]))
    return out


def heading_delta(prev_pt: Vec3, cur_pt: Vec3, next_pt: Vec3) -> float:
    ax, ay = cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1]
    bx, by = next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1]
    ang_a = math.atan2(ay, ax)
    ang_b = math.atan2(by, bx)
    d = ang_b - ang_a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def altitude_delta(prev_pt: Vec3, next_pt: Vec3) -> float:
    return float(next_pt[2]) - float(prev_pt[2])


def evenly_spaced_indices(length: int, count: int, start: int = 0, end: int | None = None) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    end = length - 1 if end is None else end
    start = max(0, start)
    end = min(length - 1, end)
    if start > end:
        return []
    if count == 1:
        return [end]
    span = end - start
    if span <= 0:
        return [start]
    indices = []
    for i in range(count):
        idx = round(start + span * (i / (count - 1)))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def within_threshold(a_start: Vec3, a_goal: Vec3, b_start: Vec3, b_goal: Vec3, start_threshold: float, goal_threshold: float) -> bool:
    return l2(a_start, b_start) <= start_threshold and l2(a_goal, b_goal) <= goal_threshold
