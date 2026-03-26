from __future__ import annotations

import math
from typing import Sequence

from voln_uav.common.geometry import l2, path_length


Vec3 = Sequence[float]



def navigation_error(pred_path: Sequence[Vec3], goal: Vec3) -> float:
    if not pred_path:
        return float("inf")
    return l2(pred_path[-1], goal)



def success(pred_path: Sequence[Vec3], goal: Vec3, radius: float) -> bool:
    return bool(pred_path) and l2(pred_path[-1], goal) <= radius



def oracle_success(pred_path: Sequence[Vec3], goal: Vec3, radius: float) -> bool:
    return any(l2(p, goal) <= radius for p in pred_path)



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
    ref_len = max(path_length(ref_path), 1e-6)
    return math.exp(-dist / (success_radius * ref_len))



def spl(pred_path: Sequence[Vec3], goal: Vec3, success_radius: float, shortest_path_length: float) -> float:
    succ = 1.0 if success(pred_path, goal, success_radius) else 0.0
    actual = max(path_length(pred_path), 1e-6)
    optimal = max(float(shortest_path_length), 1e-6)
    return succ * optimal / max(actual, optimal)



def summarize_episode(pred_path: Sequence[Vec3], ref_path: Sequence[Vec3], goal: Vec3, success_radius: float, shortest_path_length: float) -> dict[str, float]:
    return {
        "NE": navigation_error(pred_path, goal),
        "SR": float(success(pred_path, goal, success_radius)),
        "OSR": float(oracle_success(pred_path, goal, success_radius)),
        "nDTW": ndtw(pred_path, ref_path, success_radius),
        "SPL": spl(pred_path, goal, success_radius, shortest_path_length),
    }



def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {"NE": 0.0, "SR": 0.0, "OSR": 0.0, "nDTW": 0.0, "SPL": 0.0}
    keys = items[0].keys()
    return {k: sum(x[k] for x in items) / len(items) for k in keys}
