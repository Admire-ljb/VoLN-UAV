from __future__ import annotations

from pathlib import Path
from typing import Any

from voln_uav.common.geometry import altitude_delta, heading_delta, path_length
from voln_uav.common.io import read_json


def load_route_files(source_root: Path, preset_routes_dir: str, custom_routes_dir: str) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for subdir, source_name in [(preset_routes_dir, "preset"), (custom_routes_dir, "custom")]:
        route_dir = source_root / subdir
        if not route_dir.exists():
            continue
        for route_path in sorted(route_dir.glob("*.json")):
            route = read_json(route_path)
            route.setdefault("source", source_name)
            route["_route_file"] = str(route_path)
            routes.append(route)
    return routes


def states_positions(route: dict[str, Any]) -> list[list[float]]:
    return [list(map(float, st["position"])) for st in route["states"]]


def compute_path_length(route: dict[str, Any]) -> float:
    return path_length(states_positions(route))


def compute_start_goal_distance(route: dict[str, Any]) -> float:
    pts = states_positions(route)
    if len(pts) < 2:
        return 0.0
    return ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2 + (pts[-1][2] - pts[0][2]) ** 2) ** 0.5


def find_decision_points(route: dict[str, Any], turn_threshold_rad: float = 0.45, altitude_threshold: float = 1.0, min_separation_steps: int = 4) -> list[int]:
    pts = states_positions(route)
    if len(pts) < 3:
        return []
    candidates: list[int] = []
    last = -10**9
    for i in range(1, len(pts) - 1):
        turn = abs(heading_delta(pts[i - 1], pts[i], pts[i + 1]))
        climb = abs(altitude_delta(pts[i - 1], pts[i + 1]))
        if turn >= turn_threshold_rad or climb >= altitude_threshold:
            if i - last >= min_separation_steps:
                candidates.append(i)
                last = i
    return candidates


def route_goal_category(route: dict[str, Any]) -> str:
    return str(route.get("goal_category", "goal"))
