from __future__ import annotations

from typing import Any

from voln_uav.common.geometry import evenly_spaced_indices



def build_visual_goal_interface(route: dict[str, Any], task_beacons: list[dict[str, Any]], num_terminal_views: int, num_subgoals: int, num_beacons: int) -> dict[str, Any]:
    states = route["states"]
    n = len(states)
    terminal_start = max(0, n - max(num_terminal_views * 2, 1))
    terminal_indices = evenly_spaced_indices(n, num_terminal_views, start=terminal_start, end=n - 1)
    subgoal_indices = evenly_spaced_indices(n, num_subgoals + 2, start=0, end=n - 1)
    subgoal_indices = [i for i in subgoal_indices[1:-1]][:num_subgoals]
    beacon_images = [b["template_image"] for b in task_beacons[:num_beacons]]
    return {
        "V_goal": [states[i]["image"] for i in terminal_indices],
        "V_sub": [states[i]["image"] for i in subgoal_indices],
        "V_beacon": beacon_images,
        "goal_indices": terminal_indices,
        "subgoal_indices": subgoal_indices,
    }
