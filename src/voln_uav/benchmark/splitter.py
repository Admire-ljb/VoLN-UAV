from __future__ import annotations

import math
import random
from typing import Any

from voln_uav.common.geometry import within_threshold


def split_counts(total: int, split_ratios: dict[str, float]) -> dict[str, int]:
    keys = list(split_ratios)
    if total <= 0:
        return {key: 0 for key in keys}
    raw = {key: total * float(split_ratios[key]) for key in keys}
    counts = {key: int(math.floor(raw[key])) for key in keys}
    remainder = total - sum(counts.values())
    ranked = sorted(keys, key=lambda key: (raw[key] - counts[key], float(split_ratios[key])), reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1

    positive = [key for key in keys if float(split_ratios[key]) > 0.0]
    if total >= len(positive):
        for key in positive:
            if counts[key] > 0:
                continue
            donor = max(positive, key=lambda name: counts[name])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[key] = 1
    return counts


def assign_scene_splits(scene_ids: list[str], split_ratios: dict[str, float], seed: int) -> dict[str, str]:
    names = list(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(names)
    total = len(names)
    counts = split_counts(total, split_ratios)
    split_map: dict[str, str] = {}
    cursor = 0
    for split_name, count in counts.items():
        for scene_id in names[cursor : cursor + count]:
            split_map[scene_id] = split_name
        cursor += count
    return split_map



def deduplicate_episodes(episodes: list[dict[str, Any]], start_threshold: float, goal_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for episode in episodes:
        start = episode["states"][0]["position"]
        goal = episode["states"][-1]["position"]
        duplicate = False
        for other in kept:
            if episode["scene_id"] != other["scene_id"]:
                continue
            if within_threshold(start, goal, other["states"][0]["position"], other["states"][-1]["position"], start_threshold, goal_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(episode)
    return kept
