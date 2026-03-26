from __future__ import annotations

import random
from typing import Any

from voln_uav.common.geometry import within_threshold



def assign_scene_splits(scene_ids: list[str], split_ratios: dict[str, float], seed: int) -> dict[str, str]:
    names = list(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(names)
    total = len(names)
    train_n = int(total * split_ratios["train"])
    val_n = int(total * split_ratios["val"])
    split_map: dict[str, str] = {}
    for i, scene_id in enumerate(names):
        if i < train_n:
            split_map[scene_id] = "train"
        elif i < train_n + val_n:
            split_map[scene_id] = "val"
        else:
            split_map[scene_id] = "test"
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
