from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from voln_uav.common.geometry import within_threshold


PAPER_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "validation_seen": "val",
    "validation-seen": "val",
    "test": "test",
    "test_unseen": "test",
    "test-unseen": "test",
}


def normalize_paper_split(value: str) -> str:
    key = str(value).strip().casefold()
    try:
        return PAPER_SPLIT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown VoLN paper split: {value!r}") from exc


def trajectory_fingerprint(episode: dict[str, Any]) -> str:
    payload = [
        {
            "position": [round(float(value), 6) for value in state["position"][:3]],
            "yaw": round(float(state.get("yaw", 0.0)), 6),
        }
        for state in episode["states"]
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assign_episode_splits_from_manifest(
    episodes: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply an explicit episode-level split manifest without changing routes."""
    by_episode: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        episode_id = str(assignment["episode_id"])
        if episode_id in by_episode:
            raise ValueError(f"Duplicate split-manifest episode_id: {episode_id}")
        by_episode[episode_id] = assignment

    episode_ids = {str(episode["episode_id"]) for episode in episodes}
    missing = sorted(episode_ids - set(by_episode))
    extra = sorted(set(by_episode) - episode_ids)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing assignments for {len(missing)} episodes")
        if extra:
            detail.append(f"contains {len(extra)} unknown episodes")
        raise ValueError("Split manifest does not match built episodes: " + "; ".join(detail))

    assigned: list[dict[str, Any]] = []
    for episode in episodes:
        item = dict(episode)
        spec = by_episode[str(episode["episode_id"])]
        item["split"] = normalize_paper_split(str(spec["split"]))
        if spec.get("scene_source") is not None:
            item["scene_source"] = str(spec["scene_source"])
        if spec.get("shortest_path_length") is not None:
            item["shortest_path_length"] = float(spec["shortest_path_length"])
        if spec.get("shortest_path_provenance") is not None:
            item["shortest_path_provenance"] = dict(spec["shortest_path_provenance"])
        item["trajectory_hash"] = trajectory_fingerprint(item)
        expected_hash = spec.get("trajectory_hash")
        if expected_hash is not None and str(expected_hash) != item["trajectory_hash"]:
            raise ValueError(f"Trajectory hash mismatch for {item['episode_id']}")
        assigned.append(item)
    return assigned


def validate_paper_split_episodes(
    episodes: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    """Hard gate for the manuscript's episode- and environment-level split."""
    grouped = {
        split: [episode for episode in episodes if episode.get("split") == split]
        for split in ("train", "val", "test")
    }
    expected_episodes = {
        "train": int(protocol["train_episodes"]),
        "val": int(protocol["validation_seen_episodes"]),
        "test": int(protocol["test_unseen_episodes"]),
    }
    expected_environments = {
        "train": int(protocol["train_environments"]),
        "val": int(protocol["validation_seen_environments"]),
        "test": int(protocol["test_unseen_environments"]),
    }
    issues: list[str] = []
    for split in grouped:
        scenes = {str(episode["scene_id"]) for episode in grouped[split]}
        if len(grouped[split]) != expected_episodes[split]:
            issues.append(
                f"{split} has {len(grouped[split])} episodes, expected {expected_episodes[split]}"
            )
        if len(scenes) != expected_environments[split]:
            issues.append(
                f"{split} has {len(scenes)} environments, expected {expected_environments[split]}"
            )

    train_scenes = {str(episode["scene_id"]) for episode in grouped["train"]}
    val_scenes = {str(episode["scene_id"]) for episode in grouped["val"]}
    test_scenes = {str(episode["scene_id"]) for episode in grouped["test"]}
    if not val_scenes.issubset(train_scenes):
        issues.append("Validation-Seen environments must be a subset of Train environments")
    if test_scenes & (train_scenes | val_scenes):
        issues.append("Test-Unseen environments must be held out from Train/Validation-Seen")
    unique_scenes = train_scenes | val_scenes | test_scenes
    if len(unique_scenes) != int(protocol["total_environments"]):
        issues.append(
            f"benchmark has {len(unique_scenes)} unique environments, "
            f"expected {int(protocol['total_environments'])}"
        )

    train_hashes = {str(episode["trajectory_hash"]) for episode in grouped["train"]}
    val_hashes = {str(episode["trajectory_hash"]) for episode in grouped["val"]}
    if train_hashes & val_hashes:
        issues.append("Train and Validation-Seen contain overlapping trajectories")

    train_sources = {
        str(episode["scene_source"])
        for episode in grouped["train"]
        if episode.get("scene_source")
    }
    test_sources = {
        str(episode["scene_source"])
        for episode in grouped["test"]
        if episode.get("scene_source")
    }
    if bool(protocol.get("require_held_out_scene_source", True)):
        if any(not episode.get("scene_source") for episode in episodes):
            issues.append("scene_source is required for every paper-protocol episode")
        elif train_sources & test_sources:
            issues.append("Test-Unseen scene sources overlap the training pool")

    if issues:
        raise ValueError("Invalid VoLN paper split manifest:\n- " + "\n- ".join(issues))


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
