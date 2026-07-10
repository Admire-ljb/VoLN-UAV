from __future__ import annotations

import hashlib
import math
import random
from typing import Any

SIGN_ASSET_BASE = {
    "left_yaw": "label_left_yaw",
    "left_turn": "label_left_turn",
    "left90": "label_left90",
    "right_yaw": "label_right_yaw",
    "right_turn": "label_right_turn",
    "right90": "label_right90",
    "up": "label_up",
    "down": "label_down",
}
TARGET_TAG = "target_people"

DIFFICULTY_PRESETS = {
    "Easy": {"distance": 32.0, "lateral": (-5.0, -2.0), "vertical_ned": 8.0},
    "Normal": {"distance": 45.0, "lateral": (-7.0, -3.0), "vertical_ned": 10.0},
    "Hard": {"distance": 55.0, "lateral": (-10.0, -4.0), "vertical_ned": 12.0},
}

YAW_SMALL = 15.0
YAW_MED = 35.0
YAW_LARGE = 60.0
SLOPE_UP = 1.0
SLOPE_DOWN = -1.0


def stable_episode_seed(base_seed: int, episode_id: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{episode_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def norm_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def yaw_to_target_deg(src: list[float], dst: list[float]) -> float:
    return math.degrees(math.atan2(float(dst[1]) - float(src[1]), float(dst[0]) - float(src[0])))


def state_yaw_deg(states: list[dict[str, Any]], index: int) -> float:
    state = states[index]
    if "yaw" in state:
        yaw = float(state.get("yaw", 0.0))
        if abs(yaw) <= 2.0 * math.pi + 1e-3:
            return math.degrees(yaw)
        return yaw
    if index + 1 < len(states):
        return yaw_to_target_deg(states[index]["position"], states[index + 1]["position"])
    if index > 0:
        return yaw_to_target_deg(states[index - 1]["position"], states[index]["position"])
    return 0.0


def _tag_from_motion(d_yaw_deg: float, slope_m: float) -> tuple[str | None, float, str]:
    if d_yaw_deg >= YAW_LARGE:
        return "right90", abs(d_yaw_deg), f"right turn {d_yaw_deg:.1f} deg"
    if d_yaw_deg >= YAW_MED:
        return "right_turn", abs(d_yaw_deg), f"right turn {d_yaw_deg:.1f} deg"
    if d_yaw_deg >= YAW_SMALL:
        return "right_yaw", abs(d_yaw_deg), f"right yaw {d_yaw_deg:.1f} deg"
    if d_yaw_deg <= -YAW_LARGE:
        return "left90", abs(d_yaw_deg), f"left turn {d_yaw_deg:.1f} deg"
    if d_yaw_deg <= -YAW_MED:
        return "left_turn", abs(d_yaw_deg), f"left turn {d_yaw_deg:.1f} deg"
    if d_yaw_deg <= -YAW_SMALL:
        return "left_yaw", abs(d_yaw_deg), f"left yaw {d_yaw_deg:.1f} deg"
    if slope_m >= SLOPE_UP:
        return "up", abs(slope_m) * 12.0, f"ascending {slope_m:.2f} m"
    if slope_m <= SLOPE_DOWN:
        return "down", abs(slope_m) * 12.0, f"descending {slope_m:.2f} m"
    return None, 0.0, "straight segment"


def route_motion_features(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not states:
        return []
    yaws = [state_yaw_deg(states, i) for i in range(len(states))]
    features: list[dict[str, Any]] = []
    prev_pos = states[0]["position"]
    prev_yaw = yaws[0]
    prev_alt = -float(prev_pos[2])
    for idx, state in enumerate(states):
        pos = [float(v) for v in state["position"][:3]]
        yaw = yaws[idx]
        alt = -float(pos[2])
        if idx == 0:
            d_yaw = 0.0
            slope = 0.0
            ds = 0.0
        else:
            d_yaw = norm_angle_deg(yaw - prev_yaw)
            slope = alt - prev_alt
            ds = math.hypot(pos[0] - float(prev_pos[0]), pos[1] - float(prev_pos[1]))
        tag, score, reason = _tag_from_motion(d_yaw, slope)
        features.append(
            {
                "index": idx,
                "position": pos,
                "yaw_deg": yaw,
                "d_yaw_deg": d_yaw,
                "slope_m": slope,
                "ds_m": ds,
                "tag": tag,
                "score": score,
                "reason": reason,
            }
        )
        prev_pos = pos
        prev_yaw = yaw
        prev_alt = alt
    return features


def _far_enough(index: int, chosen: list[int], min_gap: int) -> bool:
    return all(abs(int(index) - int(other)) >= int(min_gap) for other in chosen)


def _fallback_indices(num_states: int, count: int, start_margin: int, end_margin: int) -> list[int]:
    lo = max(0, int(start_margin))
    hi = max(lo, num_states - 1 - int(end_margin))
    if hi <= lo:
        return [lo]
    return [min(hi, max(lo, round((i + 1) * hi / (count + 1)))) for i in range(count)]


def pose_from_path_point(
    position: list[float],
    yaw_deg: float,
    forward_m: float,
    lateral_m: float,
    vertical_ned_m: float,
    yaw_add_deg: float,
) -> dict[str, Any]:
    yaw = math.radians(float(yaw_deg))
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -math.sin(yaw), math.cos(yaw)
    x = float(position[0]) + float(forward_m) * fx + float(lateral_m) * rx
    y = float(position[1]) + float(forward_m) * fy + float(lateral_m) * ry
    z = float(position[2]) + float(vertical_ned_m)
    return {"position": [x, y, z], "yaw_rad": yaw + math.radians(float(yaw_add_deg))}


def plan_route_beacons(episode: dict[str, Any], config: dict[str, Any] | None = None, base_seed: int = 0) -> list[dict[str, Any]]:
    cfg = config or {}
    states = list(episode.get("states", []))
    if len(states) < 2:
        return []

    count = int(cfg.get("count", cfg.get("route_beacons_per_episode", 4)))
    count = max(count, 0)
    include_target = bool(cfg.get("include_target", True))
    seed = stable_episode_seed(int(cfg.get("random_seed", base_seed)), str(episode.get("episode_id", "episode")))
    rng = random.Random(seed)

    features = route_motion_features(states)
    start_margin = int(cfg.get("start_margin_steps", 1))
    end_margin = int(cfg.get("end_margin_steps", 2))
    min_gap = int(cfg.get("min_gap_steps", max(2, len(states) // max(count + 2, 3))))
    candidates = [
        item
        for item in features[start_margin : max(start_margin, len(features) - end_margin)]
        if item["tag"] in SIGN_ASSET_BASE and item["score"] > 0.0
    ]
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    pool = candidates[: max(count * 4, count)]
    rng.shuffle(pool)

    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for item in pool:
        if len(selected) >= count:
            break
        if _far_enough(int(item["index"]), selected_indices, min_gap):
            selected.append(item)
            selected_indices.append(int(item["index"]))

    fallback_tags = list(cfg.get("fallback_tags", ["left_yaw", "right_yaw", "up", "down"]))
    for idx in _fallback_indices(len(states), count, start_margin, end_margin):
        if len(selected) >= count:
            break
        if not _far_enough(idx, selected_indices, min_gap):
            continue
        feat = features[idx]
        tag = feat.get("tag") if feat.get("tag") in SIGN_ASSET_BASE else rng.choice(fallback_tags)
        selected.append({**feat, "tag": tag, "score": float(feat.get("score", 0.0)), "reason": "evenly spaced route beacon"})
        selected_indices.append(idx)

    selected.sort(key=lambda item: int(item["index"]))
    preset = dict(DIFFICULTY_PRESETS.get(str(episode.get("difficulty", "Normal")), DIFFICULTY_PRESETS["Normal"]))
    preset.update(cfg.get("preset_override", {}) or {})
    lateral_lo, lateral_hi = preset.get("lateral", (-7.0, -3.0))
    lookback_steps = int(cfg.get("lookback_steps", 2))
    distance_jitter = float(cfg.get("distance_jitter_m", 3.0))

    placements: list[dict[str, Any]] = []
    for order, item in enumerate(selected):
        idx = int(item["index"])
        ref_idx = max(0, idx - lookback_steps)
        ref = features[ref_idx]
        forward = float(preset.get("distance", 45.0)) + rng.uniform(-distance_jitter, distance_jitter)
        lateral = rng.uniform(float(lateral_lo), float(lateral_hi))
        pose = pose_from_path_point(
            position=ref["position"],
            yaw_deg=float(ref["yaw_deg"]),
            forward_m=forward,
            lateral_m=lateral,
            vertical_ned_m=float(preset.get("vertical_ned", 10.0)),
            yaw_add_deg=float(cfg.get("yaw_add_deg", 90.0)),
        )
        placements.append(
            {
                "kind": "route_beacon",
                "order": order,
                "index": idx,
                "ref_index": ref_idx,
                "tag": item["tag"],
                "reason": item.get("reason", "route cue"),
                "forward_m": forward,
                "lateral_m": lateral,
                "vertical_ned_m": float(preset.get("vertical_ned", 10.0)),
                **pose,
            }
        )

    if include_target:
        target_idx = len(states) - 1
        target_ref = features[target_idx]
        target_pose = pose_from_path_point(
            position=target_ref["position"],
            yaw_deg=float(target_ref["yaw_deg"]),
            forward_m=float(cfg.get("target_distance_m", 0.0)),
            lateral_m=float(cfg.get("target_lateral_m", 0.0)),
            vertical_ned_m=float(cfg.get("target_vertical_ned_m", 0.0)),
            yaw_add_deg=float(cfg.get("target_yaw_add_deg", 180.0)),
        )
        placements.append(
            {
                "kind": "target",
                "order": len(placements),
                "index": target_idx,
                "ref_index": target_idx,
                "tag": TARGET_TAG,
                "reason": "end of route",
                "forward_m": float(cfg.get("target_distance_m", 0.0)),
                "lateral_m": float(cfg.get("target_lateral_m", 0.0)),
                "vertical_ned_m": float(cfg.get("target_vertical_ned_m", 0.0)),
                **target_pose,
            }
        )
    return placements
