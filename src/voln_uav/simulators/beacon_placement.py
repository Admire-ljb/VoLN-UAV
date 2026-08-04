from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from voln_uav.benchmark.beacon_protocol import (
    task_beacon_route_index,
    validate_episode_task_beacons,
)

SIGN_ASSET_BASE = {
    "left_yaw": "left",
    "left_turn": "left_turn",
    "left90": "left_turn",
    "right_yaw": "right",
    "right_turn": "right_turn",
    "right90": "right_turn",
    "up": "up",
    "down": "down",
    "here": "here",
}
SIGN_ASSET_ALIASES = {
    "left_yaw": ("left_turn", "left"),
    "left_turn": ("left_turn", "left"),
    "left90": ("left_turn", "left"),
    "right_yaw": ("right_turn", "right"),
    "right_turn": ("right_turn", "right"),
    "right90": ("right_turn", "right"),
    "up": ("up",),
    "down": ("down",),
    "here": ("here",),
}
TURN_SIGN_TAGS = frozenset(
    {
        "left_yaw", "left_turn", "left90",
        "right_yaw", "right_turn", "right90",
    }
)
TARGET_TAG = "target_people"
BEACON_RENDER_MODES = {"random", "direction", "text"}
SEMANTIC_SIGN_TAG = {
    "turn-left": "left_turn",
    "left-turn": "left_turn",
    "turn-right": "right_turn",
    "right-turn": "right_turn",
    "ascend": "up",
    "up": "up",
    "descend": "down",
    "down": "down",
}

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


def normalize_beacon_render_mode(value: Any = None) -> str:
    """Normalize the active-beacon asset style selected for an evaluation run."""
    raw = str(value or "direction").strip().lower()
    aliases = {
        "icon": "direction",
        "icons": "direction",
        "label": "text",
        "labels": "text",
    }
    mode = aliases.get(raw, raw)
    if mode not in BEACON_RENDER_MODES:
        choices = ", ".join(sorted(BEACON_RENDER_MODES))
        raise ValueError(f"Unsupported beacon render mode {value!r}; use one of: {choices}")
    return mode


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


def path_yaw_deg(states: list[dict[str, Any]], index: int) -> float:
    """Return the local horizontal route heading, independent of camera yaw."""
    position = states[index]["position"]
    for next_index in range(index + 1, len(states)):
        next_position = states[next_index]["position"]
        if math.hypot(
            float(next_position[0]) - float(position[0]),
            float(next_position[1]) - float(position[1]),
        ) > 1e-6:
            return yaw_to_target_deg(position, next_position)
    for previous_index in range(index - 1, -1, -1):
        previous_position = states[previous_index]["position"]
        if math.hypot(
            float(position[0]) - float(previous_position[0]),
            float(position[1]) - float(previous_position[1]),
        ) > 1e-6:
            return yaw_to_target_deg(previous_position, position)
    return state_yaw_deg(states, index)


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
    # Beacons describe the route, not transient camera/body yaw stored in a frame.
    yaws = [path_yaw_deg(states, i) for i in range(len(states))]
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


def _route_turn_events(
    features: list[dict[str, Any]],
    onset_delta_deg: float = 0.5,
    max_straight_gap_m: float = 6.0,
    terminal_reverse_window_steps: int = 12,
    terminal_reverse_max_straight_gap_m: float = 30.0,
) -> list[dict[str, Any]]:
    """Group heading changes into route-level turns using physical spacing.

    Consecutive same-direction changes form one event. A straight gap longer
    than max_straight_gap_m separates two turns, independent of frame rate.
    """
    events: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    straight_gap_m = 0.0

    def flush() -> None:
        nonlocal active, straight_gap_m
        if active is None:
            return
        start = int(active["start"])
        end = int(active["end"])
        cumulative_yaw = float(active["d_yaw_deg"])
        tag, score, reason = _tag_from_motion(cumulative_yaw, 0.0)
        if tag not in TURN_SIGN_TAGS:
            active = None
            straight_gap_m = 0.0
            return
        events.append({
            **features[start],
            "index": start,
            "d_yaw_deg": cumulative_yaw,
            "tag": tag,
            "score": score,
            "reason": f"{reason} over route indices {start}-{end}",
            "turn_start_index": start,
            "turn_end_index": end,
        })
        active = None
        straight_gap_m = 0.0

    for index in range(1, len(features)):
        delta = float(features[index].get("d_yaw_deg", 0.0))
        segment_m = float(features[index].get("ds_m", 0.0))
        if abs(delta) < float(onset_delta_deg):
            if active is not None:
                straight_gap_m += segment_m
                if straight_gap_m > float(max_straight_gap_m):
                    flush()
            continue

        direction = 1 if delta > 0.0 else -1
        if (
            active is None
            or int(active["direction"]) != direction
            or straight_gap_m > float(max_straight_gap_m)
        ):
            flush()
            active = {
                "direction": direction,
                "start": index,
                "end": index,
                "d_yaw_deg": delta,
            }
        else:
            active["end"] = index
            active["d_yaw_deg"] = float(active["d_yaw_deg"]) + delta
        straight_gap_m = 0.0

    flush()
    return _merge_terminal_reversing_turn_events(
        events,
        features,
        onset_delta_deg=float(onset_delta_deg),
        window_steps=int(terminal_reverse_window_steps),
        max_straight_gap_m=float(terminal_reverse_max_straight_gap_m),
    )


def _merge_terminal_reversing_turn_events(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    onset_delta_deg: float,
    window_steps: int,
    max_straight_gap_m: float,
) -> list[dict[str, Any]]:
    """Merge a reversing terminal fragment using its signed net yaw.

    Small opposite corrections near the goal should not create contradictory
    signs. Direction follows the signed sum and the sign class follows its
    absolute magnitude. A fully cancelled fragment emits no turn sign.
    """
    if len(features) < 3 or int(window_steps) < 2:
        return events

    lower_bound = max(1, len(features) - int(window_steps))
    significant_indices: list[int] = []
    straight_gap_m = 0.0
    for index in range(len(features) - 1, lower_bound - 1, -1):
        delta = float(features[index].get("d_yaw_deg", 0.0))
        if abs(delta) >= float(onset_delta_deg):
            significant_indices.append(index)
            straight_gap_m = 0.0
            continue
        straight_gap_m += float(features[index].get("ds_m", 0.0))
        if straight_gap_m > float(max_straight_gap_m):
            if not significant_indices:
                return events
            break

    if len(significant_indices) < 2:
        return events
    significant_indices.sort()
    directions = [
        1 if float(features[index]["d_yaw_deg"]) > 0.0 else -1
        for index in significant_indices
    ]
    reversal_count = sum(
        first != second for first, second in zip(directions, directions[1:])
    )
    if reversal_count <= 0:
        return events

    start = significant_indices[0]
    end = significant_indices[-1]
    net_yaw_deg = sum(
        float(features[index].get("d_yaw_deg", 0.0))
        for index in range(start, end + 1)
    )
    absolute_yaw_deg = sum(
        abs(float(features[index].get("d_yaw_deg", 0.0)))
        for index in range(start, end + 1)
    )
    remaining = [
        event
        for event in events
        if int(event.get("turn_end_index", event["index"])) < start
        or int(event.get("turn_start_index", event["index"])) > end
    ]
    tag, score, reason = _tag_from_motion(net_yaw_deg, 0.0)
    if tag in TURN_SIGN_TAGS:
        remaining.append(
            {
                **features[start],
                "index": start,
                "d_yaw_deg": net_yaw_deg,
                "tag": tag,
                "score": score,
                "reason": f"{reason} net over reversing terminal route indices {start}-{end}",
                "turn_start_index": start,
                "turn_end_index": end,
                "terminal_reverse_count": reversal_count,
                "terminal_absolute_yaw_deg": absolute_yaw_deg,
            }
        )
    remaining.sort(key=lambda item: int(item["index"]))
    return remaining


def _route_warning_anchor_before_index(
    features: list[dict[str, Any]],
    turn_index: int,
    warning_distance_m: float,
) -> tuple[int, list[float], float]:
    """Find the incoming-route point a fixed horizontal distance before a turn."""
    index = min(max(int(turn_index), 0), len(features) - 1)
    target_distance = max(float(warning_distance_m), 0.0)
    if target_distance <= 0.0 or index <= 0:
        feature = features[index]
        return index, list(feature["position"]), float(feature["yaw_deg"])

    origin = [float(value) for value in features[index]["position"]]
    final_backward_direction: list[float] | None = None
    final_yaw_deg = float(features[index]["yaw_deg"])
    for current_index in range(index, 0, -1):
        current = [float(value) for value in features[current_index]["position"]]
        previous = [
            float(value) for value in features[current_index - 1]["position"]
        ]
        dx = previous[0] - current[0]
        dy = previous[1] - current[1]
        segment_sq = dx * dx + dy * dy
        if segment_sq <= 1e-12:
            continue
        segment = math.sqrt(segment_sq)
        final_backward_direction = [
            dx / segment,
            dy / segment,
        ]
        final_yaw_deg = yaw_to_target_deg(previous, current)

        relative_x = current[0] - origin[0]
        relative_y = current[1] - origin[1]
        quadratic_b = 2.0 * (relative_x * dx + relative_y * dy)
        quadratic_c = (
            relative_x * relative_x
            + relative_y * relative_y
            - target_distance * target_distance
        )
        discriminant = quadratic_b * quadratic_b - 4.0 * segment_sq * quadratic_c
        if discriminant < -1e-9:
            continue
        root = math.sqrt(max(discriminant, 0.0))
        ratios = sorted(
            ratio
            for ratio in (
                (-quadratic_b - root) / (2.0 * segment_sq),
                (-quadratic_b + root) / (2.0 * segment_sq),
            )
            if ratio > 1e-9 and ratio <= 1.0 + 1e-9
        )
        if ratios:
            ratio = min(ratios[0], 1.0)
            anchor = [
                current[axis] + ratio * (previous[axis] - current[axis])
                for axis in range(3)
            ]
            return current_index - 1, anchor, final_yaw_deg

    first = features[0]
    if final_backward_direction is None:
        return 0, list(first["position"]), float(first["yaw_deg"])
    relative_x = float(first["position"][0]) - origin[0]
    relative_y = float(first["position"][1]) - origin[1]
    projection = (
        relative_x * final_backward_direction[0]
        + relative_y * final_backward_direction[1]
    )
    quadratic_c = (
        relative_x * relative_x
        + relative_y * relative_y
        - target_distance * target_distance
    )
    discriminant = max(projection * projection - quadratic_c, 0.0)
    forward_roots = [
        distance
        for distance in (
            -projection - math.sqrt(discriminant),
            -projection + math.sqrt(discriminant),
        )
        if distance >= 0.0
    ]
    extension = min(forward_roots) if forward_roots else 0.0
    anchor = [
        float(first["position"][0]) + extension * final_backward_direction[0],
        float(first["position"][1]) + extension * final_backward_direction[1],
        float(first["position"][2]),
    ]
    return 0, anchor, final_yaw_deg


def _route_anchor_after_index(
    features: list[dict[str, Any]],
    turn_index: int,
    after_distance_m: float,
) -> tuple[int, list[float], float]:
    """Find the first route point a fixed horizontal distance after a turn."""
    index = min(max(int(turn_index), 0), len(features) - 1)
    target_distance = max(float(after_distance_m), 0.0)
    if target_distance <= 0.0 or index >= len(features) - 1:
        feature = features[index]
        return index, list(feature["position"]), float(feature["yaw_deg"])

    origin = [float(value) for value in features[index]["position"]]
    final_direction: list[float] | None = None
    final_yaw_deg = float(features[index]["yaw_deg"])
    for current_index in range(index, len(features) - 1):
        current = [float(value) for value in features[current_index]["position"]]
        following = [
            float(value) for value in features[current_index + 1]["position"]
        ]
        dx = following[0] - current[0]
        dy = following[1] - current[1]
        segment_sq = dx * dx + dy * dy
        if segment_sq <= 1e-12:
            continue
        segment = math.sqrt(segment_sq)
        final_direction = [dx / segment, dy / segment]
        final_yaw_deg = yaw_to_target_deg(current, following)

        relative_x = current[0] - origin[0]
        relative_y = current[1] - origin[1]
        quadratic_b = 2.0 * (relative_x * dx + relative_y * dy)
        quadratic_c = (
            relative_x * relative_x
            + relative_y * relative_y
            - target_distance * target_distance
        )
        discriminant = quadratic_b * quadratic_b - 4.0 * segment_sq * quadratic_c
        if discriminant < -1e-9:
            continue
        root = math.sqrt(max(discriminant, 0.0))
        ratios = sorted(
            ratio
            for ratio in (
                (-quadratic_b - root) / (2.0 * segment_sq),
                (-quadratic_b + root) / (2.0 * segment_sq),
            )
            if ratio > 1e-9 and ratio <= 1.0 + 1e-9
        )
        if ratios:
            ratio = min(ratios[0], 1.0)
            anchor = [
                current[axis] + ratio * (following[axis] - current[axis])
                for axis in range(3)
            ]
            return current_index, anchor, final_yaw_deg

    last = features[-1]
    if final_direction is None:
        return len(features) - 1, list(last["position"]), float(last["yaw_deg"])
    relative_x = float(last["position"][0]) - origin[0]
    relative_y = float(last["position"][1]) - origin[1]
    projection = (
        relative_x * final_direction[0]
        + relative_y * final_direction[1]
    )
    quadratic_c = (
        relative_x * relative_x
        + relative_y * relative_y
        - target_distance * target_distance
    )
    discriminant = max(projection * projection - quadratic_c, 0.0)
    forward_roots = [
        distance
        for distance in (
            -projection - math.sqrt(discriminant),
            -projection + math.sqrt(discriminant),
        )
        if distance >= 0.0
    ]
    extension = min(forward_roots) if forward_roots else 0.0
    anchor = [
        float(last["position"][0]) + extension * final_direction[0],
        float(last["position"][1]) + extension * final_direction[1],
        float(last["position"][2]),
    ]
    return len(features) - 1, anchor, final_yaw_deg


def _turn_direction_offset_anchor(
    features: list[dict[str, Any]],
    item: dict[str, Any],
    signed_distance_m: float,
) -> tuple[int, list[float], float]:
    """Offset from a turn pivot along the incoming flight direction."""
    start = min(max(int(item["index"]), 0), len(features) - 1)
    end = min(
        max(int(item.get("turn_end_index", start)), start),
        len(features) - 1,
    )
    pivot = max(
        range(start, end + 1),
        key=lambda index: abs(float(features[index].get("d_yaw_deg", 0.0))),
    )
    direction_reference_index = max(start - 1, 0)
    direction_yaw_deg = float(features[direction_reference_index]["yaw_deg"])
    direction_yaw_rad = math.radians(direction_yaw_deg)
    origin = [float(value) for value in features[pivot]["position"]]
    distance = float(signed_distance_m)
    anchor = [
        origin[0] + distance * math.cos(direction_yaw_rad),
        origin[1] + distance * math.sin(direction_yaw_rad),
        origin[2],
    ]
    return pivot, anchor, direction_yaw_deg


def _task_beacon_tag(
    beacon: dict[str, Any],
    feature: dict[str, Any],
    straight_tag: str,
) -> str:
    for key in ("asset_tag", "tag"):
        explicit = str(beacon.get(key, "")).strip()
        if explicit in SIGN_ASSET_BASE:
            return explicit
    semantic_type = str(beacon.get("semantic_type", "")).strip().lower()
    semantic_tag = SEMANTIC_SIGN_TAG.get(semantic_type)
    if semantic_tag is not None:
        return semantic_tag
    motion_tag = feature.get("tag")
    if motion_tag in SIGN_ASSET_BASE:
        return str(motion_tag)
    if straight_tag not in SIGN_ASSET_BASE:
        raise ValueError(f"Unsupported straight beacon tag: {straight_tag!r}")
    return straight_tag


def _features_from_episode_task_beacons(
    episode: dict[str, Any],
    cfg: dict[str, Any],
    features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_beacons, _expected_count, _path_length_m = validate_episode_task_beacons(
        episode,
        cfg,
    )
    straight_tag = str(cfg.get("straight_tag", "up"))
    selected: list[dict[str, Any]] = []
    for order, beacon in enumerate(task_beacons):
        index = task_beacon_route_index(beacon)
        feature = features[index]
        tag = _task_beacon_tag(beacon, feature, straight_tag)
        selected.append(
            {
                **feature,
                "tag": tag,
                "reason": str(
                    beacon.get(
                        "reason",
                        feature.get("reason", "episode task beacon"),
                    )
                ),
                "task_beacon_id": str(beacon["beacon_id"]),
                "task_beacon_order": order,
                "semantic_type": str(beacon["semantic_type"]),
                "source": "episode_task_beacons",
            }
        )
    return selected


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

    include_target = bool(cfg.get("include_target", True))
    seed = stable_episode_seed(int(cfg.get("random_seed", base_seed)), str(episode.get("episode_id", "episode")))
    rng = random.Random(seed)

    features = route_motion_features(states)
    straight_tag = str(cfg.get("straight_tag", "here"))
    source = str(cfg.get("source", "generated_from_route"))
    if source == "episode_task_beacons":
        selected = _features_from_episode_task_beacons(episode, cfg, features)
    elif source == "generated_from_route":
        count = max(
            int(cfg.get("count", cfg.get("route_beacons_per_episode", 4))),
            0,
        )
        allowed_tags = {
            str(tag)
            for tag in cfg.get("allowed_tags", SIGN_ASSET_BASE.keys())
            if str(tag) in SIGN_ASSET_BASE
        }
        if not allowed_tags:
            allowed_tags = set(SIGN_ASSET_BASE)
        start_margin = int(cfg.get("start_margin_steps", 1))
        end_margin = int(cfg.get("end_margin_steps", 2))
        min_gap = int(
            cfg.get("min_gap_steps", max(2, len(states) // max(count + 2, 3)))
        )
        motion_candidates = [
            item
            for item in features[
                start_margin : max(start_margin, len(features) - end_margin)
            ]
            if item["tag"] in allowed_tags and item["score"] > 0.0
        ]
        turn_candidates = [
            item
            for item in _route_turn_events(
                features,
                float(cfg.get("turn_onset_delta_deg", 0.5)),
                float(cfg.get("turn_merge_straight_gap_m", 6.0)),
                int(cfg.get("terminal_reverse_turn_window_steps", 12)),
                float(cfg.get("terminal_reverse_turn_max_straight_gap_m", 30.0)),
            )
            if int(item["index"]) >= start_margin and item["tag"] in allowed_tags
        ]
        other_candidates = [
            item for item in motion_candidates if item["tag"] not in TURN_SIGN_TAGS
        ]
        candidate_key = lambda item: (
            -float(item["score"]),
            int(item["index"]),
        )
        turn_candidates.sort(key=candidate_key)
        other_candidates.sort(key=candidate_key)
        pool_limit = max(count * 4, count)
        turn_pool = turn_candidates[:pool_limit]
        other_pool = other_candidates[:pool_limit]
        rng.shuffle(other_pool)
        pool = turn_pool + other_pool

        selected = []
        selected_indices: list[int] = []
        for item in pool:
            if len(selected) >= count:
                break
            if _far_enough(int(item["index"]), selected_indices, min_gap):
                selected.append(item)
                selected_indices.append(int(item["index"]))

        # Do not manufacture three directional cues on an instruction-free
        # straight route. Keep one terminal HERE marker instead.
        if not selected and count > 0 and straight_tag in allowed_tags:
            terminal = features[-1]
            selected.append(
                {
                    **terminal,
                    "tag": straight_tag,
                    "score": 0.0,
                    "reason": "end of straight route",
                }
            )

    else:
        raise ValueError(
            "beacon_placement.source must be 'episode_task_beacons' or "
            f"'generated_from_route', got {source!r}"
        )

    selected.sort(key=lambda item: int(item["index"]))
    preset = dict(DIFFICULTY_PRESETS.get(str(episode.get("difficulty", "Normal")), DIFFICULTY_PRESETS["Normal"]))
    preset.update(cfg.get("preset_override", {}) or {})
    scene_id = str(episode.get("scene_id", ""))
    ground_z_by_scene = dict(cfg.get("ground_z_ned_by_scene", {}) or {})
    ground_z_value = ground_z_by_scene.get(scene_id, cfg.get("ground_z_ned"))
    ground_z_ned = float(ground_z_value) if ground_z_value is not None else None
    vertical_offset_by_scene = dict(cfg.get("vertical_ned_offset_by_scene", {}) or {})
    scene_vertical_offset = vertical_offset_by_scene.get(scene_id)
    lateral_lo, lateral_hi = preset.get("lateral", (-7.0, -3.0))
    lookback_steps = int(cfg.get("lookback_steps", 2))
    distance_jitter = float(cfg.get("distance_jitter_m", 3.0))

    placements: list[dict[str, Any]] = []
    for order, item in enumerate(selected):
        idx = int(item["index"])
        tag = str(item["tag"])
        turn_warning_distance_m = 0.0
        if tag in TURN_SIGN_TAGS:
            turn_warning_distance_m = float(cfg.get("turn_warning_distance_m", 30.0))
            ref_idx, anchor_position, anchor_yaw_deg = _turn_direction_offset_anchor(
                features, item, turn_warning_distance_m
            )
            forward = float(cfg.get("turn_forward_m", 0.0))
            yaw_add_deg = float(cfg.get("turn_yaw_add_deg", 90.0))
        elif tag == straight_tag:
            ref_idx = idx
            ref = features[ref_idx]
            anchor_position = list(ref["position"])
            anchor_yaw_deg = float(ref["yaw_deg"])
            forward = float(cfg.get("here_forward_m", 0.0))
            yaw_add_deg = float(cfg.get("yaw_add_deg", 90.0))
        else:
            ref_idx = max(0, idx - lookback_steps)
            ref = features[ref_idx]
            anchor_position = list(ref["position"])
            anchor_yaw_deg = float(ref["yaw_deg"])
            forward = (
                float(preset.get("distance", 45.0))
                + rng.uniform(-distance_jitter, distance_jitter)
            )
            yaw_add_deg = float(cfg.get("yaw_add_deg", 90.0))
        lateral = rng.uniform(float(lateral_lo), float(lateral_hi))
        vertical_ned_m = (
            ground_z_ned - float(anchor_position[2])
            if ground_z_ned is not None
            else float(
                scene_vertical_offset
                if scene_vertical_offset is not None
                else preset.get("vertical_ned", 10.0)
            )
        )
        pose = pose_from_path_point(
            position=anchor_position,
            yaw_deg=anchor_yaw_deg,
            forward_m=forward,
            lateral_m=lateral,
            vertical_ned_m=vertical_ned_m,
            yaw_add_deg=yaw_add_deg,
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
                "turn_total_yaw_deg": float(item.get("d_yaw_deg", 0.0)),
                "terminal_reverse_count": int(item.get("terminal_reverse_count", 0)),
                "terminal_absolute_yaw_deg": float(item.get("terminal_absolute_yaw_deg", 0.0)),
                "turn_warning_distance_m": turn_warning_distance_m,
                "turn_event_end_index": int(item.get("turn_end_index", idx)),
                "route_anchor_position": anchor_position,
                "turn_pivot_index": ref_idx,
                "turn_direction_reference_index": max(idx - 1, 0),
                "turn_direction_yaw_deg": anchor_yaw_deg,
                "lateral_m": lateral,
                "vertical_ned_m": vertical_ned_m,
                **(
                    {
                        "task_beacon_id": item["task_beacon_id"],
                        "task_beacon_order": int(item["task_beacon_order"]),
                        "semantic_type": item["semantic_type"],
                        "source": item["source"],
                    }
                    if "task_beacon_id" in item
                    else {}
                ),
                **pose,
            }
        )

    if include_target:
        target_idx = len(states) - 1
        target_ref = features[target_idx]
        target_state = states[target_idx]
        target_direction_source = "recorded_final_yaw" if "yaw" in target_state else "route_heading_fallback"
        target_yaw_deg = state_yaw_deg(states, target_idx)
        target_vertical_ned_m = float(cfg.get("target_vertical_ned_m", 0.0))
        target_pose = pose_from_path_point(
            position=target_ref["position"],
            yaw_deg=target_yaw_deg,
            forward_m=float(cfg.get("target_distance_m", 0.0)),
            lateral_m=float(cfg.get("target_lateral_m", 0.0)),
            vertical_ned_m=target_vertical_ned_m,
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
                "vertical_ned_m": target_vertical_ned_m,
                "target_direction_source": target_direction_source,
                "target_direction_yaw_deg": target_yaw_deg,
                **target_pose,
            }
        )
    return placements
