from __future__ import annotations

import random
import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from voln_uav.common.io import ensure_dir


BEACON_COLORS = {
    "beacon-blue": (60, 120, 255),
    "beacon-red": (230, 80, 80),
    "beacon-yellow": (240, 210, 50),
    "beacon-green": (60, 190, 90),
    "road-sign": (200, 200, 220),
}


DEFAULT_SCENE_CATEGORIES = {
    "urban": ["beacon-blue", "beacon-red", "road-sign", "junction", "urban-canyon"],
    "forest": ["beacon-green", "beacon-yellow", "forest-trail", "turn-left", "turn-right"],
    "tunnel": ["beacon-red", "beacon-yellow", "tunnel", "industrial-corridor", "turn-left"],
}

CUE_FAMILIES = (
    "directional",
    "warning",
    "environmental_distractor",
    "contextual",
)


def cue_family(category: str, *, relevant: bool) -> str:
    if category in {"turn-left", "turn-right", "ascend", "descend", "road-sign", "junction"}:
        return "directional" if relevant else "environmental_distractor"
    if category in {"tunnel", "industrial-corridor"}:
        return "warning"
    if category.startswith("beacon-"):
        return "contextual"
    return "environmental_distractor"


def _beacon_color(category: str) -> tuple[int, int, int]:
    return BEACON_COLORS.get(category, (160, 160, 160))


def write_beacon_template(path: Path, category: str, label: str) -> None:
    ensure_dir(path.parent)
    img = Image.new("RGB", (64, 64), _beacon_color(category))
    draw = ImageDraw.Draw(img)
    draw.rectangle((6, 6, 58, 58), outline=(255, 255, 255), width=3)
    draw.text((10, 24), label[:10], fill=(0, 0, 0))
    img.save(path)


def generate_beacons(
    scene_id: str,
    scene_type: str,
    decision_points: list[int],
    route_length: int,
    output_root: Path,
    task_beacons_per_route: int,
    background_per_scene: int,
    semantic_bank: list[str],
    rng: random.Random,
    episode_id: str | None = None,
    task_category_allowlist: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (task_beacons, background_beacons)."""
    task_beacons: list[dict[str, Any]] = []
    background_beacons: list[dict[str, Any]] = []
    default_allowlist = {"road-sign", "turn-left", "turn-right", "junction"}
    selected_allowlist = default_allowlist if task_category_allowlist is None else set(task_category_allowlist)
    task_choices = [c for c in semantic_bank if c.startswith("beacon-") or c in selected_allowlist]
    if not task_choices:
        task_choices = ["beacon-blue", "beacon-red", "beacon-yellow"]

    # The terminal route point is reserved for the target. Task beacons are
    # immutable episode annotations: prefer route decision points, then fill
    # deterministically with evenly spaced reference-path indices.
    usable_points = list(range(max(route_length - 1, 0)))
    if len(usable_points) < task_beacons_per_route:
        raise ValueError(
            f"Route has {route_length} states but requires "
            f"{task_beacons_per_route} non-terminal task beacon positions"
        )
    selected_points: list[int] = []
    for value in decision_points:
        index = int(value)
        if index in usable_points and index not in selected_points:
            selected_points.append(index)
        if len(selected_points) == task_beacons_per_route:
            break
    if len(selected_points) < task_beacons_per_route:
        evenly_spaced = [
            min(
                usable_points[-1],
                max(
                    usable_points[0],
                    round((order + 1) * (route_length - 1) / (task_beacons_per_route + 1)),
                ),
            )
            for order in range(task_beacons_per_route)
        ]
        for index in [*evenly_spaced, *usable_points]:
            if index not in selected_points:
                selected_points.append(index)
            if len(selected_points) == task_beacons_per_route:
                break
    selected_points.sort()

    for j, idx in enumerate(selected_points):
        category = task_choices[j % len(task_choices)]
        owner_id = episode_id or scene_id
        beacon_id = f"{owner_id}_task_{j:02d}"
        template_path = output_root / "templates" / "beacons" / f"{beacon_id}.png"
        write_beacon_template(template_path, category, category.replace("beacon-", "B-"))
        task_beacons.append(
            {
                "beacon_id": beacon_id,
                "semantic_type": category,
                "cue_family": cue_family(category, relevant=True),
                "relevant": True,
                "relevance_rule": "episode_route",
                "route_index": idx,
                "visible_at": idx,
                "template_image": template_path.relative_to(output_root.parent).as_posix(),
            }
        )

    scene_categories = DEFAULT_SCENE_CATEGORIES.get(scene_type, task_choices)
    bg_choices = [c for c in scene_categories if c in semantic_bank] or scene_categories
    for j in range(background_per_scene):
        category = bg_choices[j % len(bg_choices)]
        beacon_id = f"{scene_id}_bg_{j:02d}"
        template_path = output_root / "templates" / "beacons" / f"{beacon_id}.png"
        write_beacon_template(template_path, category, category[:8])
        background_beacons.append(
            {
                "beacon_id": beacon_id,
                "semantic_type": category,
                "cue_family": cue_family(category, relevant=False),
                "relevant": False,
                "relevance_rule": "scene_passive",
                "visible_at": int((j + 1) * max(route_length, 1) / (background_per_scene + 1)),
                "template_image": template_path.relative_to(output_root.parent).as_posix(),
            }
        )
    return task_beacons, background_beacons


def generate_scene_background_beacons(
    *,
    scene_id: str,
    scene_type: str,
    scene_states: list[dict[str, Any]],
    output_root: Path,
    count: int,
    semantic_bank: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Create one deterministic passive-beacon layout shared by all scene episodes."""
    if not scene_states or count <= 0:
        return []
    digest = hashlib.sha256(f"{int(seed)}:{scene_id}:passive".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    scene_categories = DEFAULT_SCENE_CATEGORIES.get(scene_type, list(semantic_bank))
    choices = [category for category in scene_categories if category in semantic_bank]
    if not choices:
        choices = list(semantic_bank) or ["road-sign"]

    backgrounds: list[dict[str, Any]] = []
    for index in range(int(count)):
        reference = scene_states[rng.randrange(len(scene_states))]
        position = [float(value) for value in reference["position"][:3]]
        yaw = float(reference.get("yaw", 0.0))
        lateral = rng.uniform(-30.0, 30.0)
        forward = rng.uniform(-20.0, 40.0)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        placement = [
            position[0] + forward * cosine - lateral * sine,
            position[1] + forward * sine + lateral * cosine,
            position[2] + rng.uniform(-5.0, 5.0),
        ]
        category = choices[index % len(choices)]
        beacon_id = f"{scene_id}_passive_{index:03d}"
        template_path = output_root / "templates" / "beacons" / f"{beacon_id}.png"
        write_beacon_template(template_path, category, category[:8])
        backgrounds.append(
            {
                "beacon_id": beacon_id,
                "semantic_type": category,
                "cue_family": cue_family(category, relevant=False),
                "relevant": False,
                "relevance_rule": "scene_passive",
                "position": placement,
                "yaw_rad": yaw + math.pi,
                "template_image": template_path.relative_to(output_root.parent).as_posix(),
            }
        )
    return backgrounds


def background_visibility_for_route(
    backgrounds: list[dict[str, Any]],
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach route-local nearest indices without changing fixed world placements."""
    mapped: list[dict[str, Any]] = []
    for beacon in backgrounds:
        nearest = min(
            range(len(states)),
            key=lambda index: sum(
                (
                    float(states[index]["position"][axis])
                    - float(beacon["position"][axis])
                )
                ** 2
                for axis in range(3)
            ),
        )
        mapped.append({**beacon, "visible_at": nearest})
    return mapped


def visible_beacon_labels(step_index: int, task_beacons: list[dict[str, Any]], background_beacons: list[dict[str, Any]], visibility_window: int = 1) -> dict[str, Any]:
    visible: list[dict[str, Any]] = []
    for beacon in task_beacons + background_beacons:
        if abs(int(beacon["visible_at"]) - step_index) <= visibility_window:
            visible.append(
                {
                    "beacon_id": beacon["beacon_id"],
                    "semantic_type": beacon["semantic_type"],
                    "relevant": bool(beacon["relevant"]),
                    "visible": True,
                    "template_image": beacon["template_image"],
                }
            )
    return {
        "visible": bool(visible),
        "items": visible,
    }
