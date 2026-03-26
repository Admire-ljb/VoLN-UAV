from __future__ import annotations

import random
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

    selected_points = decision_points[:task_beacons_per_route]
    if len(selected_points) < task_beacons_per_route and route_length > 0:
        needed = task_beacons_per_route - len(selected_points)
        supplement = sorted(rng.sample(list(range(max(1, route_length - 1))), k=min(needed, max(1, route_length - 1))))
        for idx in supplement:
            if idx not in selected_points:
                selected_points.append(idx)
        selected_points = sorted(selected_points)[:task_beacons_per_route]

    for j, idx in enumerate(selected_points):
        category = task_choices[j % len(task_choices)]
        beacon_id = f"{scene_id}_task_{j:02d}"
        template_path = output_root / "templates" / "beacons" / f"{beacon_id}.png"
        write_beacon_template(template_path, category, category.replace("beacon-", "B-"))
        task_beacons.append(
            {
                "beacon_id": beacon_id,
                "semantic_type": category,
                "relevant": True,
                "visible_at": idx,
                "template_image": str(template_path),
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
                "relevant": False,
                "visible_at": int((j + 1) * max(route_length, 1) / (background_per_scene + 1)),
                "template_image": str(template_path),
            }
        )
    return task_beacons, background_beacons


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
