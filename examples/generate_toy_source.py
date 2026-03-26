from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from voln_uav.common.io import ensure_dir, write_json, write_jsonl


SCENES = [
    ("urban_001", "urban", (90, 110, 160)),
    ("urban_002", "urban", (70, 100, 180)),
    ("forest_001", "forest", (70, 140, 90)),
    ("forest_002", "forest", (90, 160, 80)),
    ("tunnel_001", "tunnel", (120, 110, 100)),
    ("tunnel_002", "tunnel", (140, 120, 90)),
]
GOAL_CATEGORIES = ["vehicle", "human", "entrance", "industrial-corridor", "urban-canyon", "tunnel"]


def make_segments(scene_type: str, variant: str) -> list[tuple[tuple[float, float, float], int]]:
    if scene_type == "urban":
        if variant == "easy":
            return [((2.0, 0.0, 0.0), 8), ((0.0, 2.0, 0.0), 7)]
        if variant == "normal":
            return [((2.0, 0.0, 0.0), 10), ((0.0, 2.0, 0.0), 10)]
        return [((2.0, 0.0, 0.0), 10), ((0.0, 2.0, 0.0), 10), ((2.0, 0.0, 1.0), 10)]
    if scene_type == "forest":
        if variant == "easy":
            return [((1.5, 1.0, 0.0), 10), ((1.0, 1.2, 0.0), 6)]
        if variant == "normal":
            return [((1.5, 1.0, 0.0), 10), ((0.8, 1.4, 0.5), 8), ((1.2, -0.5, 0.0), 4)]
        return [((1.5, 1.0, 0.0), 10), ((0.8, 1.4, 0.5), 8), ((1.2, -0.5, 0.0), 6), ((1.0, 1.0, 0.0), 8)]
    if variant == "easy":
        return [((2.0, 0.2, 0.0), 8), ((2.0, -0.2, 0.0), 7)]
    if variant == "normal":
        return [((2.0, 0.2, 0.0), 8), ((2.0, -0.2, 0.0), 7), ((2.0, 0.1, 0.8), 5)]
    return [((2.0, 0.2, 0.0), 8), ((2.0, -0.2, 0.0), 7), ((2.0, 0.1, 0.8), 5), ((2.0, -0.1, 0.0), 8)]


def yaw_from_delta(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)



def render_frame(path: Path, scene_id: str, scene_type: str, goal_category: str, base_color: tuple[int, int, int], pos: tuple[float, float, float], t: int, total: int) -> None:
    ensure_dir(path.parent)
    img = Image.new("RGB", (96, 96), base_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, 92, 92), outline=(255, 255, 255), width=2)
    progress = int(84 * (t / max(total - 1, 1)))
    draw.rectangle((6, 78, 6 + progress, 88), fill=(255, 255, 255))
    draw.text((8, 8), scene_type, fill=(0, 0, 0))
    draw.text((8, 24), goal_category[:10], fill=(0, 0, 0))
    draw.text((8, 40), f"z={pos[2]:.1f}", fill=(0, 0, 0))
    if scene_type == "urban":
        draw.rectangle((60, 18, 84, 54), fill=(50, 50, 60))
        draw.rectangle((66, 8, 76, 18), fill=(200, 40, 40))
    elif scene_type == "forest":
        draw.polygon([(68, 20), (56, 48), (80, 48)], fill=(20, 110, 20))
        draw.rectangle((64, 48, 72, 64), fill=(80, 40, 20))
    else:
        draw.rectangle((52, 24, 84, 60), fill=(80, 80, 80))
        draw.rectangle((56, 34, 80, 50), fill=(230, 220, 100))
    img.save(path)



def build_route(repo_root: Path, out_dir: Path, scene_id: str, scene_type: str, base_color: tuple[int, int, int], goal_category: str, trajectory_id: str, variant: str, start_offset: tuple[float, float, float]) -> dict:
    segments = make_segments(scene_type, variant)
    frames_dir = out_dir / "frames" / scene_id / trajectory_id
    pos = list(start_offset)
    states = []
    t = 0
    total_steps = sum(seg_steps for _, seg_steps in segments) + 1
    for seg_delta, seg_steps in segments:
        for _ in range(seg_steps):
            dx, dy, dz = seg_delta
            if states:
                pos[0] += dx
                pos[1] += dy
                pos[2] += dz
            img_path = frames_dir / f"{t:04d}.png"
            render_frame(
                img_path,
                scene_id=scene_id,
                scene_type=scene_type,
                goal_category=goal_category,
                base_color=base_color,
                pos=(pos[0], pos[1], pos[2]),
                t=t,
                total=total_steps,
            )
            imu = [dx, dy, dz, 0.0, 0.0, 0.0]
            state = {
                "t": t,
                "position": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
                "yaw": round(yaw_from_delta(dx, dy), 4),
                "image": str(img_path.relative_to(repo_root)),
                "imu": imu,
                "odometry": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
            }
            states.append(state)
            t += 1
    # final frame to ensure terminal view diversity
    img_path = frames_dir / f"{t:04d}.png"
    render_frame(
        img_path,
        scene_id=scene_id,
        scene_type=scene_type,
        goal_category=goal_category,
        base_color=tuple(min(255, c + 15) for c in base_color),
        pos=(pos[0], pos[1], pos[2]),
        t=t,
        total=total_steps,
    )
    states.append(
        {
            "t": t,
            "position": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
            "yaw": states[-1]["yaw"] if states else 0.0,
            "image": str(img_path.relative_to(repo_root)),
            "imu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "odometry": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
        }
    )
    return {
        "scene_id": scene_id,
        "trajectory_id": trajectory_id,
        "goal_category": goal_category,
        "states": states,
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/toy_source")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = ensure_dir(repo_root / args.out_dir)
    preset_dir = ensure_dir(out_dir / "preset_routes")
    custom_dir = ensure_dir(out_dir / "custom_routes")

    scenes = []
    for i, (scene_id, scene_type, base_color) in enumerate(SCENES):
        scenes.append({"scene_id": scene_id, "scene_type": scene_type, "scene_index": i})
        goal_category = GOAL_CATEGORIES[i % len(GOAL_CATEGORIES)]
        easy_route = build_route(repo_root, out_dir, scene_id, scene_type, base_color, goal_category, f"preset_{i:02d}_easy", "easy", start_offset=(i * 3.0, i * 2.0, 5.0))
        normal_route = build_route(repo_root, out_dir, scene_id, scene_type, base_color, goal_category, f"preset_{i:02d}_normal", "normal", start_offset=(i * 3.0 + 4.0, i * 2.0 + 1.0, 5.0))
        hard_route = build_route(repo_root, out_dir, scene_id, scene_type, base_color, goal_category, f"custom_{i:02d}_hard", "hard", start_offset=(i * 3.0 + 8.0, i * 2.0 + 2.0, 5.0))
        write_json(easy_route, preset_dir / f"{easy_route['trajectory_id']}.json")
        write_json(normal_route, preset_dir / f"{normal_route['trajectory_id']}.json")
        write_json(hard_route, custom_dir / f"{hard_route['trajectory_id']}.json")

    write_jsonl(scenes, out_dir / "scenes.jsonl")
    print(f"Toy source written to {out_dir}")


if __name__ == "__main__":
    main()
