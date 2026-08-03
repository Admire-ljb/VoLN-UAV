from __future__ import annotations

import json
from pathlib import Path

from voln_uav.benchmark.builder import BenchmarkBuilder


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_release_builder_keeps_canonical_id_and_deduplicates_before_templates(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "benchmark"
    source.mkdir(parents=True)
    (source / "scenes.jsonl").write_text(
        json.dumps({"scene_id": "Scene", "scene_type": "urban"}) + "\n",
        encoding="utf-8",
    )
    states = [
        {
            "t": index,
            "position": [float(index), 0.0, 0.0],
            "yaw": 0.0,
            "image": f"source/frames/Scene/frame_{index:03d}.png",
            "imu": [0.0] * 6,
            "odometry": [float(index), 0.0, 0.0],
        }
        for index in range(12)
    ]
    for ordinal in (1, 2):
        _write_json(
            source / "preset_routes" / f"Scene_easy_{ordinal:04d}.json",
            {
                "scene_id": "Scene",
                "trajectory_id": f"Scene_easy_{ordinal:04d}",
                "split": "train",
                "difficulty": "Easy",
                "states": states,
            },
        )
    cfg = {
        "seed": 7,
        "source_root": str(source),
        "output_root": str(output),
        "scene_manifest": "scenes.jsonl",
        "preset_routes_dir": "preset_routes",
        "custom_routes_dir": "custom_routes",
        "horizon": 2,
        "strict_paper_protocol": False,
        "require_shortest_path": False,
        "difficulty": {"easy_lt": 300.0, "normal_lt": 450.0},
        "dedup": {"start_threshold": 1.0, "goal_threshold": 1.0},
        "goal_interface": {"num_terminal_views": 3, "num_subgoals": 4, "num_beacons": 2},
        "beacons": {
            "count_by_path_length": {
                "easy_lt_m": 300.0,
                "normal_lt_m": 450.0,
                "easy": 3,
                "normal": 4,
                "hard": 5,
            },
            "background_per_scene": 0,
            "min_separation_steps": 8,
            "task_category_allowlist": ["road-sign", "turn-left", "turn-right"],
        },
        "semantic_bank": {"categories": ["road-sign", "turn-left", "turn-right"]},
    }
    summary = BenchmarkBuilder(cfg).build()
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert summary["num_episodes"] == 1
    assert rows[0]["episode_id"] == "Scene_easy_0001"
    assert len(rows[0]["task_beacons"]) == 3
    assert all(not Path(item["template_image"]).is_absolute() for item in rows[0]["task_beacons"])
    assert not list((output / "templates").rglob("*Scene_easy_0002*"))
