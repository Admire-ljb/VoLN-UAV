import json

from scripts.repair_route_discontinuities import repair_release


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repair_trims_suffix_and_promotes_last_forest_state_to_goal(tmp_path):
    release = tmp_path / "release"
    trajectory_id = "normal_BrushifyForestPack_normal_52"
    route_path = release / "source" / "preset_routes" / f"{trajectory_id}.json"
    route = {
        "scene_id": "BrushifyForestPack",
        "trajectory_id": trajectory_id,
        "difficulty": "Hard",
        "path_length": 1008.0,
        "states": [
            {"position": [0.0, 0.0, 0.0], "image": "source/frames/forest/000000.png"},
            {"position": [4.0, 0.0, 0.0], "image": "source/frames/forest/000005.png"},
            {"position": [1004.0, 0.0, 0.0], "image": "source/frames/forest/000010.png"},
            {"position": [1008.0, 0.0, 0.0], "image": "source/frames/forest/000015.png"},
        ],
    }
    _write_json(route_path, route)
    for name in ("000000.png", "000005.png", "000010.png", "000015.png"):
        frame = release / "source" / "frames" / "forest" / name
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(name.encode())

    episode = {
        "scene_id": "BrushifyForestPack",
        "trajectory_id": trajectory_id,
        "difficulty": "Hard",
        "path_length": 1008.0,
        "num_states": 4,
    }
    _write_jsonl(release / "metadata" / "episodes.jsonl", [episode])
    _write_jsonl(release / "metadata" / "source_data_index.jsonl", [episode])
    _write_jsonl(release / "splits" / "test.jsonl", [episode])
    _write_json(
        release / "manifest.json",
        {
            "difficulty_thresholds_m": {"easy_lt": 300.0, "normal_lt": 450.0},
            "episodes_by_difficulty": {"Hard": 1},
        },
    )

    report = repair_release(
        release,
        scene_id="BrushifyForestPack",
        difficulty="Hard",
        max_step_m=50.0,
        apply=True,
        backup_root=tmp_path / "backup",
    )

    repaired = json.loads(route_path.read_text(encoding="utf-8"))
    assert report["repaired_routes"] == 1
    assert report["removed_assets"] == 2
    assert repaired["difficulty"] == "Easy"
    assert repaired["path_length"] == 4.0
    assert len(repaired["states"]) == 2
    assert repaired["states"][-1]["image"].endswith("000005.png")
    assert not (release / "source" / "frames" / "forest" / "000010.png").exists()
    assert (tmp_path / "backup" / "removed_assets" / "source" / "frames" / "forest" / "000010.png").exists()

    episode_after = json.loads((release / "metadata" / "episodes.jsonl").read_text().strip())
    assert episode_after["difficulty"] == "Easy"
    assert episode_after["num_states"] == 2
    assert json.loads((release / "manifest.json").read_text())["episodes_by_difficulty"] == {"Easy": 1}


def test_repair_dry_run_does_not_modify_release(tmp_path):
    release = tmp_path / "release"
    route_path = release / "source" / "preset_routes" / "route.json"
    route = {
        "scene_id": "BrushifyForestPack",
        "trajectory_id": "route",
        "difficulty": "Hard",
        "path_length": 1004.0,
        "states": [
            {"position": [0.0, 0.0, 0.0], "image": "a.png"},
            {"position": [4.0, 0.0, 0.0], "image": "b.png"},
            {"position": [1004.0, 0.0, 0.0], "image": "c.png"},
        ],
    }
    _write_json(route_path, route)
    _write_json(release / "manifest.json", {"difficulty_thresholds_m": {}})
    before = route_path.read_text(encoding="utf-8")

    report = repair_release(
        release,
        scene_id="BrushifyForestPack",
        difficulty=None,
        max_step_m=50.0,
        apply=False,
    )

    assert report["repaired_routes"] == 1
    assert route_path.read_text(encoding="utf-8") == before


def test_repair_splits_on_timestamp_reversal_without_large_jump(tmp_path):
    release = tmp_path / "release"
    route_path = release / "source" / "preset_routes" / "route.json"
    route = {
        "scene_id": "Urban",
        "trajectory_id": "route",
        "difficulty": "Easy",
        "path_length": 8.0,
        "states": [
            {"position": [0.0, 0.0, 0.0], "image": "a.png", "raw": {"timestamp": 20}},
            {"position": [4.0, 0.0, 0.0], "image": "b.png", "raw": {"timestamp": 10}},
            {"position": [8.0, 0.0, 0.0], "image": "c.png", "raw": {"timestamp": 11}},
        ],
    }
    _write_json(route_path, route)
    _write_json(release / "manifest.json", {"difficulty_thresholds_m": {}})
    report = repair_release(
        release,
        scene_id=None,
        difficulty=None,
        max_step_m=20.0,
        apply=False,
    )
    assert report["repaired_routes"] == 1
    assert report["repairs"][0]["retained_start_index"] == 1
