import json

from scripts.rebalance_scene_difficulty import rebalance_release


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rebalance_marks_longest_routes_hard(tmp_path):
    release = tmp_path / "release"
    rows = []
    for index, length in enumerate((100.0, 200.0, 300.0, 400.0, 500.0)):
        trajectory_id = f"route_{index}"
        route = {
            "scene_id": "Forest",
            "trajectory_id": trajectory_id,
            "difficulty": "Easy",
            "difficulty_source": "reference_path_length",
            "path_length": length,
            "states": [],
        }
        _write_json(release / "source" / "preset_routes" / f"{trajectory_id}.json", route)
        rows.append(
            {
                "scene_id": "Forest",
                "trajectory_id": trajectory_id,
                "difficulty": "Easy",
                "path_length": length,
            }
        )
    _write_jsonl(release / "metadata" / "episodes.jsonl", rows)
    _write_jsonl(release / "metadata" / "source_data_index.jsonl", rows)
    _write_jsonl(release / "splits" / "test.jsonl", rows)
    _write_json(
        release / "manifest.json",
        {
            "difficulty_thresholds_m": {"easy_lt": 300.0, "normal_lt": 450.0},
            "episodes_by_difficulty": {"Easy": 5},
        },
    )

    report = rebalance_release(
        release,
        scene_id="Forest",
        hard_count=2,
        hard_min_difficulty_length_m=800.0,
        apply=True,
        backup_root=tmp_path / "backup",
    )

    assert report["scene_difficulty_counts"] == {"Easy": 2, "Hard": 2, "Normal": 1}
    assert report["hard_min_path_length_m"] == 400.0
    assert report["next_path_length_m"] == 300.0
    assert report["difficulty_length_scale"] == 2.0
    assert report["hard_min_difficulty_length_m"] == 800.0
    for index in (3, 4):
        route = json.loads(
            (release / "source" / "preset_routes" / f"route_{index}.json").read_text()
        )
        assert route["difficulty"] == "Hard"
        assert route["difficulty_source"] == "scene_path_length_rank"
        assert route["difficulty_length_m"] == route["path_length"] * 2.0
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["scene_difficulty_rules"]["Forest"]["hard_count"] == 2
    assert (
        manifest["scene_difficulty_rules"]["Forest"]["hard_min_difficulty_length_m"]
        == 800.0
    )
