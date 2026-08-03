import json

from scripts.sync_index_release import sync_index_release


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sync_index_uses_full_release_as_canonical_source(tmp_path):
    full = tmp_path / "full"
    index = tmp_path / "index"
    route = {"trajectory_id": "Forest_hard_0001", "difficulty": "Hard"}
    _write_json(full / "source" / "preset_routes" / "Forest_hard_0001.json", route)
    (full / "source" / "custom_routes").mkdir(parents=True)
    _write_jsonl(full / "source" / "scenes.jsonl", [{"scene_id": "Forest"}])
    episode = {"episode_id": "Forest_hard_0001", "trajectory_id": "Forest_hard_0001"}
    _write_jsonl(full / "metadata" / "episodes.jsonl", [episode])
    _write_jsonl(
        full / "metadata" / "source_data_index.jsonl",
        [{**episode, "asset_mode": "copy"}],
    )
    _write_jsonl(full / "splits" / "test.jsonl", [episode])
    _write_json(full / "manifest.json", {"num_scenes": 99, "asset_mode": "copy"})

    _write_json(index / "source" / "preset_routes" / "old.json", {"trajectory_id": "old"})
    (index / "source" / "custom_routes").mkdir(parents=True)
    _write_jsonl(index / "source" / "scenes.jsonl", [{"scene_id": "Old"}])
    _write_jsonl(index / "metadata" / "episodes.jsonl", [{"episode_id": "old"}])
    _write_json(index / "manifest.json", {"asset_mode": "index"})

    report = sync_index_release(
        full,
        index,
        apply=True,
        backup_root=tmp_path / "backup",
    )

    assert report["route_count"] == 1
    assert (index / "source" / "preset_routes" / "Forest_hard_0001.json").exists()
    assert not (index / "source" / "preset_routes" / "old.json").exists()
    source_row = json.loads((index / "metadata" / "source_data_index.jsonl").read_text())
    assert source_row["asset_mode"] == "index"
    manifest = json.loads((index / "manifest.json").read_text())
    assert manifest["num_scenes"] == 1
    assert manifest["asset_mode"] == "index"
