import json

from scripts.canonicalize_release_names import canonicalize_release


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_canonicalize_release_updates_ids_paths_and_sources(tmp_path):
    release = tmp_path / "release"
    old_id = "normal_Forest_normal_52"
    route = {
        "scene_id": "Forest",
        "trajectory_id": old_id,
        "source": "preset",
        "source_root": "normal",
        "source_episode": "Forest/normal_52",
        "difficulty": "Hard",
        "path_length": 500.0,
        "states": [
            {
                "position": [0.0, 0.0, 0.0],
                "image": f"source/frames/Forest/{old_id}/000000.png",
                "raw": {},
            }
        ],
    }
    _write_json(release / "source" / "preset_routes" / f"{old_id}.json", route)
    frame = release / "source" / "frames" / "Forest" / old_id / "000000.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")
    episode = {
        "episode_id": f"Forest_{old_id}",
        "scene_id": "Forest",
        "trajectory_id": old_id,
        "source": "preset",
        "difficulty": "Hard",
        "route_file": f"source/preset_routes/{old_id}.json",
    }
    _write_jsonl(release / "metadata" / "episodes.jsonl", [episode])
    _write_jsonl(
        release / "metadata" / "source_data_index.jsonl",
        [{**episode, "source_root": "normal", "source_episode": "Forest/normal_52"}],
    )
    _write_jsonl(release / "splits" / "test.jsonl", [episode])
    _write_json(release / "manifest.json", {})

    report = canonicalize_release(
        release,
        apply=True,
        backup_root=tmp_path / "backup",
    )

    new_id = "Forest_hard_0001"
    assert report["mismatches_before"] == 1
    new_route_path = release / "source" / "preset_routes" / f"{new_id}.json"
    assert new_route_path.exists()
    assert not (release / "source" / "preset_routes" / f"{old_id}.json").exists()
    new_route = json.loads(new_route_path.read_text())
    assert new_route["trajectory_id"] == new_id
    assert new_route["source_root"] == "hard"
    assert new_route["source_episode"] == "Forest/hard_0001"
    assert new_route["states"][0]["image"] == f"source/frames/Forest/{new_id}/000000.png"
    assert "original_image" not in new_route["states"][0]["raw"]
    assert new_route["provenance"]["original_trajectory_id"] == old_id
    assert (release / "source" / "frames" / "Forest" / new_id / "000000.png").exists()

    episode_after = json.loads((release / "metadata" / "episodes.jsonl").read_text().strip())
    assert episode_after["episode_id"] == new_id
    assert episode_after["trajectory_id"] == new_id
    assert episode_after["route_file"] == f"source/preset_routes/{new_id}.json"
