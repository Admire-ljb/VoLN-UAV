import json

from voln_uav.data.release_packager import prepare_dataset_release


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_frame(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")


def test_prepare_dataset_release_index_mode(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_c = tmp_path / "source-c"

    ep = source_a / "BattlefieldKitDesert" / "route_001"
    _write_json(
        ep / "log" / "000001.json",
        {
            "step": 1,
            "timestamp": 1.0,
            "sensors": {
                "state": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "collision": {"has_collided": False},
                }
            },
        },
    )
    _write_json(
        ep / "log" / "000002.json",
        {
            "step": 2,
            "timestamp": 2.0,
            "sensors": {
                "state": {
                    "position": [1.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "collision": {"has_collided": False},
                }
            },
        },
    )
    _write_frame(ep / "FrontCamera" / "000001.png")
    _write_frame(ep / "FrontCamera" / "000002.png")

    normal_ep = source_b / "BrushifyUrban" / "route_001"
    _write_json(
        normal_ep / "log" / "000005.json",
        {
            "step": 5,
            "sensors": {
                "state": {
                    "position": [10.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.1, 0.995],
                }
            },
        },
    )
    _write_json(
        normal_ep / "log" / "000010.json",
        {
            "step": 10,
            "sensors": {
                "state": {
                    "position": [11.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.1, 0.995],
                }
            },
        },
    )
    _write_frame(normal_ep / "FrontCamera" / "000005.png")
    _write_frame(normal_ep / "FrontCamera" / "000010.png")

    hard_ep = source_c / "route_without_metadata" / "front_camera"
    _write_frame(hard_ep / "frame_00001.png")
    _write_frame(hard_ep / "frame_00002.png")

    out_root = tmp_path / "release"
    zip_path = tmp_path / "release.zip"
    summary = prepare_dataset_release(
        source_roots=[source_a, source_b, source_c],
        out_root=out_root,
        dataset_url="https://huggingface.co/datasets/Louj/VoLN-UAV-Dataset",
        env_url="https://huggingface.co/datasets/Louj/VoLN-UAV-ENV",
        asset_mode="index",
        zip_path=zip_path,
    )

    assert summary["num_routes"] == 2
    assert summary["num_skipped"] == 1
    assert zip_path.exists()
    assert (out_root / "source" / "scenes.jsonl").exists()
    assert (out_root / "metadata" / "source_data_index.jsonl").exists()
    routes = list((out_root / "source" / "preset_routes").glob("*.json"))
    assert routes
    route = json.loads(routes[0].read_text(encoding="utf-8"))
    assert route["states"][0]["image"]
    skipped = (out_root / "metadata" / "skipped.jsonl").read_text(encoding="utf-8")
    assert "incomplete trajectory metadata" in skipped
