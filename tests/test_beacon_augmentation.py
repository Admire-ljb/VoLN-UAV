import random

from voln_uav.benchmark.beacon_augmentation import generate_beacons


def test_generate_beacons_uses_current_semantic_labels(tmp_path):
    semantic_bank = [
        "beacon-blue",
        "beacon-red",
        "road-sign",
        "junction",
        "urban-canyon",
        "industrial-corridor",
    ]
    task_beacons, bg_beacons = generate_beacons(
        scene_id="scene_001",
        scene_type="urban",
        decision_points=[2, 5, 9],
        route_length=12,
        output_root=tmp_path,
        task_beacons_per_route=3,
        background_per_scene=5,
        semantic_bank=semantic_bank,
        rng=random.Random(7),
    )

    assert task_beacons
    assert bg_beacons
    assert all(item["semantic_type"] in semantic_bank for item in task_beacons)
    assert all(item["semantic_type"] in semantic_bank for item in bg_beacons)
    assert any(item["semantic_type"] == "junction" for item in bg_beacons)


def test_generate_beacons_respects_task_allowlist(tmp_path):
    semantic_bank = ["vehicle", "ascend", "beacon-blue", "junction"]
    task_beacons, _ = generate_beacons(
        scene_id="scene_002",
        scene_type="urban",
        decision_points=[1, 3, 6],
        route_length=10,
        output_root=tmp_path,
        task_beacons_per_route=3,
        background_per_scene=2,
        semantic_bank=semantic_bank,
        rng=random.Random(11),
        task_category_allowlist=["ascend"],
    )

    task_types = {item["semantic_type"] for item in task_beacons}
    assert "vehicle" not in task_types
    assert task_types.issubset({"beacon-blue", "ascend"})


def test_generate_beacons_empty_allowlist_disables_non_beacon_categories(tmp_path):
    semantic_bank = ["ascend", "junction"]
    task_beacons, _ = generate_beacons(
        scene_id="scene_003",
        scene_type="urban",
        decision_points=[1, 2],
        route_length=5,
        output_root=tmp_path,
        task_beacons_per_route=2,
        background_per_scene=1,
        semantic_bank=semantic_bank,
        rng=random.Random(5),
        task_category_allowlist=[],
    )
    task_types = {item["semantic_type"] for item in task_beacons}
    assert task_types == {"beacon-blue", "beacon-red"}
