import random

from voln_uav.benchmark.beacon_augmentation import (
    background_visibility_for_route,
    generate_beacons,
    generate_scene_background_beacons,
)


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

    assert len(task_beacons) == 3
    assert [item["route_index"] for item in task_beacons] == [2, 5, 9]
    assert bg_beacons
    assert all(item["semantic_type"] in semantic_bank for item in task_beacons)
    assert all(item["semantic_type"] in semantic_bank for item in bg_beacons)
    assert any(item["semantic_type"] == "junction" for item in bg_beacons)


def test_generate_beacons_deterministically_fills_missing_decision_points(tmp_path):
    first, _ = generate_beacons(
        scene_id="scene_fill",
        scene_type="urban",
        decision_points=[4],
        route_length=12,
        output_root=tmp_path,
        task_beacons_per_route=4,
        background_per_scene=0,
        semantic_bank=["road-sign"],
        rng=random.Random(7),
    )
    second, _ = generate_beacons(
        scene_id="scene_fill",
        scene_type="urban",
        decision_points=[4],
        route_length=12,
        output_root=tmp_path,
        task_beacons_per_route=4,
        background_per_scene=0,
        semantic_bank=["road-sign"],
        rng=random.Random(999),
    )

    assert [item["route_index"] for item in first] == [
        item["route_index"] for item in second
    ]
    assert len(first) == 4
    assert len({item["route_index"] for item in first}) == 4


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


def test_passive_beacon_layout_is_scene_fixed_across_routes(tmp_path):
    scene_states = [
        {"position": [float(index), 0.0, -10.0], "yaw": 0.0}
        for index in range(10)
    ]
    first = generate_scene_background_beacons(
        scene_id="forest",
        scene_type="forest",
        scene_states=scene_states,
        output_root=tmp_path,
        count=6,
        semantic_bank=["beacon-green", "turn-left", "forest-trail"],
        seed=7,
    )
    second = generate_scene_background_beacons(
        scene_id="forest",
        scene_type="forest",
        scene_states=scene_states,
        output_root=tmp_path,
        count=6,
        semantic_bank=["beacon-green", "turn-left", "forest-trail"],
        seed=7,
    )

    assert first == second
    assert all(item["relevance_rule"] == "scene_passive" for item in first)
    route_a = background_visibility_for_route(first, scene_states[:5])
    route_b = background_visibility_for_route(first, scene_states[5:])
    assert [item["position"] for item in route_a] == [item["position"] for item in route_b]
