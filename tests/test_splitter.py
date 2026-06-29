from voln_uav.benchmark.splitter import assign_scene_splits, split_counts


def test_split_counts_keeps_validation_when_possible():
    counts = split_counts(5, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert counts == {"train": 3, "val": 1, "test": 1}


def test_assign_scene_splits_scene_level_nonempty():
    scene_ids = [f"scene_{i}" for i in range(5)]
    split_map = assign_scene_splits(scene_ids, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    assigned = set(split_map.values())
    assert assigned == {"train", "val", "test"}
    assert set(split_map) == set(scene_ids)
