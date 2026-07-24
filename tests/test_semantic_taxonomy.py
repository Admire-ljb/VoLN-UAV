from voln_uav.common.config import load_config


def test_release_benchmark_configs_use_current_labels():
    for path in (
        "configs/benchmark_dataset_release.yaml",
        "configs/benchmark_library_update.yaml",
    ):
        cfg = load_config(path)
        categories = set(cfg["semantic_bank"]["categories"])
        assert "person" not in categories
        assert "corridor" not in categories
        assert "intersection" not in categories
        assert "industrial-corridor" in categories
        assert "junction" in categories


def test_beacon_task_allowlist_is_explicit():
    for path in (
        "configs/benchmark_dataset_release.yaml",
        "configs/benchmark_library_update.yaml",
    ):
        cfg = load_config(path)
        assert "task_category_allowlist" in cfg["beacons"]
        assert "junction" in set(cfg["beacons"]["task_category_allowlist"])
