import ast
from pathlib import Path

from voln_uav.common.config import load_config


def _load_goal_categories() -> list[str]:
    source = Path("examples/generate_toy_source.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GOAL_CATEGORIES":
                    return [elt.value for elt in node.value.elts]  # type: ignore[attr-defined]
    raise AssertionError("GOAL_CATEGORIES not found")


def test_toy_goal_categories_use_current_labels():
    goal_categories = _load_goal_categories()
    assert "person" not in goal_categories
    assert "corridor" not in goal_categories
    assert "human" in goal_categories
    assert "industrial-corridor" in goal_categories


def test_toy_benchmark_config_uses_current_labels():
    cfg = load_config("configs/benchmark_toy.yaml")
    categories = set(cfg["semantic_bank"]["categories"])
    assert "person" not in categories
    assert "corridor" not in categories
    assert "intersection" not in categories
    assert "human" in categories
    assert "industrial-corridor" in categories
    assert "junction" in categories


def test_beacon_task_allowlist_is_explicit():
    toy_cfg = load_config("configs/benchmark_toy.yaml")
    lib_cfg = load_config("configs/benchmark_library_update.yaml")
    assert "task_category_allowlist" in toy_cfg["beacons"]
    assert "task_category_allowlist" in lib_cfg["beacons"]
    assert "junction" in set(lib_cfg["beacons"]["task_category_allowlist"])
