from __future__ import annotations

from pathlib import Path

import pytest

from voln_uav.common.io import write_jsonl
from voln_uav.evaluation.paper_protocol import (
    inspect_benchmark_protocol,
    load_paper_protocol,
    select_available_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_missing_optional_scene_is_skipped_not_scored() -> None:
    episodes = [
        {"episode_id": "a", "scene_id": "Forest", "difficulty": "Easy"},
        {"episode_id": "b", "scene_id": "Forest", "difficulty": "Hard"},
    ]
    selected, coverage = select_available_episodes(
        episodes,
        {
            "scene_allowlist": ["Forest", "Campus"],
            "optional_scene_ids": ["Campus"],
            "strict_scenes": False,
        },
    )

    assert [episode["episode_id"] for episode in selected] == ["a", "b"]
    assert coverage["missing_optional_scenes"] == ["Campus"]
    assert coverage["missing_required_scenes"] == []
    assert coverage["selected_episodes"] == 2


def test_only_missing_optional_scene_produces_empty_selection() -> None:
    selected, coverage = select_available_episodes(
        [{"episode_id": "a", "scene_id": "Forest"}],
        {
            "scene_allowlist": ["Tunnel"],
            "optional_scene_ids": ["Tunnel"],
            "strict_scenes": False,
        },
    )

    assert selected == []
    assert coverage["selected_episodes"] == 0
    assert coverage["missing_optional_scenes"] == ["Tunnel"]


def test_strict_scene_selection_rejects_missing_scene() -> None:
    with pytest.raises(ValueError, match="Campus"):
        select_available_episodes(
            [{"episode_id": "a", "scene_id": "Forest"}],
            {"scene_allowlist": ["Campus"], "strict_scenes": True},
        )


def test_protocol_inspection_accepts_seen_validation_and_unseen_test(tmp_path: Path) -> None:
    write_jsonl(
        [{"episode_id": "train-a", "scene_id": "Forest", "difficulty": "Easy"}],
        tmp_path / "train.jsonl",
    )
    write_jsonl(
        [{"episode_id": "val-a", "scene_id": "Forest", "difficulty": "Normal"}],
        tmp_path / "val.jsonl",
    )
    write_jsonl(
        [{"episode_id": "test-a", "scene_id": "City", "difficulty": "Hard"}],
        tmp_path / "test.jsonl",
    )

    report = inspect_benchmark_protocol(
        tmp_path,
        load_paper_protocol(ROOT / "configs" / "paper_protocol.yaml"),
    )

    assert report["issues"] == []
    assert report["status"] == "partial"
    assert report["total_episodes"] == 3
    assert report["splits"]["validation_seen"]["scenes"] == ["Forest"]
    assert report["splits"]["test_unseen"]["scenes"] == ["City"]


def test_protocol_inspection_detects_test_scene_leakage(tmp_path: Path) -> None:
    for name in ("train", "val", "test"):
        write_jsonl(
            [{"episode_id": name, "scene_id": "Forest", "difficulty": "Easy"}],
            tmp_path / f"{name}.jsonl",
        )

    report = inspect_benchmark_protocol(
        tmp_path,
        load_paper_protocol(ROOT / "configs" / "paper_protocol.yaml"),
    )

    assert any("Test-Unseen overlaps" in issue for issue in report["issues"])
