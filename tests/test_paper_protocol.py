from __future__ import annotations

from pathlib import Path

import pytest

from voln_uav.common.io import write_jsonl
from voln_uav.evaluation.paper_protocol import (
    inspect_benchmark_protocol,
    load_paper_protocol,
    require_full_paper_split_selection,
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

    assert report["issues"]
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


def test_protocol_ready_requires_episode_environment_and_source_invariants(tmp_path: Path) -> None:
    def episode(episode_id: str, scene_id: str, scene_source: str, offset: float) -> dict:
        return {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "scene_source": scene_source,
            "difficulty": "Easy",
            "states": [
                {"position": [offset, 0.0, 0.0], "yaw": 0.0},
                {"position": [offset + 1.0, 0.0, 0.0], "yaw": 0.0},
            ],
        }

    write_jsonl([episode("train", "Forest", "source-a", 0.0)], tmp_path / "train.jsonl")
    write_jsonl([episode("val", "Forest", "source-a", 10.0)], tmp_path / "val.jsonl")
    write_jsonl([episode("test", "City", "source-b", 20.0)], tmp_path / "test.jsonl")
    protocol = {
        "name": "unit",
        "dataset": {
            "expected_episodes": 3,
            "expected_environments": 2,
            "difficulty_mix": {},
        },
        "splits": {
            "require_held_out_scene_source": True,
            "train": {"file": "train.jsonl", "expected_episodes": 1, "expected_environments": 1},
            "validation_seen": {"file": "val.jsonl", "expected_episodes": 1, "expected_environments": 1},
            "test_unseen": {"file": "test.jsonl", "expected_episodes": 1, "expected_environments": 1},
        },
    }

    report = inspect_benchmark_protocol(tmp_path, protocol)

    assert report["status"] == "ready"
    assert report["issues"] == []


def test_strict_evaluation_requires_the_complete_selected_split(tmp_path: Path) -> None:
    split_file = tmp_path / "test.jsonl"
    split_file.write_text("", encoding="utf-8")
    report = {
        "splits": {
            "test_unseen": {
                "file": str(split_file),
                "expected_episodes": 2,
            }
        }
    }
    config = {"episodes_file": "test.jsonl"}
    episodes = [{"episode_id": "a"}, {"episode_id": "b"}]
    require_full_paper_split_selection(config, report, episodes)

    with pytest.raises(ValueError, match="selected 1 of 2"):
        require_full_paper_split_selection(config, report, episodes[:1])
