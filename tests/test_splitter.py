from voln_uav.benchmark.splitter import (
    assign_episode_splits_from_manifest,
    split_counts,
    validate_paper_split_episodes,
)


def test_split_counts_keeps_validation_when_possible():
    counts = split_counts(5, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert counts == {"train": 3, "val": 1, "test": 1}


def _episode(episode_id: str, scene_id: str, scene_source: str, offset: float) -> dict:
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "scene_source": scene_source,
        "states": [
            {"position": [offset, 0.0, 0.0], "yaw": 0.0},
            {"position": [offset + 1.0, 0.0, 0.0], "yaw": 0.0},
        ],
    }


def test_episode_manifest_supports_seen_validation_and_unseen_test():
    episodes = [
        _episode("train-a", "forest", "source-a", 0.0),
        _episode("val-a", "forest", "source-a", 10.0),
        _episode("test-a", "city", "source-b", 20.0),
    ]
    assigned = assign_episode_splits_from_manifest(
        episodes,
        [
            {"episode_id": "train-a", "split": "train", "scene_source": "source-a"},
            {"episode_id": "val-a", "split": "validation_seen", "scene_source": "source-a"},
            {"episode_id": "test-a", "split": "test_unseen", "scene_source": "source-b"},
        ],
    )
    validate_paper_split_episodes(
        assigned,
        {
            "train_episodes": 1,
            "validation_seen_episodes": 1,
            "test_unseen_episodes": 1,
            "train_environments": 1,
            "validation_seen_environments": 1,
            "test_unseen_environments": 1,
            "total_environments": 2,
            "require_held_out_scene_source": True,
        },
    )
