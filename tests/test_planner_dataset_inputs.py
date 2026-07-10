from pathlib import Path

from PIL import Image

from voln_uav.common.io import write_jsonl
from voln_uav.data.episode_dataset import PlannerDataset


def _write_frame(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return str(path)


def test_planner_dataset_does_not_expose_subgoal_images(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    records_root = benchmark_root / "records"
    frames_root = tmp_path / "frames"
    records_root.mkdir(parents=True)

    image_paths = [_write_frame(frames_root / f"{i:04d}.png", (20 * i, 40, 120)) for i in range(6)]
    episode = {
        "episode_id": "episode_001",
        "states": [
            {
                "image": image_paths[i],
                "position": [float(i), 0.0, 0.0],
                "imu": [0.0] * 6,
                "odometry": [float(i), 0.0, 0.0],
            }
            for i in range(6)
        ],
    }
    record = {
        "record_id": "episode_001_0002",
        "episode_id": "episode_001",
        "step": 2,
        "image": image_paths[2],
        "proprio": [0.0] * 9,
        "visual_goal": {
            "V_goal": [image_paths[4], image_paths[5]],
            "V_sub": [image_paths[1], image_paths[3]],
        },
        "future_waypoints": [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        "anchor_waypoint": [4.0, 0.0, 0.0],
        "stop": False,
        "path_length": 5.0,
    }

    write_jsonl([episode], benchmark_root / "episodes.jsonl")
    write_jsonl([record], records_root / "train.jsonl")

    item = PlannerDataset(benchmark_root, "records/train.jsonl", image_size=16, memory_len=3)[0]

    assert "subgoal_images" not in item
    assert "subgoal_image_paths" not in item
    assert "subgoal_image_embeddings" not in item
    assert item["image"].shape == (3, 16, 16)
    assert item["history_images"].shape == (3, 3, 16, 16)
    assert item["goal_images"].shape == (2, 3, 16, 16)
    assert item["future_waypoints"].shape == (2, 3)