from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from voln_uav.common.image import load_image_tensor, stack_images
from voln_uav.common.io import read_jsonl


class _BenchmarkBase(Dataset):
    def __init__(self, benchmark_root: str | Path, records_file: str | Path, image_size: int = 64) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.repo_root = self.benchmark_root.parent
        self.records = read_jsonl(self.benchmark_root / records_file)
        self.episodes = {ep["episode_id"]: ep for ep in read_jsonl(self.benchmark_root / "episodes.jsonl")}
        self.image_size = image_size

    def _resolve_path(self, path_like: str) -> Path:
        path = Path(path_like)
        if path.exists():
            return path
        path2 = self.repo_root / path_like
        if path2.exists():
            return path2
        path3 = self.benchmark_root / path_like
        if path3.exists():
            return path3
        raise FileNotFoundError(f"Could not resolve image path: {path_like}")

    def __len__(self) -> int:
        return len(self.records)


class AdapterDistillDataset(_BenchmarkBase):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image = load_image_tensor(self._resolve_path(record["image"]), image_size=self.image_size)
        return {"image": image, "record_id": record["record_id"]}


class PlannerDataset(_BenchmarkBase):
    def __init__(self, benchmark_root: str | Path, records_file: str | Path, image_size: int = 64, memory_len: int = 4) -> None:
        super().__init__(benchmark_root, records_file, image_size=image_size)
        self.memory_len = memory_len

    def _history_paths(self, episode: dict[str, Any], step_idx: int) -> list[Path]:
        states = episode["states"]
        history_indices = list(range(max(0, step_idx - self.memory_len + 1), step_idx + 1))
        while len(history_indices) < self.memory_len:
            history_indices.insert(0, history_indices[0])
        return [self._resolve_path(states[i]["image"]) for i in history_indices]

    def _history_proprio(self, episode: dict[str, Any], step_idx: int) -> torch.Tensor:
        states = episode["states"]
        history_indices = list(range(max(0, step_idx - self.memory_len + 1), step_idx + 1))
        while len(history_indices) < self.memory_len:
            history_indices.insert(0, history_indices[0])
        hist = [list(states[i].get("imu", [])) + list(states[i].get("odometry", [])) for i in history_indices]
        return torch.tensor(hist, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        episode = self.episodes[record["episode_id"]]
        step_idx = int(record["step"])
        cur_image = load_image_tensor(self._resolve_path(record["image"]), image_size=self.image_size)
        history_images = stack_images(self._history_paths(episode, step_idx), image_size=self.image_size)
        history_proprio = self._history_proprio(episode, step_idx)
        goal_images = stack_images([self._resolve_path(p) for p in record["visual_goal"]["V_goal"]], image_size=self.image_size)
        subgoal_images = stack_images([self._resolve_path(p) for p in record["visual_goal"]["V_sub"]], image_size=self.image_size)
        beacon_images = stack_images([self._resolve_path(p) for p in record["visual_goal"]["V_beacon"]], image_size=self.image_size)
        item = {
            "record_id": record["record_id"],
            "episode_id": record["episode_id"],
            "step": step_idx,
            "image": cur_image,
            "history_images": history_images,
            "history_proprio": history_proprio,
            "proprio": torch.tensor(record["proprio"], dtype=torch.float32),
            "goal_images": goal_images,
            "subgoal_images": subgoal_images,
            "beacon_images": beacon_images,
            "future_waypoints": torch.tensor(record["future_waypoints"], dtype=torch.float32),
            "anchor_waypoint": torch.tensor(record["anchor_waypoint"], dtype=torch.float32),
            "stop": torch.tensor(float(record["stop"]), dtype=torch.float32),
            "shortest_path_length": torch.tensor(float(record.get("shortest_path_length", record["path_length"])), dtype=torch.float32),
        }
        return item
