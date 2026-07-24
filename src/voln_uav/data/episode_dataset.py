from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from voln_uav.common.image import load_image_tensor, stack_images
from voln_uav.common.io import read_jsonl
from voln_uav.common.navigation_frames import encode_proprioception, world_point_to_body


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
    def __init__(
        self,
        benchmark_root: str | Path,
        records_file: str | Path,
        image_size: int = 64,
        memory_len: int = 4,
        image_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(benchmark_root, records_file, image_size=image_size)
        self.memory_len = memory_len
        self.image_embeddings = image_embeddings

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
        hist = [
            encode_proprioception(states[index], states[index - 1] if index > 0 else None)
            for index in history_indices
        ]
        return torch.tensor(hist, dtype=torch.float32)

    @staticmethod
    def _relative_waypoints(
        episode: dict[str, Any],
        step_idx: int,
        horizon: int,
    ) -> list[list[float]]:
        states = episode["states"]
        current = states[step_idx]
        origin = current["position"]
        yaw = float(current.get("yaw", 0.0))
        return [
            world_point_to_body(
                states[min(step_idx + offset, len(states) - 1)]["position"],
                origin,
                yaw,
                current.get("orientation"),
            )
            for offset in range(1, horizon + 1)
        ]

    def _lookup_embedding(self, path: Path) -> torch.Tensor:
        if self.image_embeddings is None:
            raise RuntimeError("image_embeddings is not configured")
        key = str(path)
        try:
            return self.image_embeddings[key].float()
        except KeyError as exc:
            raise KeyError(f"Missing precomputed image embedding for {key}") from exc

    def image_paths_for_record(self, record: dict[str, Any]) -> list[Path]:
        episode = self.episodes[record["episode_id"]]
        step_idx = int(record["step"])
        return [
            self._resolve_path(record["image"]),
            *self._history_paths(episode, step_idx),
            *[self._resolve_path(p) for p in record["visual_goal"]["V_goal"]],
        ]

    def iter_unique_image_paths(self) -> list[Path]:
        seen: set[str] = set()
        paths: list[Path] = []
        for record in self.records:
            for path in self.image_paths_for_record(record):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
        return paths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        episode = self.episodes[record["episode_id"]]
        step_idx = int(record["step"])
        cur_path = self._resolve_path(record["image"])
        history_paths = self._history_paths(episode, step_idx)
        goal_paths = [self._resolve_path(p) for p in record["visual_goal"]["V_goal"]]
        if len(goal_paths) != 3:
            raise ValueError(
                f"VoLN paper protocol requires exactly three terminal goal views; "
                f"record {record['record_id']} has {len(goal_paths)}"
            )
        history_proprio = self._history_proprio(episode, step_idx)
        horizon = len(record["future_waypoints"])
        future_waypoints = self._relative_waypoints(episode, step_idx, horizon)
        item = {
            "record_id": record["record_id"],
            "episode_id": record["episode_id"],
            "step": step_idx,
            "image_path": str(cur_path),
            "history_image_paths": [str(p) for p in history_paths],
            "history_proprio": history_proprio,
            "proprio": torch.tensor(
                encode_proprioception(
                    episode["states"][step_idx],
                    episode["states"][step_idx - 1] if step_idx > 0 else None,
                ),
                dtype=torch.float32,
            ),
            "goal_image_paths": [str(p) for p in goal_paths],
            "future_waypoints": torch.tensor(future_waypoints, dtype=torch.float32),
            "anchor_waypoint": torch.tensor(future_waypoints[-1], dtype=torch.float32),
            "stop": torch.tensor(float(record["stop"]), dtype=torch.float32),
            "shortest_path_length": torch.tensor(
                float(record["shortest_path_length"])
                if record.get("shortest_path_length") is not None
                else float("nan"),
                dtype=torch.float32,
            ),
        }
        if self.image_embeddings is None:
            item.update(
                {
                    "image": load_image_tensor(cur_path, image_size=self.image_size),
                    "history_images": stack_images(history_paths, image_size=self.image_size),
                    "goal_images": stack_images(goal_paths, image_size=self.image_size),
                }
            )
        else:
            item.update(
                {
                    "image_embedding": self._lookup_embedding(cur_path),
                    "history_image_embeddings": torch.stack([self._lookup_embedding(p) for p in history_paths], dim=0),
                    "goal_image_embeddings": torch.stack([self._lookup_embedding(p) for p in goal_paths], dim=0),
                }
            )
        return item
