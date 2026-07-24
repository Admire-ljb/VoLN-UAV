from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from voln_uav.benchmark.splitter import (
    assign_episode_splits_from_manifest,
    assign_scene_splits,
    validate_paper_split_episodes,
)
from voln_uav.common.geometry import path_length
from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.common.navigation_frames import (
    DEFAULT_SAMPLE_INTERVAL_SEC,
    PROPRIO_SCHEMA,
    encode_proprioception,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CAMERA_PRIORITY = (
    "FrontCamera",
    "front_camera",
    "front",
    "RGB",
    "rgb",
    "RightCamera",
    "images",
)
FRAME_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")
DEFAULT_PAPER_SPLIT_PROTOCOL = {
    "train_episodes": 5047,
    "validation_seen_episodes": 1082,
    "test_unseen_episodes": 1081,
    "train_environments": 12,
    "validation_seen_environments": 5,
    "test_unseen_environments": 5,
    "total_environments": 17,
    "require_held_out_scene_source": True,
}


@dataclass(frozen=True)
class RawSourceRoot:
    key: str
    path: Path


@dataclass(frozen=True)
class RawEpisode:
    source: RawSourceRoot
    episode_dir: Path
    scene_id: str
    camera_dir: Path | None
    log_dir: Path | None


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
    return cleaned or "item"


def frame_number(path: Path) -> int | None:
    match = FRAME_NUMBER_RE.search(path.stem)
    return int(match.group(1)) if match else None


def quaternion_to_yaw(quat: list[float] | tuple[float, ...] | None) -> float:
    if not quat or len(quat) != 4:
        return 0.0
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def infer_scene_type(scene_id: str) -> str:
    name = scene_id.lower()
    if "desert" in name:
        return "desert"
    if "forest" in name:
        return "forest"
    if "urban" in name:
        return "urban"
    if "country" in name or "road" in name:
        return "rural"
    if "tunnel" in name or "corridor" in name:
        return "tunnel"
    return "unknown"


def difficulty_from_length(path_length_m: float, easy_lt: float = 300.0, normal_lt: float = 450.0) -> str:
    if path_length_m < easy_lt:
        return "Easy"
    if path_length_m < normal_lt:
        return "Normal"
    return "Hard"


def route_source_for_episode(ep: RawEpisode) -> str:
    text = str(ep.episode_dir).lower()
    if "custom" in text or "hard" in text:
        return "custom"
    return "preset"


def make_source_roots(source_roots: Iterable[str | Path]) -> list[RawSourceRoot]:
    return [RawSourceRoot(f"source_{idx:02d}", Path(path)) for idx, path in enumerate(source_roots, start=1)]


def _iter_files(path: Path, suffixes: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def _select_camera_dir(episode_dir: Path, requested_camera: str | None = None) -> Path | None:
    if requested_camera:
        direct = episode_dir / requested_camera
        if direct.exists() and _iter_files(direct, IMAGE_EXTENSIONS):
            return direct
    for name in CAMERA_PRIORITY:
        direct = episode_dir / name
        if direct.exists() and _iter_files(direct, IMAGE_EXTENSIONS):
            return direct
    for child in episode_dir.iterdir():
        if child.is_dir() and child.name.lower() in {c.lower() for c in CAMERA_PRIORITY}:
            if _iter_files(child, IMAGE_EXTENSIONS):
                return child
    return None


def _scene_id_from_episode(root: Path, episode_dir: Path) -> str:
    rel = episode_dir.relative_to(root)
    if not rel.parts:
        return sanitize_id(episode_dir.name)
    return sanitize_id(rel.parts[0])


def discover_raw_episodes(
    roots: Iterable[RawSourceRoot],
    camera: str | None = None,
    max_episodes_per_source: int | None = None,
) -> list[RawEpisode]:
    episodes: list[RawEpisode] = []
    seen_dirs: set[Path] = set()
    per_source: dict[str, int] = {}

    for root in roots:
        if not root.path.exists():
            continue

        for current_dir, dirnames, _ in os.walk(root.path):
            count = per_source.get(root.key, 0)
            if max_episodes_per_source is not None and count >= max_episodes_per_source:
                dirnames[:] = []
                break

            current = Path(current_dir)
            log_dir = current / "log"
            if log_dir.exists() and _iter_files(log_dir, {".json"}):
                ep_dir = current
                if ep_dir not in seen_dirs:
                    camera_dir = _select_camera_dir(ep_dir, camera)
                    episodes.append(
                        RawEpisode(
                            source=root,
                            episode_dir=ep_dir,
                            scene_id=_scene_id_from_episode(root.path, ep_dir),
                            camera_dir=camera_dir,
                            log_dir=log_dir,
                        )
                    )
                    seen_dirs.add(ep_dir)
                    per_source[root.key] = count + 1
                    if max_episodes_per_source is not None and per_source[root.key] >= max_episodes_per_source:
                        dirnames[:] = []
                        break
                continue

            camera_dir = _select_camera_dir(current, camera)
            if camera_dir is None:
                continue
            if current in seen_dirs or (current / "log").exists():
                continue
            episodes.append(
                RawEpisode(
                    source=root,
                    episode_dir=current,
                    scene_id=_scene_id_from_episode(root.path, current),
                    camera_dir=camera_dir,
                    log_dir=None,
                )
            )
            seen_dirs.add(current)
            per_source[root.key] = count + 1
            if max_episodes_per_source is not None and per_source[root.key] >= max_episodes_per_source:
                dirnames[:] = []
                break

    return episodes


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _position_from_log(log: dict[str, Any] | None) -> list[float] | None:
    if not log:
        return None
    candidates = [
        log.get("position"),
        log.get("state", {}).get("position") if isinstance(log.get("state"), dict) else None,
        log.get("sensors", {}).get("state", {}).get("position") if isinstance(log.get("sensors"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list) and len(candidate) >= 3:
            return [float(candidate[0]), float(candidate[1]), float(candidate[2])]
    return None


def _yaw_from_log(log: dict[str, Any] | None) -> float | None:
    if not log:
        return None
    if "yaw" in log:
        return float(log["yaw"])
    state = log.get("state") if isinstance(log.get("state"), dict) else {}
    sensors_state = log.get("sensors", {}).get("state", {}) if isinstance(log.get("sensors"), dict) else {}
    orientation = state.get("orientation") or sensors_state.get("orientation")
    if isinstance(orientation, list):
        return quaternion_to_yaw(orientation)
    return None


def _orientation_from_log(log: dict[str, Any] | None) -> list[float] | None:
    if not log:
        return None
    direct = log.get("orientation")
    state = log.get("state") if isinstance(log.get("state"), dict) else {}
    sensors_state = log.get("sensors", {}).get("state", {}) if isinstance(log.get("sensors"), dict) else {}
    orientation = direct or state.get("orientation") or sensors_state.get("orientation")
    if not isinstance(orientation, list) or len(orientation) != 4:
        return None
    return [float(value) for value in orientation]


def _timestamp_from_log(log: dict[str, Any] | None) -> float | None:
    if log and "timestamp" in log:
        timestamp = float(log["timestamp"])
        if abs(timestamp) > 1e12:
            timestamp /= 1e9
        return timestamp
    return None


def _collision_from_log(log: dict[str, Any] | None) -> bool | None:
    if not log:
        return None
    sensors_state = log.get("sensors", {}).get("state", {}) if isinstance(log.get("sensors"), dict) else {}
    collision = sensors_state.get("collision") if isinstance(sensors_state, dict) else None
    if isinstance(collision, dict) and "has_collided" in collision:
        return bool(collision["has_collided"])
    return None


def _nearest_log(frame_idx: int, log_numbers: list[int], log_map: dict[int, Path]) -> dict[str, Any] | None:
    if not log_numbers:
        return None
    if frame_idx in log_map:
        return _load_json(log_map[frame_idx])
    insert = bisect.bisect_left(log_numbers, frame_idx)
    candidates = []
    if insert > 0:
        candidates.append(log_numbers[insert - 1])
    if insert < len(log_numbers):
        candidates.append(log_numbers[insert])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - frame_idx))
    return _load_json(log_map[nearest])


def _copy_or_index_image(
    image_path: Path,
    out_root: Path,
    frames_root: Path,
    scene_id: str,
    trajectory_id: str,
    asset_mode: str,
    source_root: Path,
) -> str:
    if asset_mode == "copy":
        dst = frames_root / scene_id / trajectory_id / image_path.name
        ensure_dir(dst.parent)
        if not dst.exists() or dst.stat().st_size != image_path.stat().st_size:
            shutil.copy2(image_path, dst)
        return str(dst.relative_to(out_root)).replace("\\", "/")
    return str(image_path.relative_to(source_root.parent)).replace("\\", "/")


def _resample_states(
    states: list[dict[str, Any]],
    *,
    sample_interval_sec: float,
    timestamp_tolerance_sec: float,
    strict_timestamps: bool,
) -> list[dict[str, Any]]:
    timestamps = [
        state.get("raw", {}).get("timestamp")
        if isinstance(state.get("raw"), dict)
        else None
        for state in states
    ]
    if any(value is None for value in timestamps):
        if strict_timestamps:
            raise ValueError("Paper release requires a timestamp for every synchronized RGB/state sample")
        return states
    numeric = [float(value) for value in timestamps]
    if any(numeric[index] <= numeric[index - 1] for index in range(1, len(numeric))):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    interval = float(sample_interval_sec)
    tolerance = float(timestamp_tolerance_sec)
    if interval <= 0.0 or tolerance < 0.0:
        raise ValueError("Invalid trajectory sampling interval/tolerance")

    selected: list[dict[str, Any]] = []
    previous_index = -1
    target = numeric[0]
    while target <= numeric[-1] + tolerance:
        candidates = range(previous_index + 1, len(states))
        nearest = min(candidates, key=lambda index: abs(numeric[index] - target), default=None)
        if nearest is None:
            break
        error = abs(numeric[nearest] - target)
        if error > tolerance:
            if strict_timestamps:
                raise ValueError(
                    f"No synchronized sample within {tolerance:.3f}s of target timestamp {target:.6f}"
                )
            target += interval
            continue
        selected.append(states[nearest])
        previous_index = nearest
        target += interval
    return selected


def build_route_from_raw_episode(
    ep: RawEpisode,
    out_root: Path,
    frames_root: Path,
    asset_mode: str,
    easy_lt: float,
    normal_lt: float,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    timestamp_tolerance_sec: float = 0.25,
    strict_timestamps: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative_episode = ep.episode_dir.relative_to(ep.source.path)
    trajectory_id = sanitize_id(f"{ep.source.key}_{relative_episode}")
    image_files = _iter_files(ep.camera_dir, IMAGE_EXTENSIONS) if ep.camera_dir else []
    log_files = _iter_files(ep.log_dir, {".json"}) if ep.log_dir else []
    log_map = {num: path for path in log_files if (num := frame_number(path)) is not None}
    log_numbers = sorted(log_map)

    if not image_files and not log_numbers:
        return None, {
            "episode_dir": str(ep.episode_dir),
            "reason": "no images or logs",
        }

    if not image_files:
        return None, {
            "episode_dir": str(ep.episode_dir),
            "reason": "no egocentric RGB frames",
        }

    if not log_numbers:
        return None, {
            "episode_dir": str(ep.episode_dir),
            "reason": "incomplete trajectory metadata",
            "num_image_files": len(image_files),
        }

    states: list[dict[str, Any]] = []
    prev_yaw: float | None = None

    skipped_frames = 0
    for state_index, frame_path in enumerate(image_files):
        frame_idx = frame_number(frame_path) or state_index
        log = _nearest_log(frame_idx, log_numbers, log_map) if log_numbers else None
        position = _position_from_log(log)
        if position is None:
            skipped_frames += 1
            continue
        yaw = _yaw_from_log(log)
        if yaw is None:
            yaw = prev_yaw if prev_yaw is not None else 0.0
        orientation = _orientation_from_log(log)
        image_ref = ""
        if frame_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_ref = _copy_or_index_image(
                frame_path,
                out_root=out_root,
                frames_root=frames_root,
                scene_id=ep.scene_id,
                trajectory_id=trajectory_id,
                asset_mode=asset_mode,
                source_root=ep.source.path,
            )
        states.append(
            {
                "t": int(frame_idx),
                "position": [round(float(v), 6) for v in position],
                "yaw": round(float(yaw), 6),
                "orientation": (
                    [round(float(value), 8) for value in orientation]
                    if orientation is not None
                    else None
                ),
                "image": image_ref,
                "raw": {
                    "frame_index": int(frame_idx),
                    "timestamp": _timestamp_from_log(log),
                    "collision": _collision_from_log(log),
                    "pose_source": "airsim_log",
                },
            }
        )
        prev_yaw = yaw

    states = _resample_states(
        states,
        sample_interval_sec=sample_interval_sec,
        timestamp_tolerance_sec=timestamp_tolerance_sec,
        strict_timestamps=strict_timestamps,
    )
    for index, state in enumerate(states):
        proprio = encode_proprioception(
            state,
            states[index - 1] if index > 0 else None,
            default_interval_sec=sample_interval_sec,
        )
        state["imu"] = [round(value, 6) for value in proprio[:6]]
        state["odometry"] = [round(value, 6) for value in proprio[6:]]
        state["proprio_schema"] = PROPRIO_SCHEMA

    if len(states) < 2:
        return None, {
            "episode_dir": str(ep.episode_dir),
            "reason": "incomplete trajectory metadata",
            "num_image_files": len(image_files),
            "num_log_files": len(log_files),
            "skipped_frames": skipped_frames,
        }

    length_m = path_length([state["position"] for state in states])
    difficulty = difficulty_from_length(length_m, easy_lt=easy_lt, normal_lt=normal_lt)
    route = {
        "scene_id": ep.scene_id,
        "trajectory_id": trajectory_id,
        "source": route_source_for_episode(ep),
        "goal_category": "visual_goal",
        "split": None,
        "difficulty": difficulty,
        "path_length": round(length_m, 4),
        "source_root": ep.source.key,
        "source_episode": str(relative_episode).replace("\\", "/"),
        "camera": ep.camera_dir.name if ep.camera_dir else None,
        "pose_source": "airsim_log",
        "difficulty_source": "reference_path_length",
        "sample_interval_sec": float(sample_interval_sec),
        "proprio_schema": PROPRIO_SCHEMA,
        "states": states,
    }
    index = {
        "trajectory_id": trajectory_id,
        "scene_id": ep.scene_id,
        "difficulty": difficulty,
        "source_root": ep.source.key,
        "source_episode": route["source_episode"],
        "camera": route["camera"],
        "num_states": len(states),
        "num_log_files": len(log_files),
        "num_image_files": len(image_files),
        "path_length": route["path_length"],
        "pose_source": route["pose_source"],
        "difficulty_source": route["difficulty_source"],
        "skipped_frames": skipped_frames,
        "asset_mode": asset_mode,
    }
    return route, index


def _count_by(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def reset_release_tree(out_root: Path) -> None:
    generated_dirs = [
        out_root / "source" / "preset_routes",
        out_root / "source" / "custom_routes",
        out_root / "source" / "frames",
        out_root / "splits",
        out_root / "metadata",
    ]
    generated_files = [
        out_root / "source" / "scenes.jsonl",
        out_root / "README.md",
        out_root / "manifest.json",
        out_root / "checksums.sha256",
    ]
    for path in generated_dirs:
        if path.exists():
            shutil.rmtree(path)
    for path in generated_files:
        if path.exists():
            path.unlink()


def write_dataset_card(out_root: Path, summary: dict[str, Any], dataset_url: str, env_url: str) -> Path:
    card_path = out_root / "README.md"
    text = f"""---
license: mit
task_categories:
- robotics
- reinforcement-learning
language:
- en
pretty_name: VoLN-UAV Dataset
---

# VoLN-UAV Dataset

This Hugging Face dataset entry is intended for the navigation data used by VoLN-UAV. The environment assets are hosted separately so that users can fetch the simulator package and the trajectory data independently.

## Hugging Face Entries

- env: {env_url}
- dataset: {dataset_url}

## Data Organization

The release package provides the benchmark inputs required by VoLN-UAV:

- scene-level Train/Validation/Test split manifests;
- route JSON files with RGB frame references and pose-derived state fields;
- benchmark metadata for scenes, episodes, and checksums;
- optional copied RGB frames under `source/frames/` when the package is built in `copy` mode.

## Usage

1. Download the dataset package and the `env` package.
2. Unzip the dataset package.
3. Set `source_root` in the benchmark config to the unzipped `source/` directory.
4. Run `python -m voln_uav.cli.build_benchmark --config <config.yaml>`.

The generated `manifest.json` contains the release summary and Hugging Face resource links.

## Recommended Citation

Please cite the VoLN-UAV paper and this dataset repository once the manuscript metadata is finalized.
"""
    card_path.write_text(text, encoding="utf-8")
    return card_path


def write_checksums(root: Path) -> Path:
    checksum_path = root / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != checksum_path.name):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return checksum_path


def zip_release(root: Path, zip_path: Path) -> Path:
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.resolve() == zip_path.resolve():
                continue
            compress_type = zipfile.ZIP_STORED if path.suffix.lower() in IMAGE_EXTENSIONS else zipfile.ZIP_DEFLATED
            zf.write(path, arcname=Path(root.name) / path.relative_to(root), compress_type=compress_type)
    return zip_path


def prepare_dataset_release(
    easy_root: str | Path | None = None,
    normal_root: str | Path | None = None,
    hard_root: str | Path | None = None,
    out_root: str | Path | None = None,
    dataset_url: str = "",
    env_url: str = "",
    source_roots: Iterable[str | Path] | None = None,
    seed: int = 7,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    split_manifest: str | Path | None = None,
    strict_paper_protocol: bool = True,
    paper_split_protocol: dict[str, Any] | None = None,
    camera: str | None = None,
    asset_mode: str = "index",
    zip_path: str | Path | None = None,
    write_zip: bool = True,
    max_episodes_per_source: int | None = None,
    easy_lt: float = 300.0,
    normal_lt: float = 450.0,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    timestamp_tolerance_sec: float = 0.25,
    strict_timestamps: bool = True,
) -> dict[str, Any]:
    if out_root is None:
        raise ValueError("out_root is required")
    if asset_mode not in {"index", "copy"}:
        raise ValueError("asset_mode must be 'index' or 'copy'")
    ratio_values = (train_ratio, val_ratio, test_ratio)
    ratios: dict[str, float] | None = None
    if any(value is not None for value in ratio_values):
        if strict_paper_protocol:
            raise ValueError(
                "Scene-level train/val/test ratios cannot represent the VoLN paper split; "
                "provide an episode-level split_manifest"
            )
        if any(value is None for value in ratio_values):
            raise ValueError("train_ratio, val_ratio, and test_ratio must be provided together")
        ratios = {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        }
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError("train, val, and test ratios must sum to 1.0")

    split_assignments = (
        read_jsonl(split_manifest)
        if split_manifest is not None
        else None
    )
    out_root = ensure_dir(out_root)
    reset_release_tree(out_root)
    source_root = ensure_dir(out_root / "source")
    preset_root = ensure_dir(source_root / "preset_routes")
    custom_root = ensure_dir(source_root / "custom_routes")
    frames_root = ensure_dir(source_root / "frames")
    splits_root = ensure_dir(out_root / "splits")
    metadata_root = ensure_dir(out_root / "metadata")

    if source_roots is not None:
        raw_roots = make_source_roots(source_roots)
    else:
        if easy_root is None or normal_root is None or hard_root is None:
            raise ValueError("provide source_roots or all three legacy root arguments")
        raw_roots = make_source_roots([easy_root, normal_root, hard_root])
    raw_episodes = discover_raw_episodes(
        raw_roots,
        camera=camera,
        max_episodes_per_source=max_episodes_per_source,
    )

    routes: list[dict[str, Any]] = []
    source_index: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for ep in raw_episodes:
        route, index = build_route_from_raw_episode(
            ep,
            out_root=out_root,
            frames_root=frames_root,
            asset_mode=asset_mode,
            easy_lt=easy_lt,
            normal_lt=normal_lt,
            sample_interval_sec=sample_interval_sec,
            timestamp_tolerance_sec=timestamp_tolerance_sec,
            strict_timestamps=strict_timestamps,
        )
        if route is None:
            skipped.append(index)
            continue
        routes.append(route)
        source_index.append(index)

    for route in routes:
        route["episode_id"] = f"{route['scene_id']}_{route['trajectory_id']}"
        route["scene_source"] = route["source_root"]

    scene_ids = sorted({route["scene_id"] for route in routes})
    if split_assignments is not None:
        routes = assign_episode_splits_from_manifest(routes, split_assignments)
        if strict_paper_protocol:
            validate_paper_split_episodes(
                routes,
                paper_split_protocol or DEFAULT_PAPER_SPLIT_PROTOCOL,
            )
    else:
        if strict_paper_protocol:
            raise ValueError(
                "strict_paper_protocol requires an episode-level split_manifest"
            )
        if ratios is None:
            raise ValueError(
                "Diagnostic packaging without split_manifest requires explicit train/val/test ratios"
            )
        split_map = assign_scene_splits(scene_ids, ratios, seed)
        for route in routes:
            route["split"] = split_map[route["scene_id"]]

    scenes = []
    for scene_id in scene_ids:
        scene_routes = [route for route in routes if route["scene_id"] == scene_id]
        scene_splits = {str(route["split"]) for route in scene_routes}
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_type": infer_scene_type(scene_id),
                "scene_source": scene_routes[0]["scene_source"],
                "paper_pool": "test_unseen" if scene_splits == {"test"} else "train",
                "validation_seen": "val" in scene_splits,
            }
        )
    write_jsonl(scenes, source_root / "scenes.jsonl")

    episodes = []
    split_items: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for route in routes:
        route_path = (preset_root if route["source"] == "preset" else custom_root) / f"{route['trajectory_id']}.json"
        write_json(route, route_path)
        episode = {
            "episode_id": route["episode_id"],
            "scene_id": route["scene_id"],
            "trajectory_id": route["trajectory_id"],
            "source": route["source"],
            "split": route["split"],
            "difficulty": route["difficulty"],
            "path_length": route["path_length"],
            "num_states": len(route["states"]),
            "camera": route["camera"],
            "pose_source": route["pose_source"],
            "route_file": str(route_path.relative_to(out_root)).replace("\\", "/"),
        }
        episodes.append(episode)
        split_items[route["split"]].append(episode)

    write_jsonl(episodes, metadata_root / "episodes.jsonl")
    write_jsonl(source_index, metadata_root / "source_data_index.jsonl")
    write_jsonl(skipped, metadata_root / "skipped.jsonl")
    if split_assignments is not None:
        write_jsonl(
            split_assignments,
            metadata_root / "paper_episode_splits.jsonl",
        )
    for split_name, items in split_items.items():
        write_jsonl(items, splits_root / f"{split_name}.jsonl")

    summary = {
        "num_scenes": len(scene_ids),
        "num_routes": len(routes),
        "num_skipped": len(skipped),
        "asset_mode": asset_mode,
        "split_mode": "paper_manifest" if split_assignments is not None else "diagnostic_scene_ratios",
        "split_manifest_file": (
            "metadata/paper_episode_splits.jsonl"
            if split_assignments is not None
            else None
        ),
        "splits": ratios,
        "sample_interval_sec": float(sample_interval_sec),
        "proprio_schema": PROPRIO_SCHEMA,
        "episodes_by_split": _count_by(episodes, "split"),
        "episodes_by_difficulty": _count_by(episodes, "difficulty"),
        "episodes_by_pose_source": _count_by(episodes, "pose_source"),
        "difficulty_thresholds_m": {"easy_lt": easy_lt, "normal_lt": normal_lt},
        "hf": {"env": env_url, "dataset": dataset_url},
    }
    if write_zip:
        zip_target = Path(zip_path) if zip_path is not None else out_root.with_suffix(".zip")
        summary["zip_path"] = str(zip_target)
    write_json(summary, out_root / "manifest.json")
    write_dataset_card(out_root, summary, dataset_url=dataset_url, env_url=env_url)
    write_checksums(out_root)
    if write_zip:
        zip_release(out_root, zip_target)
    return summary
