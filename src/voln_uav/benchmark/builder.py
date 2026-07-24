from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from voln_uav.benchmark.beacon_augmentation import (
    background_visibility_for_route,
    generate_beacons,
    generate_scene_background_beacons,
    visible_beacon_labels,
)
from voln_uav.benchmark.splitter import (
    assign_episode_splits_from_manifest,
    deduplicate_episodes,
    normalize_paper_split,
    trajectory_fingerprint,
    validate_paper_split_episodes,
)
from voln_uav.benchmark.trajectory import (
    compute_path_length,
    compute_start_goal_distance,
    find_decision_points,
    load_route_files,
)
from voln_uav.benchmark.visual_goal import build_visual_goal_interface
from voln_uav.common.geometry import l2
from voln_uav.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from voln_uav.common.navigation_frames import encode_proprioception, world_point_to_body


class BenchmarkBuilder:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.seed = int(config["seed"])
        self.rng = random.Random(self.seed)
        self.source_root = Path(config["source_root"])
        self.output_root = ensure_dir(config["output_root"])
        self.horizon = int(config["horizon"])
        self.semantic_bank = list(config.get("semantic_bank", {}).get("categories", []))

    def load_scene_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self.source_root / self.cfg.get("scene_manifest", "scenes.jsonl")
        return read_jsonl(manifest_path)

    def difficulty_label(self, route_length_m: float) -> str:
        easy_lt = float(self.cfg["difficulty"]["easy_lt"])
        normal_lt = float(self.cfg["difficulty"]["normal_lt"])
        if route_length_m < easy_lt:
            return "Easy"
        if route_length_m < normal_lt:
            return "Normal"
        return "Hard"

    def _relative(self, path_like: str | Path) -> str:
        path = Path(path_like)
        try:
            return str(path.relative_to(self.output_root.parent))
        except ValueError:
            return str(path)

    def build(self) -> dict[str, Any]:
        scenes = self.load_scene_manifest()
        scene_map = {s["scene_id"]: s for s in scenes}

        routes = load_route_files(self.source_root, self.cfg["preset_routes_dir"], self.cfg["custom_routes_dir"])
        routes_by_scene: dict[str, list[dict[str, Any]]] = {}
        for route in routes:
            routes_by_scene.setdefault(str(route["scene_id"]), []).append(route)
        scene_backgrounds = {
            scene_id: generate_scene_background_beacons(
                scene_id=scene_id,
                scene_type=str(scene_map[scene_id].get("scene_type", "unknown")),
                scene_states=[
                    state
                    for route in scene_routes
                    for state in route["states"]
                ],
                output_root=self.output_root,
                count=int(self.cfg["beacons"]["background_per_scene"]),
                semantic_bank=self.semantic_bank,
                seed=self.seed,
            )
            for scene_id, scene_routes in routes_by_scene.items()
        }
        episodes: list[dict[str, Any]] = []

        for route in tqdm(routes, desc="building-episodes"):
            scene_id = route["scene_id"]
            scene_meta = scene_map[scene_id]
            route_len = compute_path_length(route)
            decision_points = find_decision_points(
                route,
                min_separation_steps=int(self.cfg["beacons"]["min_separation_steps"]),
            )
            default_task_beacons = int(
                self.cfg["beacons"].get("task_beacons_per_route", 4)
            )
            min_task_beacons = int(
                self.cfg["beacons"].get(
                    "task_beacons_min_per_route",
                    default_task_beacons,
                )
            )
            max_task_beacons = int(
                self.cfg["beacons"].get(
                    "task_beacons_max_per_route",
                    default_task_beacons,
                )
            )
            if min_task_beacons < 0 or max_task_beacons < min_task_beacons:
                raise ValueError("Invalid active-beacon count range")
            task_beacons, background_beacons = generate_beacons(
                scene_id=scene_id,
                scene_type=str(scene_meta.get("scene_type", "unknown")),
                decision_points=decision_points,
                route_length=len(route["states"]),
                output_root=self.output_root,
                task_beacons_per_route=self.rng.randint(
                    min_task_beacons,
                    max_task_beacons,
                ),
                background_per_scene=0,
                semantic_bank=self.semantic_bank,
                rng=self.rng,
                task_category_allowlist=self.cfg["beacons"].get("task_category_allowlist"),
            )
            background_beacons = background_visibility_for_route(
                scene_backgrounds[scene_id],
                route["states"],
            )
            visual_goal = build_visual_goal_interface(
                route,
                task_beacons,
                num_terminal_views=int(self.cfg["goal_interface"]["num_terminal_views"]),
                num_subgoals=int(self.cfg["goal_interface"]["num_subgoals"]),
                num_beacons=int(self.cfg["goal_interface"]["num_beacons"]),
            )
            episode = {
                "episode_id": f"{scene_id}_{route['trajectory_id']}",
                "scene_id": scene_id,
                "scene_type": scene_meta.get("scene_type", "unknown"),
                "scene_source": scene_meta.get("scene_source"),
                "route_source": route.get("source", "preset"),
                "split": route.get("split"),
                "goal_category": route.get("goal_category", "goal"),
                "path_length": route_len,
                "start_to_goal_distance": compute_start_goal_distance(route),
                "shortest_path_length": route.get("shortest_path_length"),
                "shortest_path_provenance": route.get("shortest_path_provenance"),
                "difficulty": self.difficulty_label(route_len),
                "visual_goal": visual_goal,
                "task_beacons": task_beacons,
                "background_beacons": background_beacons,
                "states": route["states"],
            }
            episodes.append(episode)

        episodes = deduplicate_episodes(
            episodes,
            start_threshold=float(self.cfg["dedup"]["start_threshold"]),
            goal_threshold=float(self.cfg["dedup"]["goal_threshold"]),
        )
        split_manifest = self.cfg.get("split_manifest")
        if split_manifest:
            manifest_path = Path(split_manifest)
            if not manifest_path.is_absolute():
                manifest_path = self.source_root / manifest_path
            episodes = assign_episode_splits_from_manifest(
                episodes,
                read_jsonl(manifest_path),
            )
        else:
            if bool(self.cfg.get("strict_paper_protocol", False)):
                raise ValueError(
                    "strict_paper_protocol requires an explicit episode-level split_manifest; "
                    "scene-level random ratios cannot represent Validation-Seen"
                )
            for episode in episodes:
                if episode.get("split") is None:
                    raise ValueError(
                        f"Episode {episode['episode_id']} has no split. Supply split_manifest."
                    )
                episode["split"] = normalize_paper_split(str(episode["split"]))
                episode["trajectory_hash"] = trajectory_fingerprint(episode)

        split_protocol = self.cfg.get("paper_split_protocol")
        if bool(self.cfg.get("strict_paper_protocol", False)):
            if not isinstance(split_protocol, dict):
                raise ValueError("strict_paper_protocol requires paper_split_protocol")
            validate_paper_split_episodes(episodes, split_protocol)

        require_shortest_path = bool(self.cfg.get("require_shortest_path", False))
        if require_shortest_path:
            missing_shortest = [
                episode["episode_id"]
                for episode in episodes
                if episode.get("shortest_path_length") is None
                or not isinstance(episode.get("shortest_path_provenance"), dict)
            ]
            if missing_shortest:
                raise ValueError(
                    "Paper SPL requires independently computed shortest paths; "
                    f"{len(missing_shortest)} episodes are missing length/provenance"
                )

        write_jsonl(episodes, self.output_root / "episodes.jsonl")

        split_episodes: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for episode in episodes:
            split_episodes[episode["split"]].append(episode)
        for split_name, items in split_episodes.items():
            write_jsonl(items, self.output_root / f"{split_name}.jsonl")

        records_root = ensure_dir(self.output_root / "records")
        summary = self._build_records(episodes, records_root)

        semantic_root = ensure_dir(self.output_root / "semantic_bank")
        with (semantic_root / "categories.txt").open("w", encoding="utf-8") as f:
            for cat in self.semantic_bank:
                f.write(cat + "\n")

        write_json(summary, self.output_root / "summary.json")
        return summary

    def _build_records(self, episodes: list[dict[str, Any]], records_root: Path) -> dict[str, Any]:
        per_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        difficulty_hist: dict[str, int] = {"Easy": 0, "Normal": 0, "Hard": 0}
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for episode in tqdm(episodes, desc="building-records"):
            split_counts[episode["split"]] += 1
            difficulty_hist[episode["difficulty"]] = difficulty_hist.get(episode["difficulty"], 0) + 1
            states = episode["states"]
            for step_idx, state in enumerate(states):
                future = []
                for off in range(1, self.horizon + 1):
                    future_idx = min(step_idx + off, len(states) - 1)
                    future.append(
                        world_point_to_body(
                            states[future_idx]["position"],
                            state["position"],
                            float(state.get("yaw", 0.0)),
                            state.get("orientation"),
                        )
                    )
                anchor = future[-1]
                previous_state = states[step_idx - 1] if step_idx > 0 else None
                record = {
                    "record_id": f"{episode['episode_id']}_{step_idx:04d}",
                    "episode_id": episode["episode_id"],
                    "scene_id": episode["scene_id"],
                    "scene_type": episode["scene_type"],
                    "split": episode["split"],
                    "difficulty": episode["difficulty"],
                    "step": step_idx,
                    "image": state["image"],
                    "proprio": encode_proprioception(state, previous_state),
                    "proprio_schema": "body_linear_angular_relative_v1",
                    "waypoint_frame": "body_relative",
                    "future_waypoints": future,
                    "anchor_waypoint": anchor,
                    "stop": l2(state["position"], states[-1]["position"])
                    <= float(self.cfg.get("success_radius_m", 4.0)),
                    "visual_goal": episode["visual_goal"],
                    "beacon_label": visible_beacon_labels(step_idx, episode["task_beacons"], episode["background_beacons"]),
                    "path_length": episode["path_length"],
                    "start_to_goal_distance": episode["start_to_goal_distance"],
                    "shortest_path_length": episode["shortest_path_length"],
                }
                per_split[episode["split"]].append(record)

        for split_name, items in per_split.items():
            write_jsonl(items, records_root / f"{split_name}.jsonl")

        return {
            "num_episodes": len(episodes),
            "episodes_by_split": split_counts,
            "difficulty_hist": difficulty_hist,
            "records_by_split": {k: len(v) for k, v in per_split.items()},
        }
