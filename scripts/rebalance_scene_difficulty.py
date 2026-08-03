from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.repair_route_discontinuities import (
        _backup_file,
        _read_json,
        _read_jsonl,
        _refresh_checksums,
        _write_json,
        _write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from repair_route_discontinuities import (  # type: ignore[no-redef]
        _backup_file,
        _read_json,
        _read_jsonl,
        _refresh_checksums,
        _write_json,
        _write_jsonl,
    )


def _threshold_label(path_length_m: float, easy_lt: float, normal_lt: float) -> str:
    if path_length_m < easy_lt:
        return "Easy"
    if path_length_m < normal_lt:
        return "Normal"
    return "Hard"


def rebalance_release(
    release_root: Path,
    *,
    scene_id: str,
    hard_count: int,
    hard_min_difficulty_length_m: float | None,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    if hard_count <= 0:
        raise ValueError("hard_count must be positive")
    if backup_root is not None:
        backup_root = backup_root.resolve()
        if backup_root == release_root or release_root in backup_root.parents:
            raise ValueError("backup_root must be outside release_root")

    manifest_path = release_root / "manifest.json"
    manifest = _read_json(manifest_path)
    thresholds = manifest.get("difficulty_thresholds_m", {})
    easy_lt = float(thresholds.get("easy_lt", 300.0))
    normal_lt = float(thresholds.get("normal_lt", 450.0))

    route_paths = sorted((release_root / "source" / "preset_routes").glob("*.json"))
    route_paths += sorted((release_root / "source" / "custom_routes").glob("*.json"))
    scene_routes: list[tuple[Path, dict[str, Any]]] = []
    for route_path in route_paths:
        route = _read_json(route_path)
        if route.get("scene_id") == scene_id:
            scene_routes.append((route_path, route))
    if hard_count >= len(scene_routes):
        raise ValueError(
            f"hard_count={hard_count} must be smaller than the number of scene routes={len(scene_routes)}"
        )

    ranked = sorted(
        scene_routes,
        key=lambda item: (-float(item[1]["path_length"]), str(item[1]["trajectory_id"])),
    )
    hard_ids = {str(route["trajectory_id"]) for _, route in ranked[:hard_count]}
    hard_min_length = float(ranked[hard_count - 1][1]["path_length"])
    next_length = float(ranked[hard_count][1]["path_length"])
    if hard_min_difficulty_length_m is not None and hard_min_difficulty_length_m <= 0.0:
        raise ValueError("hard_min_difficulty_length_m must be positive")
    difficulty_length_scale = (
        float(hard_min_difficulty_length_m) / hard_min_length
        if hard_min_difficulty_length_m is not None
        else 1.0
    )
    assignments: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    modified_paths: set[Path] = set()

    for route_path, route in scene_routes:
        trajectory_id = str(route["trajectory_id"])
        length_m = float(route["path_length"])
        new_label = (
            "Hard"
            if trajectory_id in hard_ids
            else _threshold_label(length_m, easy_lt=easy_lt, normal_lt=normal_lt)
        )
        new_source = (
            "scene_path_length_rank"
            if new_label == "Hard"
            else "reference_path_length"
        )
        difficulty_length_m = (
            round(length_m * difficulty_length_scale, 4)
            if hard_min_difficulty_length_m is not None
            else None
        )
        assignments[trajectory_id] = new_label
        if (
            route.get("difficulty") == new_label
            and route.get("difficulty_source") == new_source
            and (
                difficulty_length_m is None
                or route.get("difficulty_length_m") == difficulty_length_m
            )
        ):
            continue
        changes.append(
            {
                "trajectory_id": trajectory_id,
                "path_length": length_m,
                "old_difficulty": route.get("difficulty"),
                "new_difficulty": new_label,
                "new_difficulty_source": new_source,
                "old_difficulty_length_m": route.get("difficulty_length_m"),
                "new_difficulty_length_m": difficulty_length_m,
            }
        )
        if not apply:
            continue
        _backup_file(route_path, release_root, backup_root)
        route["difficulty"] = new_label
        route["difficulty_source"] = new_source
        if difficulty_length_m is not None:
            route["difficulty_length_m"] = difficulty_length_m
        _write_json(route_path, route)
        modified_paths.add(route_path)

    if apply:
        metadata_paths = [
            release_root / "metadata" / "episodes.jsonl",
            release_root / "metadata" / "source_data_index.jsonl",
            release_root / "splits" / "train.jsonl",
            release_root / "splits" / "val.jsonl",
            release_root / "splits" / "test.jsonl",
        ]
        episode_rows: list[dict[str, Any]] = []
        for path in metadata_paths:
            if not path.exists():
                continue
            _backup_file(path, release_root, backup_root)
            rows = _read_jsonl(path)
            for row in rows:
                if row.get("scene_id") != scene_id:
                    continue
                trajectory_id = str(row.get("trajectory_id"))
                row["difficulty"] = assignments[trajectory_id]
                if hard_min_difficulty_length_m is not None:
                    row["difficulty_length_m"] = round(
                        float(row["path_length"]) * difficulty_length_scale,
                        4,
                    )
                if "difficulty_source" in row:
                    row["difficulty_source"] = (
                        "scene_path_length_rank"
                        if assignments[trajectory_id] == "Hard"
                        else "reference_path_length"
                    )
            _write_jsonl(path, rows)
            modified_paths.add(path)
            if path.name == "episodes.jsonl":
                episode_rows = rows

        _backup_file(manifest_path, release_root, backup_root)
        manifest["episodes_by_difficulty"] = dict(
            sorted(Counter(str(row.get("difficulty", "unknown")) for row in episode_rows).items())
        )
        scene_rules = dict(manifest.get("scene_difficulty_rules", {}))
        scene_rule = {
            "hard_assignment": "top_k_by_reference_path_length",
            "hard_count": hard_count,
            "physical_hard_min_path_length_m": hard_min_length,
            "next_path_length_m": next_length,
            "tie_breaker": "trajectory_id_ascending",
        }
        if hard_min_difficulty_length_m is not None:
            scene_rule.update(
                {
                    "difficulty_length_field": "difficulty_length_m",
                    "difficulty_length_scale": difficulty_length_scale,
                    "hard_min_difficulty_length_m": float(hard_min_difficulty_length_m),
                }
            )
        scene_rules[scene_id] = scene_rule
        manifest["scene_difficulty_rules"] = scene_rules
        _write_json(manifest_path, manifest)
        modified_paths.add(manifest_path)
        _refresh_checksums(release_root, modified_paths, set())

    scene_counts = Counter(assignments.values())
    return {
        "release_root": str(release_root),
        "scene_id": scene_id,
        "applied": apply,
        "scene_route_count": len(scene_routes),
        "target_hard_count": hard_count,
        "hard_min_path_length_m": hard_min_length,
        "next_path_length_m": next_length,
        "difficulty_length_scale": difficulty_length_scale,
        "hard_min_difficulty_length_m": (
            float(hard_min_difficulty_length_m)
            if hard_min_difficulty_length_m is not None
            else hard_min_length
        ),
        "scene_difficulty_counts": dict(sorted(scene_counts.items())),
        "changed_routes": len(changes),
        "change_counts": dict(
            sorted(Counter(f"{item['old_difficulty']} -> {item['new_difficulty']}" for item in changes).items())
        ),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign a fixed number of the longest routes in one scene to Hard."
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--hard-count", required=True, type=int)
    parser.add_argument("--hard-min-difficulty-length-m", type=float)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = rebalance_release(
        args.release_root,
        scene_id=args.scene_id,
        hard_count=args.hard_count,
        hard_min_difficulty_length_m=args.hard_min_difficulty_length_m,
        apply=args.apply,
        backup_root=args.backup_root,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
