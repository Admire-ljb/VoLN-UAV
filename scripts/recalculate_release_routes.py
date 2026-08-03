from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.repair_route_discontinuities import (
        _backup_file,
        _path_length,
        _read_json,
        _refresh_checksums,
        _write_json,
    )
except ModuleNotFoundError:
    from repair_route_discontinuities import (  # type: ignore[no-redef]
        _backup_file,
        _path_length,
        _read_json,
        _refresh_checksums,
        _write_json,
    )


def _threshold_label(length_m: float, easy_lt: float, normal_lt: float) -> str:
    if length_m < easy_lt:
        return "Easy"
    if length_m < normal_lt:
        return "Normal"
    return "Hard"


def recalculate_release_routes(
    release_root: Path,
    *,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    manifest_path = release_root / "manifest.json"
    manifest = _read_json(manifest_path)
    thresholds = dict(manifest.get("difficulty_thresholds_m", {}))
    easy_lt = float(thresholds.get("easy_lt", 300.0))
    normal_lt = float(thresholds.get("normal_lt", 450.0))
    route_paths = sorted((release_root / "source" / "preset_routes").glob("*.json"))
    route_paths += sorted((release_root / "source" / "custom_routes").glob("*.json"))
    routes = [(path, _read_json(path)) for path in route_paths]

    lengths = {
        str(route["trajectory_id"]): round(_path_length(list(route.get("states", []))), 4)
        for _, route in routes
    }
    special_hard: set[str] = set()
    scene_rules = dict(manifest.get("scene_difficulty_rules", {}))
    refreshed_rules: dict[str, dict[str, Any]] = {}
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, route in routes:
        by_scene[str(route["scene_id"])].append(route)
    for scene_id, rule_value in scene_rules.items():
        rule = dict(rule_value or {})
        if rule.get("hard_assignment") != "top_k_by_reference_path_length":
            refreshed_rules[scene_id] = rule
            continue
        hard_count = int(rule["hard_count"])
        ranked = sorted(
            by_scene.get(scene_id, []),
            key=lambda route: (
                -lengths[str(route["trajectory_id"])],
                str(route["trajectory_id"]),
            ),
        )
        if not 0 < hard_count < len(ranked):
            raise ValueError(
                f"Invalid hard_count={hard_count} for scene {scene_id!r} "
                f"with {len(ranked)} routes"
            )
        special_hard.update(str(route["trajectory_id"]) for route in ranked[:hard_count])
        refreshed_rules[scene_id] = {
            "hard_assignment": "top_k_by_reference_path_length",
            "hard_count": hard_count,
            "physical_hard_min_path_length_m": lengths[str(ranked[hard_count - 1]["trajectory_id"])],
            "next_path_length_m": lengths[str(ranked[hard_count]["trajectory_id"])],
            "tie_breaker": "trajectory_id_ascending",
        }

    changes: list[dict[str, Any]] = []
    assignments: Counter[str] = Counter()
    modified_paths: set[Path] = set()
    for route_path, route in routes:
        trajectory_id = str(route["trajectory_id"])
        length_m = lengths[trajectory_id]
        difficulty = (
            "Hard"
            if trajectory_id in special_hard
            else _threshold_label(length_m, easy_lt, normal_lt)
        )
        difficulty_source = (
            "scene_path_length_rank"
            if trajectory_id in special_hard
            else "reference_path_length"
        )
        assignments[difficulty] += 1
        changed = (
            abs(float(route.get("path_length", -1.0)) - length_m) > 1e-4
            or str(route.get("difficulty")) != difficulty
            or str(route.get("difficulty_source")) != difficulty_source
            or "difficulty_length_m" in route
        )
        if not changed:
            continue
        changes.append(
            {
                "trajectory_id": trajectory_id,
                "old_path_length": route.get("path_length"),
                "new_path_length": length_m,
                "old_difficulty": route.get("difficulty"),
                "new_difficulty": difficulty,
            }
        )
        if not apply:
            continue
        _backup_file(route_path, release_root, backup_root)
        route["path_length"] = length_m
        route["difficulty"] = difficulty
        route["difficulty_source"] = difficulty_source
        route.pop("difficulty_length_m", None)
        _write_json(route_path, route)
        modified_paths.add(route_path)

    if apply:
        _backup_file(manifest_path, release_root, backup_root)
        manifest["episodes_by_difficulty"] = dict(sorted(assignments.items()))
        manifest["scene_difficulty_rules"] = refreshed_rules
        manifest.pop("zip_path", None)
        _write_json(manifest_path, manifest)
        modified_paths.add(manifest_path)
        _refresh_checksums(release_root, modified_paths, set())

    return {
        "release_root": str(release_root),
        "applied": apply,
        "route_count": len(routes),
        "changed_routes": len(changes),
        "difficulty_counts": dict(sorted(assignments.items())),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute route length and release difficulty from canonical states."
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = recalculate_release_routes(
        args.release_root,
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
