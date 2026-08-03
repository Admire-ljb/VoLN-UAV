from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.repair_route_discontinuities import (
        _backup_file,
        _read_json,
        _read_jsonl,
        _write_json,
        _write_jsonl,
    )
except ModuleNotFoundError:
    from repair_route_discontinuities import (  # type: ignore[no-redef]
        _backup_file,
        _read_json,
        _read_jsonl,
        _write_json,
        _write_jsonl,
    )


def _portable_source_hint(value: str) -> str:
    normalized = value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and not part.endswith(":")]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "unknown")


def refresh_release_metadata(
    release_root: Path,
    *,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    route_paths = sorted((release_root / "source" / "preset_routes").glob("*.json"))
    route_paths += sorted((release_root / "source" / "custom_routes").glob("*.json"))
    route_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for route_path in route_paths:
        route = _read_json(route_path)
        route_by_id[str(route["trajectory_id"])] = (route_path, route)

    episodes_path = release_root / "metadata" / "episodes.jsonl"
    source_index_path = release_root / "metadata" / "source_data_index.jsonl"
    episode_rows = _read_jsonl(episodes_path)
    source_rows = _read_jsonl(source_index_path)
    if {str(row.get("trajectory_id")) for row in episode_rows} != set(route_by_id):
        raise ValueError("metadata/episodes.jsonl does not cover the route set exactly")
    if {str(row.get("trajectory_id")) for row in source_rows} != set(route_by_id):
        raise ValueError("metadata/source_data_index.jsonl does not cover the route set exactly")

    for row in episode_rows:
        trajectory_id = str(row["trajectory_id"])
        route_path, route = route_by_id[trajectory_id]
        states = list(route.get("states", []))
        row.update(
            {
                "episode_id": trajectory_id,
                "scene_id": route["scene_id"],
                "trajectory_id": trajectory_id,
                "source": route.get("source", "preset"),
                "split": route["split"],
                "difficulty": route["difficulty"],
                "path_length": route["path_length"],
                "num_states": len(states),
                "camera": route.get("camera", row.get("camera", "FrontCamera")),
                "pose_source": route.get("pose_source", row.get("pose_source", "airsim_log")),
                "route_file": route_path.relative_to(release_root).as_posix(),
            }
        )

    for row in source_rows:
        trajectory_id = str(row["trajectory_id"])
        _, route = route_by_id[trajectory_id]
        states = list(route.get("states", []))
        row.update(
            {
                "trajectory_id": trajectory_id,
                "scene_id": route["scene_id"],
                "difficulty": route["difficulty"],
                "source_root": str(route["difficulty"]).lower(),
                "source_episode": route["source_episode"],
                "camera": route.get("camera", row.get("camera", "FrontCamera")),
                "num_states": len(states),
                "num_image_files": len(states),
                "path_length": route["path_length"],
                "pose_source": route.get("pose_source", row.get("pose_source", "airsim_log")),
                "difficulty_source": route.get("difficulty_source", "reference_path_length"),
            }
        )
        row.pop("difficulty_length_m", None)

    split_rows = {
        split: [row for row in episode_rows if row.get("split") == split]
        for split in ("train", "val", "test")
    }
    skipped_path = release_root / "metadata" / "skipped.jsonl"
    skipped_rows = _read_jsonl(skipped_path)
    for row in skipped_rows:
        if row.get("episode_dir"):
            row["episode_dir"] = _portable_source_hint(str(row["episode_dir"]))

    manifest_path = release_root / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "num_scenes": len(_read_jsonl(release_root / "source" / "scenes.jsonl")),
            "num_routes": len(route_by_id),
            "num_source_routes": len(route_by_id),
            "num_skipped": len(skipped_rows),
            "episodes_by_split": {key: len(value) for key, value in split_rows.items()},
            "episodes_by_difficulty": dict(
                sorted(Counter(str(row["difficulty"]) for row in episode_rows).items())
            ),
        }
    )
    benchmark_summary_path = release_root / "benchmark" / "summary.json"
    if benchmark_summary_path.is_file():
        benchmark_summary = _read_json(benchmark_summary_path)
        manifest["num_benchmark_episodes"] = int(benchmark_summary["num_episodes"])
        manifest["benchmark"] = {
            "num_episodes": int(benchmark_summary["num_episodes"]),
            "episodes_by_split": dict(benchmark_summary["episodes_by_split"]),
            "episodes_by_difficulty": dict(benchmark_summary["difficulty_hist"]),
            "records_by_split": dict(benchmark_summary["records_by_split"]),
        }
    manifest.pop("zip_path", None)

    changed_paths = [episodes_path, source_index_path, skipped_path, manifest_path]
    changed_paths += [release_root / "splits" / f"{split}.jsonl" for split in split_rows]
    if apply:
        for path in changed_paths:
            _backup_file(path, release_root, backup_root)
        _write_jsonl(episodes_path, episode_rows)
        _write_jsonl(source_index_path, source_rows)
        _write_jsonl(skipped_path, skipped_rows)
        for split, rows in split_rows.items():
            _write_jsonl(release_root / "splits" / f"{split}.jsonl", rows)
        _write_json(manifest_path, manifest)

    absolute_path_hits = 0
    drive_pattern = re.compile(r"[A-Za-z]:[\\/]")
    for row in skipped_rows:
        absolute_path_hits += int(bool(drive_pattern.search(json.dumps(row, ensure_ascii=False))))
    return {
        "release_root": str(release_root),
        "applied": apply,
        "route_count": len(route_by_id),
        "split_counts": {key: len(value) for key, value in split_rows.items()},
        "difficulty_counts": manifest["episodes_by_difficulty"],
        "source_count_mismatches_after": sum(
            int(row["num_states"] != row["num_image_files"]) for row in source_rows
        ),
        "absolute_skipped_path_hits_after": absolute_path_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh portable release metadata from routes.")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = refresh_release_metadata(
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
