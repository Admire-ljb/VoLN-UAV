from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def _path_length(states: list[dict[str, Any]]) -> float:
    return sum(
        _distance(states[index - 1]["position"], states[index]["position"])
        for index in range(1, len(states))
    )


def _difficulty(path_length_m: float, easy_lt: float, normal_lt: float) -> str:
    if path_length_m < easy_lt:
        return "Easy"
    if path_length_m < normal_lt:
        return "Normal"
    return "Hard"


def _timestamp(state: dict[str, Any]) -> float | None:
    raw = state.get("raw")
    if not isinstance(raw, dict) or raw.get("timestamp") is None:
        return None
    return float(raw["timestamp"])


def _discontinuities(
    states: list[dict[str, Any]],
    max_step_m: float,
) -> list[dict[str, Any]]:
    """Return boundaries that cannot belong to one continuous flight."""
    boundaries: list[dict[str, Any]] = []
    for index in range(1, len(states)):
        step_m = _distance(states[index - 1]["position"], states[index]["position"])
        previous_timestamp = _timestamp(states[index - 1])
        current_timestamp = _timestamp(states[index])
        timestamp_reversal = (
            previous_timestamp is not None
            and current_timestamp is not None
            and current_timestamp <= previous_timestamp
        )
        if step_m <= max_step_m and not timestamp_reversal:
            continue
        boundaries.append(
            {
                "index": index,
                "step_m": step_m,
                "timestamp_reversal": timestamp_reversal,
                "previous_timestamp": previous_timestamp,
                "current_timestamp": current_timestamp,
            }
        )
    return boundaries


def _select_continuous_segment(
    states: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> tuple[int, int]:
    """Choose the longest sampled flight segment, then the longest path."""
    cuts = [0, *(int(item["index"]) for item in boundaries), len(states)]
    segments = [(start, end) for start, end in zip(cuts, cuts[1:]) if end > start]
    if not segments:
        return 0, 0
    return max(
        segments,
        key=lambda item: (
            item[1] - item[0],
            _path_length(states[item[0] : item[1]]),
            -item[0],
        ),
    )


def _safe_release_path(release_root: Path, relative_path: str) -> Path:
    candidate = (release_root / Path(relative_path)).resolve()
    candidate.relative_to(release_root.resolve())
    return candidate


def _backup_file(path: Path, release_root: Path, backup_root: Path | None) -> None:
    if backup_root is None or not path.exists():
        return
    relative = path.resolve().relative_to(release_root.resolve())
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def _move_removed_asset(path: Path, release_root: Path, backup_root: Path | None) -> None:
    if not path.exists():
        return
    if backup_root is None:
        path.unlink()
        return
    relative = path.resolve().relative_to(release_root.resolve())
    destination = backup_root / "removed_assets" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        path.unlink()
    else:
        shutil.move(str(path), str(destination))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_checksums(
    release_root: Path,
    modified_paths: set[Path],
    removed_paths: set[Path],
) -> None:
    checksum_path = release_root / "checksums.sha256"
    if not checksum_path.exists():
        return

    entries: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        entries.append((digest, relative.strip()))

    modified = {
        path.resolve().relative_to(release_root.resolve()).as_posix()
        for path in modified_paths
        if path.exists()
    }
    removed = {
        path.resolve().relative_to(release_root.resolve()).as_posix()
        for path in removed_paths
    }
    seen: set[str] = set()
    updated: list[tuple[str, str]] = []
    for digest, relative in entries:
        if relative in removed or not (release_root / relative).exists():
            continue
        if relative in modified:
            digest = _sha256(release_root / relative)
        updated.append((digest, relative))
        seen.add(relative)
    for relative in sorted(modified - seen):
        updated.append((_sha256(release_root / relative), relative))
    checksum_path.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in updated),
        encoding="utf-8",
    )


def repair_release(
    release_root: Path,
    *,
    scene_id: str | None,
    difficulty: str | None,
    max_step_m: float,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    if backup_root is not None:
        backup_root = backup_root.resolve()
        if backup_root == release_root or release_root in backup_root.parents:
            raise ValueError("backup_root must be outside release_root")

    route_paths = sorted((release_root / "source" / "preset_routes").glob("*.json"))
    route_paths += sorted((release_root / "source" / "custom_routes").glob("*.json"))
    repairs: list[dict[str, Any]] = []
    repaired_records: dict[str, dict[str, Any]] = {}
    removed_assets: set[Path] = set()
    modified_paths: set[Path] = set()

    manifest_path = release_root / "manifest.json"
    manifest = _read_json(manifest_path)
    thresholds = manifest.get("difficulty_thresholds_m", {})
    easy_lt = float(thresholds.get("easy_lt", 300.0))
    normal_lt = float(thresholds.get("normal_lt", 450.0))

    for route_path in route_paths:
        route = _read_json(route_path)
        if scene_id is not None and route.get("scene_id") != scene_id:
            continue
        if difficulty is not None and route.get("difficulty") != difficulty:
            continue
        states = list(route.get("states", []))
        boundaries = _discontinuities(states, max_step_m)
        if not boundaries:
            continue
        start_index, end_index = _select_continuous_segment(states, boundaries)
        retained = states[start_index:end_index]
        dropped = states[:start_index] + states[end_index:]
        if len(retained) < 2:
            raise ValueError(f"Cannot retain a valid continuous route segment: {route_path}")
        new_length = round(_path_length(retained), 4)
        new_difficulty = _difficulty(new_length, easy_lt, normal_lt)
        trajectory_id = str(route["trajectory_id"])
        repair = {
            "trajectory_id": trajectory_id,
            "old_difficulty": route.get("difficulty"),
            "new_difficulty": new_difficulty,
            "old_path_length": route.get("path_length"),
            "new_path_length": new_length,
            "old_num_states": len(states),
            "new_num_states": len(retained),
            "dropped_num_states": len(dropped),
            "retained_start_index": start_index,
            "retained_end_index_exclusive": end_index,
            "discontinuities": [
                {
                    **item,
                    "step_m": round(float(item["step_m"]), 4),
                }
                for item in boundaries
            ],
            "max_jump_m": round(
                max(float(item["step_m"]) for item in boundaries),
                4,
            ),
            "new_goal_image": retained[-1].get("image"),
            "new_goal_position": retained[-1].get("position"),
        }
        repairs.append(repair)
        repaired_records[trajectory_id] = repair

        for state in dropped:
            image_ref = state.get("image")
            if not image_ref:
                continue
            asset_path = _safe_release_path(release_root, str(image_ref))
            if asset_path.exists():
                removed_assets.add(asset_path)

        if not apply:
            continue

        _backup_file(route_path, release_root, backup_root)
        route["states"] = retained
        route["path_length"] = new_length
        route["difficulty"] = new_difficulty
        _write_json(route_path, route)
        modified_paths.add(route_path)

    if apply and repairs:
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
                repair = repaired_records.get(str(row.get("trajectory_id")))
                if repair is None:
                    continue
                row["difficulty"] = repair["new_difficulty"]
                row["path_length"] = repair["new_path_length"]
                row["num_states"] = repair["new_num_states"]
                if "num_image_files" in row:
                    row["num_image_files"] = repair["new_num_states"]
                if "difficulty_source" in row:
                    row["difficulty_source"] = "reference_path_length"
            _write_jsonl(path, rows)
            modified_paths.add(path)
            if path.name == "episodes.jsonl":
                episode_rows = rows

        _backup_file(manifest_path, release_root, backup_root)
        manifest["episodes_by_difficulty"] = dict(
            sorted(Counter(str(row.get("difficulty", "unknown")) for row in episode_rows).items())
        )
        _write_json(manifest_path, manifest)
        modified_paths.add(manifest_path)

        for asset_path in sorted(removed_assets):
            _move_removed_asset(asset_path, release_root, backup_root)
        _refresh_checksums(release_root, modified_paths, removed_assets)

    return {
        "release_root": str(release_root),
        "scene_id_filter": scene_id,
        "difficulty_filter": difficulty,
        "max_step_m": max_step_m,
        "applied": apply,
        "repaired_routes": len(repairs),
        "removed_assets": len(removed_assets),
        "dropped_states": sum(item["dropped_num_states"] for item in repairs),
        "difficulty_changes": dict(
            sorted(Counter(f"{item['old_difficulty']} -> {item['new_difficulty']}" for item in repairs).items())
        ),
        "repairs": repairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split routes at implausible pose/timestamp discontinuities and retain "
            "the longest continuous flight segment."
        )
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--difficulty", choices=["Easy", "Normal", "Hard"])
    parser.add_argument("--max-step-m", type=float, default=50.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = repair_release(
        args.release_root,
        scene_id=args.scene_id,
        difficulty=args.difficulty,
        max_step_m=args.max_step_m,
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
