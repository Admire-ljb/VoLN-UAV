from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scripts.repair_route_discontinuities import _read_json, _read_jsonl
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from repair_route_discontinuities import _read_json, _read_jsonl  # type: ignore[no-redef]


def audit_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    asset_mode = str(manifest.get("asset_mode"))
    errors: list[str] = []
    route_paths = sorted((root / "source" / "preset_routes").glob("*.json"))
    route_paths += sorted((root / "source" / "custom_routes").glob("*.json"))
    route_ids: set[str] = set()
    image_refs = 0
    raw_original_image_fields = 0
    max_step_m = 0.0

    for route_path in route_paths:
        route = _read_json(route_path)
        trajectory_id = str(route.get("trajectory_id"))
        scene_id = str(route.get("scene_id"))
        difficulty = str(route.get("difficulty")).lower()
        expected = re.compile(
            rf"^{re.escape(re.sub(r'[^A-Za-z0-9]+', '_', scene_id).strip('_'))}_{difficulty}_\d{{4}}$"
        )
        if trajectory_id in route_ids:
            errors.append(f"duplicate trajectory_id: {trajectory_id}")
        route_ids.add(trajectory_id)
        if route_path.stem != trajectory_id:
            errors.append(f"route filename mismatch: {route_path.name} != {trajectory_id}.json")
        if not expected.fullmatch(trajectory_id):
            errors.append(f"non-canonical trajectory_id: {trajectory_id}")
        if str(route.get("source_root")) != difficulty:
            errors.append(f"source_root mismatch: {trajectory_id}")
        source_leaf = Path(str(route.get("source_episode", "")).replace("\\", "/")).name
        if not source_leaf.startswith(f"{difficulty}_"):
            errors.append(f"source_episode mismatch: {trajectory_id}")
        image_prefix = f"source/frames/{scene_id}/{trajectory_id}/"
        states = list(route.get("states", []))
        calculated_length = 0.0
        previous_timestamp: float | None = None
        for index, state in enumerate(states):
            image_ref = str(state.get("image", ""))
            if not image_ref:
                continue
            image_refs += 1
            if not image_ref.startswith(image_prefix):
                errors.append(f"image reference mismatch: {trajectory_id}: {image_ref}")
                continue
            if asset_mode == "copy" and not (root / image_ref).is_file():
                errors.append(f"missing copied image: {image_ref}")
            raw = state.get("raw")
            if isinstance(raw, dict) and raw.get("original_image"):
                raw_original_image_fields += 1
            if index:
                step_m = math.dist(states[index - 1]["position"], state["position"])
                calculated_length += step_m
                max_step_m = max(max_step_m, step_m)
                if step_m > 20.0:
                    errors.append(f"route discontinuity {step_m:.3f}m: {trajectory_id}")
            timestamp = raw.get("timestamp") if isinstance(raw, dict) else None
            if timestamp is not None:
                current_timestamp = float(timestamp)
                if previous_timestamp is not None and current_timestamp <= previous_timestamp:
                    errors.append(f"non-increasing timestamp: {trajectory_id}")
                previous_timestamp = current_timestamp
        if abs(calculated_length - float(route.get("path_length", -1.0))) > 1e-3:
            errors.append(f"route path_length mismatch: {trajectory_id}")

    episodes = _read_jsonl(root / "metadata" / "episodes.jsonl")
    episode_ids: set[str] = set()
    for episode in episodes:
        trajectory_id = str(episode.get("trajectory_id"))
        episode_id = str(episode.get("episode_id"))
        difficulty = str(episode.get("difficulty")).lower()
        if episode_id != trajectory_id:
            errors.append(f"episode_id mismatch: {episode_id} != {trajectory_id}")
        if trajectory_id not in route_ids:
            errors.append(f"episode has no route: {trajectory_id}")
        if f"_{difficulty}_" not in trajectory_id:
            errors.append(f"episode difficulty mismatch: {trajectory_id}")
        route_file = root / str(episode.get("route_file", ""))
        if not route_file.is_file():
            errors.append(f"episode route_file missing: {trajectory_id}")
        episode_ids.add(episode_id)

    source_rows = _read_jsonl(root / "metadata" / "source_data_index.jsonl")
    for row in source_rows:
        trajectory_id = str(row.get("trajectory_id"))
        difficulty = str(row.get("difficulty")).lower()
        if trajectory_id not in route_ids:
            errors.append(f"source index has no route: {trajectory_id}")
        if str(row.get("source_root")) != difficulty:
            errors.append(f"source index difficulty mismatch: {trajectory_id}")
        if str(row.get("asset_mode")) != asset_mode:
            errors.append(f"source index asset_mode mismatch: {trajectory_id}")
        if int(row.get("num_states", -1)) != int(row.get("num_image_files", -2)):
            errors.append(f"source index image count mismatch: {trajectory_id}")

    split_ids: set[str] = set()
    for split_path in sorted((root / "splits").glob("*.jsonl")):
        split_ids.update(str(row.get("episode_id")) for row in _read_jsonl(split_path))
    if split_ids != episode_ids:
        errors.append(
            f"split membership mismatch: missing={len(episode_ids - split_ids)} extra={len(split_ids - episode_ids)}"
        )

    checksum_path = root / "checksums.sha256"
    checksum_files = {
        line.split(None, 1)[1].strip()
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if checksum_files != actual_files:
        errors.append(
            f"checksum path set mismatch: missing={len(actual_files - checksum_files)} stale={len(checksum_files - actual_files)}"
        )

    scenes = _read_jsonl(root / "source" / "scenes.jsonl")
    if int(manifest.get("num_scenes", -1)) != len(scenes):
        errors.append("manifest num_scenes mismatch")
    if int(manifest.get("num_routes", -1)) != len(route_paths):
        errors.append("manifest num_routes mismatch")

    portable_payloads = [json.dumps(manifest, ensure_ascii=False)]
    portable_payloads.extend(
        json.dumps(row, ensure_ascii=False)
        for row in _read_jsonl(root / "metadata" / "skipped.jsonl")
    )
    # Do not mistake the final letter of a URL scheme (``https:/``) for a
    # Windows drive.  A drive prefix must not be immediately preceded by a
    # scheme/identifier character.
    if any(
        re.search(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]", payload)
        for payload in portable_payloads
    ):
        errors.append("public metadata contains an absolute Windows path")

    benchmark_path = root / "benchmark" / "episodes.jsonl"
    benchmark_count = 0
    benchmark_image_refs = 0
    if benchmark_path.is_file():
        benchmark_rows = _read_jsonl(benchmark_path)
        benchmark_count = len(benchmark_rows)
        for episode in benchmark_rows:
            episode_id = str(episode.get("episode_id"))
            if episode_id not in route_ids:
                errors.append(f"benchmark episode has no canonical route: {episode_id}")
            states = list(episode.get("states", []))
            expected_beacons = 3 if float(episode["path_length"]) < 300.0 else (
                4 if float(episode["path_length"]) < 450.0 else 5
            )
            if len(episode.get("task_beacons", [])) != expected_beacons:
                errors.append(f"benchmark task beacon count mismatch: {episode_id}")
            referenced = [str(state.get("image", "")) for state in states]
            visual_goal = dict(episode.get("visual_goal", {}) or {})
            referenced += [str(value) for value in visual_goal.get("V_goal", [])]
            referenced += [str(value) for value in visual_goal.get("V_sub", [])]
            referenced += [str(value) for value in visual_goal.get("V_beacon", [])]
            for beacon in list(episode.get("task_beacons", [])) + list(
                episode.get("background_beacons", [])
            ):
                if beacon.get("template_image"):
                    referenced.append(str(beacon["template_image"]))
            for reference in referenced:
                if not reference:
                    continue
                benchmark_image_refs += 1
                if Path(reference).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", reference):
                    errors.append(f"absolute benchmark image path: {episode_id}")
                    continue
                if asset_mode == "copy" and not (root / reference).is_file():
                    errors.append(f"missing benchmark image: {reference}")
        if int(manifest.get("num_benchmark_episodes", -1)) != benchmark_count:
            errors.append("manifest num_benchmark_episodes mismatch")

    return {
        "root": str(root),
        "asset_mode": asset_mode,
        "scene_count": len(scenes),
        "route_count": len(route_paths),
        "episode_count": len(episodes),
        "source_index_count": len(source_rows),
        "image_reference_count": image_refs,
        "checksum_entry_count": len(checksum_files),
        "benchmark_episode_count": benchmark_count,
        "benchmark_image_reference_count": benchmark_image_refs,
        "raw_original_image_fields": raw_original_image_fields,
        "max_step_m": round(max_step_m, 4),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical release naming and references.")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_release(args.release_root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
