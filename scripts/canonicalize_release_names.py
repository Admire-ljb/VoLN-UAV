from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.repair_route_discontinuities import (
        _backup_file,
        _read_json,
        _read_jsonl,
        _sha256,
        _write_json,
        _write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from repair_route_discontinuities import (  # type: ignore[no-redef]
        _backup_file,
        _read_json,
        _read_jsonl,
        _sha256,
        _write_json,
        _write_jsonl,
    )


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token or "scene"


def _route_paths(release_root: Path) -> list[Path]:
    paths = sorted((release_root / "source" / "preset_routes").glob("*.json"))
    paths += sorted((release_root / "source" / "custom_routes").glob("*.json"))
    return paths


def _build_mapping(routes: list[tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for route_path, route in routes:
        groups[(str(route["scene_id"]), str(route["difficulty"]).lower())].append(
            (route_path, route)
        )

    mapping: dict[str, dict[str, Any]] = {}
    new_ids: set[str] = set()
    for (scene_id, difficulty), items in sorted(groups.items()):
        ranked = sorted(
            items,
            key=lambda item: (-float(item[1]["path_length"]), str(item[1]["trajectory_id"])),
        )
        for ordinal, (route_path, route) in enumerate(ranked, start=1):
            old_id = str(route["trajectory_id"])
            new_id = f"{_safe_token(scene_id)}_{difficulty}_{ordinal:04d}"
            if new_id in new_ids:
                raise ValueError(f"Canonical trajectory collision: {new_id}")
            new_ids.add(new_id)
            mapping[old_id] = {
                "old_trajectory_id": old_id,
                "new_trajectory_id": new_id,
                "scene_id": scene_id,
                "difficulty": str(route["difficulty"]),
                "ordinal": ordinal,
                "path_length": float(route["path_length"]),
                "old_route_path": route_path,
                "new_route_path": route_path.with_name(f"{new_id}.json"),
                "old_source_root": route.get("source_root"),
                "old_source_episode": route.get("source_episode"),
                "new_source_root": difficulty,
                "new_source_episode": f"{scene_id}/{difficulty}_{ordinal:04d}",
            }
    return mapping


def _canonical_image_ref(scene_id: str, new_id: str, old_ref: str) -> str:
    filename = Path(old_ref.replace("\\", "/")).name
    return f"source/frames/{scene_id}/{new_id}/{filename}"


def _move_frame_directories(
    release_root: Path,
    mapping: dict[str, dict[str, Any]],
) -> dict[str, str]:
    renamed_prefixes: dict[str, str] = {}
    staged: list[tuple[Path, Path, Path]] = []
    old_directories = {
        (
            release_root
            / "source"
            / "frames"
            / str(item["scene_id"])
            / old_id
        ).resolve()
        for old_id, item in mapping.items()
    }
    for old_id, item in mapping.items():
        scene_id = str(item["scene_id"])
        new_id = str(item["new_trajectory_id"])
        old_dir = release_root / "source" / "frames" / scene_id / old_id
        new_dir = release_root / "source" / "frames" / scene_id / new_id
        old_prefix = old_dir.relative_to(release_root).as_posix() + "/"
        new_prefix = new_dir.relative_to(release_root).as_posix() + "/"
        renamed_prefixes[old_prefix] = new_prefix
        if not old_dir.exists() or old_dir.resolve() == new_dir.resolve():
            continue
        # Re-canonicalization can permute names that already exist (for
        # example hard_0001 and hard_0002 exchanging ranks).  Such targets are
        # safe because every old directory is staged before any final rename.
        if new_dir.exists() and new_dir.resolve() not in old_directories:
            raise FileExistsError(f"Canonical frame directory already exists: {new_dir}")
        stage_dir = old_dir.with_name(
            f".canonicalize-{hashlib.sha1(old_id.encode()).hexdigest()[:12]}"
        )
        if stage_dir.exists():
            raise FileExistsError(f"Staging directory already exists: {stage_dir}")
        staged.append((old_dir, stage_dir, new_dir))

    for old_dir, stage_dir, _ in staged:
        old_dir.rename(stage_dir)
    for _, stage_dir, new_dir in staged:
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir.rename(new_dir)
    return renamed_prefixes


def _rewrite_checksums(
    release_root: Path,
    *,
    renamed_prefixes: dict[str, str],
    old_route_paths: set[str],
    modified_paths: set[Path],
) -> None:
    checksum_path = release_root / "checksums.sha256"
    if not checksum_path.exists():
        return
    modified_rel = {
        path.resolve().relative_to(release_root.resolve()).as_posix()
        for path in modified_paths
        if path.exists()
    }
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        relative = relative.strip()
        if relative in old_route_paths:
            continue
        for old_prefix, new_prefix in renamed_prefixes.items():
            if relative.startswith(old_prefix):
                relative = new_prefix + relative[len(old_prefix) :]
                break
        if (release_root / relative).exists():
            entries[relative] = digest
    for relative in modified_rel:
        entries[relative] = _sha256(release_root / relative)
    checksum_path.write_text(
        "".join(f"{entries[path]}  {path}\n" for path in sorted(entries)),
        encoding="utf-8",
    )


def canonicalize_release(
    release_root: Path,
    *,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    if backup_root is not None:
        backup_root = backup_root.resolve()
        if backup_root == release_root or release_root in backup_root.parents:
            raise ValueError("backup_root must be outside release_root")

    routes = [(path, _read_json(path)) for path in _route_paths(release_root)]
    mapping = _build_mapping(routes)
    mismatches_before = sum(
        str(route["difficulty"]).lower() not in str(route["trajectory_id"]).lower().split("_")
        for _, route in routes
    )
    if not apply:
        return {
            "release_root": str(release_root),
            "applied": False,
            "route_count": len(routes),
            "mismatches_before": mismatches_before,
            "mapping": [
                {key: value for key, value in item.items() if not key.endswith("_path")}
                for item in mapping.values()
            ],
        }

    manifest_path = release_root / "manifest.json"
    modified_paths: set[Path] = set()
    old_route_paths: set[str] = set()
    old_episode_to_new: dict[str, str] = {}

    episodes_path = release_root / "metadata" / "episodes.jsonl"
    for row in _read_jsonl(episodes_path):
        old_id = str(row.get("trajectory_id"))
        if old_id in mapping:
            old_episode_to_new[str(row.get("episode_id"))] = str(
                mapping[old_id]["new_trajectory_id"]
            )

    renamed_prefixes = _move_frame_directories(release_root, mapping)

    staged_routes: list[tuple[Path, Path]] = []
    for old_path, route in routes:
        old_id = str(route["trajectory_id"])
        item = mapping[old_id]
        new_id = str(item["new_trajectory_id"])
        new_path = Path(item["new_route_path"])
        _backup_file(old_path, release_root, backup_root)
        provenance = dict(route.get("provenance", {}))
        provenance.setdefault("original_trajectory_id", old_id)
        provenance.setdefault("original_source_root", item["old_source_root"])
        provenance.setdefault("original_source_episode", item["old_source_episode"])
        route["provenance"] = provenance
        route["trajectory_id"] = new_id
        route["source_root"] = item["new_source_root"]
        route["source_episode"] = item["new_source_episode"]
        for state in route.get("states", []):
            old_ref = str(state.get("image", ""))
            if not old_ref:
                continue
            raw = dict(state.get("raw", {}))
            # Route-level provenance already preserves the source identifier.
            # Keeping a stale image path here creates a second, dangling path
            # field after canonical renaming.
            raw.pop("original_image", None)
            state["raw"] = raw
            state["image"] = _canonical_image_ref(str(route["scene_id"]), new_id, old_ref)
        stage_path = old_path.with_name(
            f".canonical-route-{hashlib.sha1(old_id.encode()).hexdigest()[:12]}.json"
        )
        if stage_path.exists():
            raise FileExistsError(f"Staging route already exists: {stage_path}")
        _write_json(stage_path, route)
        staged_routes.append((stage_path, new_path))
        modified_paths.add(new_path)
        old_route_paths.add(old_path.relative_to(release_root).as_posix())

    # All route payloads are staged before removing the old namespace.  This
    # prevents one canonical target from overwriting another route that still
    # needs to be renamed.
    for old_path, _ in routes:
        old_path.unlink()
    for stage_path, new_path in staged_routes:
        if new_path.exists():
            raise FileExistsError(f"Canonical route already exists: {new_path}")
        stage_path.rename(new_path)

    metadata_paths = sorted((release_root / "metadata").glob("*.jsonl"))
    metadata_paths += sorted((release_root / "splits").glob("*.jsonl"))
    for path in metadata_paths:
        _backup_file(path, release_root, backup_root)
        rows = _read_jsonl(path)
        changed = False
        for row in rows:
            old_id = str(row.get("trajectory_id", ""))
            item = mapping.get(old_id)
            if item is not None:
                new_id = str(item["new_trajectory_id"])
                provenance = dict(row.get("provenance", {}))
                provenance.setdefault("original_trajectory_id", old_id)
                provenance.setdefault("original_episode_id", row.get("episode_id"))
                provenance.setdefault("original_source_root", row.get("source_root"))
                provenance.setdefault("original_source_episode", row.get("source_episode"))
                row["provenance"] = provenance
                row["trajectory_id"] = new_id
                if "episode_id" in row:
                    row["episode_id"] = new_id
                if "route_file" in row:
                    route_dir = "custom_routes" if row.get("source") == "custom" else "preset_routes"
                    row["route_file"] = f"source/{route_dir}/{new_id}.json"
                if "source_root" in row:
                    row["source_root"] = item["new_source_root"]
                if "source_episode" in row:
                    row["source_episode"] = item["new_source_episode"]
                changed = True
                continue
            old_episode_id = str(row.get("episode_id", ""))
            if old_episode_id in old_episode_to_new:
                row.setdefault("provenance", {})["original_episode_id"] = old_episode_id
                row["episode_id"] = old_episode_to_new[old_episode_id]
                changed = True
        if changed:
            _write_jsonl(path, rows)
            modified_paths.add(path)

    if manifest_path.exists():
        _backup_file(manifest_path, release_root, backup_root)
        manifest = _read_json(manifest_path)
        manifest["naming_convention"] = {
            "trajectory_id": "<Scene>_<easy|normal|hard>_<four-digit ordinal>",
            "episode_id": "same_as_trajectory_id",
            "ordinal_order": "path_length_descending_then_original_trajectory_id",
            "difficulty_consistent": True,
        }
        _write_json(manifest_path, manifest)
        modified_paths.add(manifest_path)

    _rewrite_checksums(
        release_root,
        renamed_prefixes=renamed_prefixes,
        old_route_paths=old_route_paths,
        modified_paths=modified_paths,
    )
    return {
        "release_root": str(release_root),
        "applied": True,
        "route_count": len(routes),
        "mismatches_before": mismatches_before,
        "renamed_frame_directories": sum(
            1
            for old_prefix, new_prefix in renamed_prefixes.items()
            if old_prefix != new_prefix
            and (release_root / new_prefix).parent.exists()
        ),
        "mapping": [
            {key: value for key, value in item.items() if not key.endswith("_path")}
            for item in mapping.values()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize trajectory, episode, route, source, and RGB names by difficulty."
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = canonicalize_release(
        args.release_root,
        apply=args.apply,
        backup_root=args.backup_root,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
