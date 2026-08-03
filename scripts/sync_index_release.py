from __future__ import annotations

import argparse
import json
import shutil
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


def _write_all_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == checksum_path:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{_sha256(path)}  {relative}\n")
    checksum_path.write_text("".join(rows), encoding="utf-8")


def _scene_count(root: Path) -> int:
    return len(_read_jsonl(root / "source" / "scenes.jsonl"))


def sync_index_release(
    full_root: Path,
    index_root: Path,
    *,
    apply: bool,
    backup_root: Path | None = None,
    rewrite_full_checksums: bool = True,
) -> dict[str, Any]:
    full_root = full_root.resolve()
    index_root = index_root.resolve()
    if full_root == index_root:
        raise ValueError("full_root and index_root must differ")
    full_routes = sorted((full_root / "source" / "preset_routes").glob("*.json"))
    index_routes = sorted((index_root / "source" / "preset_routes").glob("*.json"))
    if not apply:
        return {
            "full_root": str(full_root),
            "index_root": str(index_root),
            "applied": False,
            "full_route_count": len(full_routes),
            "index_route_count": len(index_routes),
            "full_scene_count": _scene_count(full_root),
            "index_scene_count": _scene_count(index_root),
        }
    if backup_root is None:
        raise ValueError("backup_root is required when applying")
    backup_root = backup_root.resolve()
    if backup_root == index_root or index_root in backup_root.parents:
        raise ValueError("backup_root must be outside index_root")
    backup_root.mkdir(parents=True, exist_ok=True)

    full_manifest_path = full_root / "manifest.json"
    _backup_file(full_manifest_path, full_root, backup_root / "full_manifest")
    full_manifest = _read_json(full_manifest_path)
    full_manifest["num_scenes"] = _scene_count(full_root)
    _write_json(full_manifest_path, full_manifest)
    if rewrite_full_checksums:
        _write_all_checksums(full_root)

    for relative in (Path("source/preset_routes"), Path("source/custom_routes")):
        source = full_root / relative
        target = index_root / relative
        backup = backup_root / "index_before_sync" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if backup.exists():
                raise FileExistsError(f"Backup target already exists: {backup}")
            shutil.move(str(target), str(backup))
        shutil.copytree(source, target)

    benchmark_source = full_root / "benchmark"
    benchmark_target = index_root / "benchmark"
    benchmark_backup = backup_root / "index_before_sync" / "benchmark"
    if benchmark_target.exists():
        benchmark_backup.parent.mkdir(parents=True, exist_ok=True)
        if benchmark_backup.exists():
            raise FileExistsError(f"Backup target already exists: {benchmark_backup}")
        shutil.move(str(benchmark_target), str(benchmark_backup))
    if benchmark_source.exists():
        shutil.copytree(benchmark_source, benchmark_target)

    for relative in (Path("metadata"), Path("splits")):
        source_dir = full_root / relative
        target_dir = index_root / relative
        for source in sorted(source_dir.glob("*.jsonl")):
            target = target_dir / source.name
            _backup_file(target, index_root, backup_root / "index_before_sync")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    scenes_source = full_root / "source" / "scenes.jsonl"
    scenes_target = index_root / "source" / "scenes.jsonl"
    _backup_file(scenes_target, index_root, backup_root / "index_before_sync")
    shutil.copy2(scenes_source, scenes_target)

    readme_source = full_root / "README.md"
    readme_target = index_root / "README.md"
    if readme_source.is_file():
        _backup_file(readme_target, index_root, backup_root / "index_before_sync")
        shutil.copy2(readme_source, readme_target)

    source_index_path = index_root / "metadata" / "source_data_index.jsonl"
    source_rows = _read_jsonl(source_index_path)
    for row in source_rows:
        row["asset_mode"] = "index"
    _write_jsonl(source_index_path, source_rows)

    index_manifest_path = index_root / "manifest.json"
    _backup_file(index_manifest_path, index_root, backup_root / "index_before_sync")
    index_manifest = dict(full_manifest)
    index_manifest["asset_mode"] = "index"
    index_manifest.pop("zip_path", None)
    _write_json(index_manifest_path, index_manifest)
    _write_all_checksums(index_root)

    return {
        "full_root": str(full_root),
        "index_root": str(index_root),
        "applied": True,
        "route_count": len(full_routes),
        "scene_count": _scene_count(full_root),
        "episode_count": len(_read_jsonl(index_root / "metadata" / "episodes.jsonl")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize a metadata-only index release from a canonical full RGB release."
    )
    parser.add_argument("--full-root", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument(
        "--skip-full-checksums",
        action="store_true",
        help="Reuse an already refreshed checksum file for the full RGB release.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = sync_index_release(
        args.full_root,
        args.index_root,
        apply=args.apply,
        backup_root=args.backup_root,
        rewrite_full_checksums=not args.skip_full_checksums,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
