from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from voln_uav.benchmark.builder import BenchmarkBuilder
from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed


def rebuild_release_benchmark(
    release_root: Path,
    *,
    base_config: Path,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    base_config = base_config.resolve()
    output_root = release_root / "benchmark"
    cfg = load_config(base_config)
    cfg.update(
        {
            "source_root": str(release_root / "source"),
            "output_root": str(output_root),
            "split_manifest": None,
            "strict_paper_protocol": False,
            "require_shortest_path": False,
        }
    )
    report: dict[str, Any] = {
        "release_root": str(release_root),
        "base_config": str(base_config),
        "output_root": str(output_root),
        "applied": apply,
        "active_beacon_counts": dict(cfg["beacons"]["count_by_path_length"]),
    }
    if not apply:
        return report
    if backup_root is None:
        raise ValueError("backup_root is required when applying")
    backup_root = backup_root.resolve()
    backup_target = backup_root / "benchmark_before_rebuild"
    if output_root.exists():
        if backup_target.exists():
            raise FileExistsError(f"Backup target already exists: {backup_target}")
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_root), str(backup_target))
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))
    summary = BenchmarkBuilder(cfg).build()
    report["summary"] = summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the released benchmark and episode-level active beacons."
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = rebuild_release_benchmark(
        args.release_root,
        base_config=args.base_config,
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
