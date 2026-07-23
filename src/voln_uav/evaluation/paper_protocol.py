from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from voln_uav.common.io import read_jsonl, write_json


PAPER_SPLIT_KEYS = ("train", "validation_seen", "test_unseen")


def _normalise_scene_id(scene_id: str) -> str:
    return "".join(char for char in str(scene_id).casefold() if char.isalnum())


def load_paper_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Paper protocol must be a mapping: {path}")
    return payload


def resolve_protocol_path(config: dict[str, Any]) -> Path | None:
    configured = config.get("paper_protocol")
    if not configured:
        return None
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    config_dir = Path(config.get("_config_dir", "."))
    for root in (config_dir, config_dir.parent, Path.cwd()):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (config_dir / candidate).resolve()


def protocol_optional_scenes(config: dict[str, Any]) -> list[str]:
    explicit = list(config.get("optional_scene_ids", []) or [])
    protocol_path = resolve_protocol_path(config)
    if protocol_path is None or not protocol_path.exists():
        return explicit
    protocol = load_paper_protocol(protocol_path)
    combined = [*explicit, *(protocol.get("optional_scene_ids", []) or [])]
    return list(dict.fromkeys(str(scene) for scene in combined))


def select_available_episodes(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select requested scenes while reporting absent partial-release scenes.

    Missing optional scenes are metadata, not failed episodes. A strict caller
    can reject any absent requested scene with ``strict_scenes: true``.
    """

    available = sorted({str(episode["scene_id"]) for episode in episodes})
    available_by_key = {_normalise_scene_id(scene): scene for scene in available}
    requested = [str(scene) for scene in (config.get("scene_allowlist", []) or [])]
    optional = protocol_optional_scenes(config)
    optional_by_key = {_normalise_scene_id(scene): scene for scene in optional}

    missing_requested = [
        scene for scene in requested if _normalise_scene_id(scene) not in available_by_key
    ]
    missing_optional = [
        scene for scene in optional if _normalise_scene_id(scene) not in available_by_key
    ]
    missing_required = [
        scene
        for scene in missing_requested
        if _normalise_scene_id(scene) not in optional_by_key
    ]
    if bool(config.get("strict_scenes", False)) and missing_requested:
        raise ValueError(
            "Requested scenes are absent from the selected episode file: "
            + ", ".join(missing_requested)
        )

    requested_keys = {
        _normalise_scene_id(scene)
        for scene in requested
        if _normalise_scene_id(scene) in available_by_key
    }
    selected = [
        episode
        for episode in episodes
        if not requested
        or _normalise_scene_id(str(episode["scene_id"])) in requested_keys
    ]
    selected_scenes = sorted({str(episode["scene_id"]) for episode in selected})
    report = {
        "available_scenes": available,
        "requested_scenes": requested,
        "selected_scenes": selected_scenes,
        "missing_requested_scenes": missing_requested,
        "missing_required_scenes": missing_required,
        "missing_optional_scenes": missing_optional,
        "available_episodes": len(episodes),
        "selected_episodes": len(selected),
        "strict_scenes": bool(config.get("strict_scenes", False)),
    }
    return selected, report


def _resolve_split_file(
    benchmark_root: Path,
    split_spec: dict[str, Any],
) -> tuple[Path | None, list[str]]:
    candidates = [
        str(split_spec["file"]),
        *(str(item) for item in (split_spec.get("aliases", []) or [])),
    ]
    for relative in candidates:
        path = benchmark_root / relative
        if path.exists():
            return path, candidates
    return None, candidates


def inspect_benchmark_protocol(
    benchmark_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Read split manifests without changing them and audit paper compatibility."""

    root = Path(benchmark_root)
    split_reports: dict[str, Any] = {}
    split_scenes: dict[str, set[str]] = {}
    total_episodes = 0
    issues: list[str] = []
    warnings: list[str] = []

    for split_name in PAPER_SPLIT_KEYS:
        spec = dict(protocol["splits"][split_name])
        path, candidates = _resolve_split_file(root, spec)
        if path is None:
            split_reports[split_name] = {
                "status": "missing",
                "file_candidates": candidates,
                "expected_episodes": int(spec["expected_episodes"]),
            }
            split_scenes[split_name] = set()
            issues.append(f"Missing {split_name} episode file ({', '.join(candidates)})")
            continue

        episodes = read_jsonl(path)
        scenes = {str(episode["scene_id"]) for episode in episodes}
        difficulty_counts = Counter(str(episode.get("difficulty", "Unknown")) for episode in episodes)
        expected_episodes = int(spec["expected_episodes"])
        expected_environments = int(spec["expected_environments"])
        report = {
            "status": "complete" if len(episodes) == expected_episodes else "partial",
            "file": str(path.resolve()),
            "episodes": len(episodes),
            "expected_episodes": expected_episodes,
            "episode_count_matches": len(episodes) == expected_episodes,
            "scenes": sorted(scenes),
            "environment_count": len(scenes),
            "expected_environments": expected_environments,
            "environment_count_matches": len(scenes) == expected_environments,
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
        }
        split_reports[split_name] = report
        split_scenes[split_name] = scenes
        total_episodes += len(episodes)
        if report["status"] == "partial":
            warnings.append(
                f"{split_name} contains {len(episodes)} episodes; "
                f"the manuscript reports {expected_episodes}"
            )

    train_scenes = split_scenes["train"]
    validation_scenes = split_scenes["validation_seen"]
    test_scenes = split_scenes["test_unseen"]
    if train_scenes and validation_scenes and not validation_scenes.issubset(train_scenes):
        issues.append(
            "Validation-Seen contains environments not represented in Train: "
            + ", ".join(sorted(validation_scenes - train_scenes))
        )
    test_overlap = test_scenes & (train_scenes | validation_scenes)
    if test_overlap:
        issues.append(
            "Test-Unseen overlaps Train/Validation-Seen environments: "
            + ", ".join(sorted(test_overlap))
        )

    present_scenes = train_scenes | validation_scenes | test_scenes
    optional_scenes = [str(scene) for scene in protocol.get("optional_scene_ids", []) or []]
    present_keys = {_normalise_scene_id(scene) for scene in present_scenes}
    missing_optional = [
        scene for scene in optional_scenes if _normalise_scene_id(scene) not in present_keys
    ]
    expected_total = int(protocol["dataset"]["expected_episodes"])
    report = {
        "protocol": str(protocol.get("name", "VoLN-UAV paper protocol")),
        "benchmark_root": str(root.resolve()),
        "status": "ready" if not issues and total_episodes == expected_total else "partial",
        "splits": split_reports,
        "total_episodes": total_episodes,
        "expected_total_episodes": expected_total,
        "total_count_matches": total_episodes == expected_total,
        "unique_environments": len(present_scenes),
        "expected_environments": int(protocol["dataset"]["expected_environments"]),
        "missing_optional_scenes": missing_optional,
        "warnings": warnings,
        "issues": issues,
    }
    return report


def write_protocol_report(report: dict[str, Any], path: str | Path) -> None:
    write_json(report, path)
