from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from voln_uav.common.io import read_jsonl, write_json
from voln_uav.benchmark.splitter import trajectory_fingerprint


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
    split_episodes: dict[str, list[dict[str, Any]]] = {}
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
            split_episodes[split_name] = []
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
        split_episodes[split_name] = episodes
        total_episodes += len(episodes)
        if not report["episode_count_matches"]:
            issues.append(
                f"{split_name} contains {len(episodes)} episodes; "
                f"the manuscript reports {expected_episodes}"
            )
        if not report["environment_count_matches"]:
            issues.append(
                f"{split_name} contains {len(scenes)} environments; "
                f"the manuscript reports {expected_environments}"
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

    all_episode_ids: list[str] = []
    for episodes in split_episodes.values():
        all_episode_ids.extend(str(episode["episode_id"]) for episode in episodes)
    duplicate_episode_ids = sorted(
        episode_id
        for episode_id, count in Counter(all_episode_ids).items()
        if count > 1
    )
    if duplicate_episode_ids:
        issues.append(
            f"Episode IDs occur in more than one split ({len(duplicate_episode_ids)} duplicates)"
        )

    def route_identity(episode: dict[str, Any]) -> str | None:
        if episode.get("trajectory_hash"):
            return str(episode["trajectory_hash"])
        if episode.get("states"):
            return trajectory_fingerprint(episode)
        if episode.get("trajectory_id"):
            return f"{episode['scene_id']}:{episode['trajectory_id']}"
        return None

    train_routes = {
        identity
        for episode in split_episodes["train"]
        if (identity := route_identity(episode)) is not None
    }
    val_routes = {
        identity
        for episode in split_episodes["validation_seen"]
        if (identity := route_identity(episode)) is not None
    }
    if split_episodes["train"] and len(train_routes) != len(split_episodes["train"]):
        issues.append("Train episodes lack trajectory IDs/hashes required for split auditing")
    if split_episodes["validation_seen"] and len(val_routes) != len(split_episodes["validation_seen"]):
        issues.append("Validation-Seen episodes lack trajectory IDs/hashes required for split auditing")
    if train_routes & val_routes:
        issues.append("Train and Validation-Seen contain overlapping trajectories")

    train_sources = {
        str(episode["scene_source"])
        for episode in split_episodes["train"]
        if episode.get("scene_source")
    }
    test_sources = {
        str(episode["scene_source"])
        for episode in split_episodes["test_unseen"]
        if episode.get("scene_source")
    }
    require_source_holdout = bool(protocol.get("splits", {}).get("require_held_out_scene_source", True))
    if require_source_holdout:
        if any(
            not episode.get("scene_source")
            for split_name in ("train", "test_unseen")
            for episode in split_episodes[split_name]
        ):
            issues.append("Train/Test-Unseen episodes lack scene_source for source-holdout auditing")
        elif train_sources & test_sources:
            issues.append("Test-Unseen scene sources overlap the training pool")

    present_scenes = train_scenes | validation_scenes | test_scenes
    optional_scenes = [str(scene) for scene in protocol.get("optional_scene_ids", []) or []]
    present_keys = {_normalise_scene_id(scene) for scene in present_scenes}
    missing_optional = [
        scene for scene in optional_scenes if _normalise_scene_id(scene) not in present_keys
    ]
    expected_total = int(protocol["dataset"]["expected_episodes"])
    expected_environment_count = int(protocol["dataset"]["expected_environments"])
    if total_episodes != expected_total:
        issues.append(
            f"Benchmark contains {total_episodes} episodes; the manuscript reports {expected_total}"
        )
    if len(present_scenes) != expected_environment_count:
        issues.append(
            f"Benchmark contains {len(present_scenes)} unique environments; "
            f"the manuscript reports {expected_environment_count}"
        )

    difficulty_counts = Counter(
        str(episode.get("difficulty", "Unknown"))
        for episodes in split_episodes.values()
        for episode in episodes
    )
    difficulty_mix = dict(protocol.get("dataset", {}).get("difficulty_mix", {}) or {})
    tolerance = float(protocol.get("dataset", {}).get("difficulty_mix_tolerance", 0.01))
    if total_episodes and difficulty_mix:
        for difficulty, expected_ratio in difficulty_mix.items():
            actual_ratio = difficulty_counts.get(str(difficulty), 0) / total_episodes
            if abs(actual_ratio - float(expected_ratio)) > tolerance:
                issues.append(
                    f"{difficulty} difficulty ratio is {actual_ratio:.4f}; "
                    f"expected {float(expected_ratio):.4f}±{tolerance:.4f}"
                )
    report = {
        "protocol": str(protocol.get("name", "VoLN-UAV paper protocol")),
        "benchmark_root": str(root.resolve()),
        "status": "ready" if not issues else "partial",
        "splits": split_reports,
        "total_episodes": total_episodes,
        "expected_total_episodes": expected_total,
        "total_count_matches": total_episodes == expected_total,
        "unique_environments": len(present_scenes),
        "expected_environments": expected_environment_count,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "missing_optional_scenes": missing_optional,
        "warnings": warnings,
        "issues": issues,
    }
    return report


def write_protocol_report(report: dict[str, Any], path: str | Path) -> None:
    write_json(report, path)


def require_paper_protocol_ready(
    benchmark_root: str | Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Run the full benchmark gate before a paper-result evaluation."""
    if not bool(config.get("strict_paper_protocol", False)):
        return None
    protocol_path = resolve_protocol_path(config)
    if protocol_path is None or not protocol_path.exists():
        raise ValueError("strict_paper_protocol requires a valid paper_protocol YAML")
    report = inspect_benchmark_protocol(
        benchmark_root,
        load_paper_protocol(protocol_path),
    )
    if report["status"] != "ready":
        preview = "\n- ".join(str(issue) for issue in report["issues"][:12])
        raise ValueError("Benchmark is not paper-protocol ready:\n- " + preview)
    return report


def require_full_paper_split_selection(
    config: dict[str, Any],
    protocol_report: dict[str, Any] | None,
    selected_episodes: list[dict[str, Any]],
) -> None:
    """Require a complete manuscript split when strict evaluation is enabled."""
    if protocol_report is None:
        return
    configured_name = Path(str(config["episodes_file"])).name
    matching_split: dict[str, Any] | None = None
    matching_name: str | None = None
    for split_name, split_report in protocol_report["splits"].items():
        reported_file = split_report.get("file")
        if reported_file and Path(str(reported_file)).name == configured_name:
            matching_split = split_report
            matching_name = split_name
            break
    if matching_split is None:
        raise ValueError(
            "Strict paper evaluation requires the complete Validation-Seen or "
            f"Test-Unseen split, not {configured_name!r}"
        )
    expected = int(matching_split["expected_episodes"])
    if len(selected_episodes) != expected:
        raise ValueError(
            f"Strict paper evaluation selected {len(selected_episodes)} of {expected} "
            f"{matching_name} episodes. Use diagnostic mode for scene, difficulty, "
            "index, stride, trial, or episode-limit subsets."
        )
