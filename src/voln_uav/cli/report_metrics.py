from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from voln_uav.common.io import read_json, read_jsonl, write_json
from voln_uav.evaluation.metrics import METRIC_KEYS, aggregate_by_difficulty, aggregate_metrics


PAPER_COLUMNS = ("NE", "SR", "OSR", "nDTW", "SPL", "CT_mean", "CT_p95", "EER", "collisions")


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        if isinstance(metrics, dict):
            return metrics
    return {}


def _load_details(run_dir: Path) -> list[dict[str, Any]]:
    details_path = run_dir / "details.jsonl"
    if not details_path.exists():
        return []
    return read_jsonl(details_path)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(0.95 * max(len(values) - 1, 0)), max(len(values) - 1, 0))
    return values[idx]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics = _load_metrics(run_path)
    details = _load_details(run_path)

    summary: dict[str, Any] = {
        "run_dir": str(run_path.resolve()),
        "metrics_file": str((run_path / "metrics.json").resolve()) if (run_path / "metrics.json").exists() else None,
        "details_file": str((run_path / "details.jsonl").resolve()) if (run_path / "details.jsonl").exists() else None,
    }

    if details:
        detail_metrics = [{key: float(item[key]) for key in METRIC_KEYS} for item in details if all(key in item for key in METRIC_KEYS)]
        summary.update(aggregate_metrics(detail_metrics))
        summary["episodes"] = len(details)
        summary["by_difficulty"] = aggregate_by_difficulty(details)
        cycle_times = [float(v) for item in details for v in item.get("cycle_times", [])]
        if cycle_times:
            summary["CT_mean"] = _mean(cycle_times)
            summary["CT_p95"] = _p95(cycle_times)
        cycle_errors = sum(int(item.get("cycle_errors", 0)) for item in details)
        num_cycles = sum(int(item.get("num_cycles", len(item.get("cycle_times", [])))) for item in details)
        if num_cycles:
            summary["EER"] = cycle_errors / num_cycles
        if any("collisions" in item for item in details):
            summary["collisions"] = sum(int(item.get("collisions", 0)) for item in details)

    for key, value in metrics.items():
        if key in {"details"}:
            continue
        if key not in summary or key in {"CT_mean", "CT_p95", "EER", "collisions", "by_difficulty"}:
            summary[key] = value

    return summary


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number * 100.0:.2f}"
    if abs(number) >= 100.0:
        return f"{number:.2f}"
    return f"{number:.4f}"


def _row(name: str, metrics: dict[str, Any], percent_rates: bool) -> list[str]:
    out = [name, str(metrics.get("episodes", "-"))]
    for key in PAPER_COLUMNS:
        if key == "collisions" and key not in metrics:
            out.append("-")
        elif key in {"SR", "OSR", "nDTW", "SPL", "EER"}:
            out.append(_fmt(metrics.get(key), percent=percent_rates))
        else:
            out.append(_fmt(metrics.get(key)))
    return out


def format_markdown(summary: dict[str, Any], percent_rates: bool = True) -> str:
    headers = ["Split", "Episodes", *PAPER_COLUMNS]
    rows = [_row("Overall", summary, percent_rates)]
    for difficulty, item in (summary.get("by_difficulty") or {}).items():
        if isinstance(item, dict):
            rows.append(_row(str(difficulty), item, percent_rates))

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VoLN-UAV paper metrics from an evaluation run directory.")
    parser.add_argument("--run-dir", required=True, help="Directory containing metrics.json and/or details.jsonl.")
    parser.add_argument("--out-json", help="Optional path for a normalized metrics JSON file.")
    parser.add_argument("--out-md", help="Optional path for a Markdown paper table.")
    parser.add_argument("--raw-rates", action="store_true", help="Print SR/OSR/nDTW/SPL/EER as 0-1 values instead of percentages.")
    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    markdown = format_markdown(summary, percent_rates=not args.raw_rates)
    if args.out_json:
        write_json(summary, args.out_json)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown + "\n", encoding="utf-8")
    print(json.dumps({key: summary.get(key) for key in ["run_dir", "episodes", *PAPER_COLUMNS]}, indent=2, ensure_ascii=False))
    print(markdown)


if __name__ == "__main__":
    main()
