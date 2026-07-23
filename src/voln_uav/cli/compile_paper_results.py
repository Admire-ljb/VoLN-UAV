from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from voln_uav.common.io import ensure_dir, read_json, write_json


METRICS = ("NE", "SR", "OSR", "nDTW", "SPL")
METHOD_SLUGS = {
    "Random": "random",
    "Seq2Seq-VG": "seq2seq_vg",
    "CMA-VG": "cma",
    "LAG-VG": "lag",
    "VoLN-MLLM": "voln_mllm",
}
SPLIT_SLUGS = {
    "Validation-Seen": "validation_seen",
    "Test-Unseen": "test_unseen",
}


def load_reported_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Paper results must be a mapping: {path}")
    if payload.get("source") != "manuscript_reported":
        raise ValueError("paper_results.yaml must identify its source as manuscript_reported")
    difficulties = list(payload.get("difficulties", []))
    if difficulties != ["Easy", "Normal", "Hard"]:
        raise ValueError("Paper result difficulty order must be Easy, Normal, Hard")
    for split_name, methods in payload.get("main_results", {}).items():
        for method_name, metrics in methods.items():
            for metric in METRICS:
                values = metrics.get(metric)
                if not isinstance(values, list) or len(values) != len(difficulties):
                    raise ValueError(
                        f"{split_name}/{method_name}/{metric} must have "
                        f"{len(difficulties)} values"
                    )
    return payload


def long_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    difficulties = list(results["difficulties"])
    for split_name, methods in results["main_results"].items():
        for method_name, metrics in methods.items():
            for index, difficulty in enumerate(difficulties):
                rows.append(
                    {
                        "source": results["source"],
                        "split": split_name,
                        "method": method_name,
                        "difficulty": difficulty,
                        **{metric: float(metrics[metric][index]) for metric in METRICS},
                    }
                )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source", "split", "method", "difficulty", *METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)


def _main_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Manuscript-reported VoLN-UAV results",
        "",
        "> Source: `manuscript_reported`. These values reproduce the paper table; "
        "they are not substituted for missing evaluation runs.",
        "",
    ]
    difficulties = list(results["difficulties"])
    for split_name, methods in results["main_results"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Method | Difficulty | NE (m) ↓ | SR (%) ↑ | OSR (%) ↑ | nDTW (%) ↑ | SPL (%) ↑ |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method_name, metrics in methods.items():
            for index, difficulty in enumerate(difficulties):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            method_name,
                            difficulty,
                            f"{float(metrics['NE'][index]):.1f}",
                            f"{float(metrics['SR'][index]):.1f}",
                            f"{float(metrics['OSR'][index]):.1f}",
                            f"{float(metrics['nDTW'][index]):.1f}",
                            f"{float(metrics['SPL'][index]):.1f}",
                        ]
                    )
                    + " |"
                )
        lines.append("")

    ablations = results["ablation_results"]
    lines.extend(
        [
            "## Test-Unseen ablations",
            "",
            "| Variant | CT (s) ↓ | EER (%) ↓ | NE (m) ↓ | SR (%) ↑ | OSR (%) ↑ | nDTW (%) ↑ | SPL (%) ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in ablations["rows"].items():
        lines.append(
            f"| {name} | {row['CT']:.2f} | {row['EER']:.1f} | {row['NE']:.1f} | "
            f"{row['SR']:.1f} | {row['OSR']:.1f} | {row['nDTW']:.1f} | {row['SPL']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _actual_run_dir(
    runs_root: Path,
    backend: str,
    split_name: str,
    method_name: str,
) -> Path:
    return runs_root / (
        f"eval_{backend}_{METHOD_SLUGS[method_name]}_"
        f"{SPLIT_SLUGS[split_name]}_paper"
    )


def _compare_run(
    metrics: dict[str, Any],
    reported: dict[str, list[float]],
    difficulties: list[str],
    tolerance: float,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    by_difficulty = metrics.get("by_difficulty", {})
    for index, difficulty in enumerate(difficulties):
        actual_group = by_difficulty.get(difficulty)
        if not isinstance(actual_group, dict):
            comparisons.append(
                {
                    "difficulty": difficulty,
                    "status": "missing",
                    "reason": "metrics.json has no by_difficulty entry",
                }
            )
            continue
        for metric in METRICS:
            actual = actual_group.get(metric)
            if actual is None:
                comparisons.append(
                    {
                        "difficulty": difficulty,
                        "metric": metric,
                        "status": "missing",
                    }
                )
                continue
            actual_value = float(actual)
            if metric != "NE":
                actual_value *= 100.0
            expected_value = float(reported[metric][index])
            absolute_error = abs(actual_value - expected_value)
            comparisons.append(
                {
                    "difficulty": difficulty,
                    "metric": metric,
                    "actual": actual_value,
                    "reported": expected_value,
                    "absolute_error": absolute_error,
                    "status": "match" if absolute_error <= tolerance else "mismatch",
                }
            )
    statuses = {item["status"] for item in comparisons}
    status = "match" if statuses == {"match"} else ("mismatch" if "mismatch" in statuses else "partial")
    return {"status": status, "comparisons": comparisons}


def inspect_runs(
    results: dict[str, Any],
    runs_root: Path | None,
    backend: str,
    tolerance: float,
) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "source": "evaluation_runs",
        "backend": backend,
        "runs_root": str(runs_root.resolve()) if runs_root is not None else None,
        "runs": {},
        "missing_runs": [],
        "matched_runs": [],
        "mismatched_runs": [],
    }
    if runs_root is None:
        coverage["status"] = "not_requested"
        return coverage

    for split_name, methods in results["main_results"].items():
        for method_name, reported in methods.items():
            key = f"{split_name}/{method_name}"
            run_dir = _actual_run_dir(runs_root, backend, split_name, method_name)
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                coverage["runs"][key] = {
                    "status": "skipped_missing",
                    "run_dir": str(run_dir.resolve()),
                }
                coverage["missing_runs"].append(key)
                continue
            comparison = _compare_run(
                read_json(metrics_path),
                reported,
                list(results["difficulties"]),
                tolerance,
            )
            comparison["run_dir"] = str(run_dir.resolve())
            comparison["metrics_file"] = str(metrics_path.resolve())
            coverage["runs"][key] = comparison
            if comparison["status"] == "match":
                coverage["matched_runs"].append(key)
            elif comparison["status"] == "mismatch":
                coverage["mismatched_runs"].append(key)

    coverage["status"] = (
        "mismatch"
        if coverage["mismatched_runs"]
        else ("complete" if not coverage["missing_runs"] else "partial")
    )
    return coverage


def _plot_test_unseen(results: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return []

    ensure_dir(output_dir)
    methods = list(results["main_results"]["Test-Unseen"])
    difficulties = list(results["difficulties"])
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    generated: list[str] = []
    for metric in ("SR", "nDTW"):
        fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
        x = np.arange(len(methods))
        width = 0.22
        for index, difficulty in enumerate(difficulties):
            values = [
                float(results["main_results"]["Test-Unseen"][method][metric][index])
                for method in methods
            ]
            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=difficulty,
                color=colors[index],
                edgecolor="white",
                linewidth=0.7,
            )
        ax.set_xticks(x, methods)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"Test-Unseen {metric} by difficulty")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        ax.text(
            0.995,
            0.985,
            "Manuscript-reported results",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#666666",
        )
        stem = f"test_unseen_{metric.casefold()}"
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=220 if suffix == "png" else None)
            generated.append(str(path.resolve()))
        plt.close(fig)
    return generated


def compile_results(
    results_path: str | Path,
    output_dir: str | Path,
    runs_root: str | Path | None = None,
    backend: str = "airsim",
    tolerance: float = 0.15,
) -> dict[str, Any]:
    results = load_reported_results(results_path)
    output = ensure_dir(output_dir)
    rows = long_rows(results)
    csv_path = output / "paper_results_long.csv"
    markdown_path = output / "paper_results.md"
    json_path = output / "paper_results.json"
    coverage_path = output / "run_coverage.json"
    _write_csv(rows, csv_path)
    markdown_path.write_text(_main_markdown(results), encoding="utf-8")
    write_json(results, json_path)
    coverage = inspect_runs(
        results,
        Path(runs_root) if runs_root is not None else None,
        backend,
        tolerance,
    )
    write_json(coverage, coverage_path)
    figures = _plot_test_unseen(results, output / "figures")
    return {
        "source": results["source"],
        "csv": str(csv_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "json": str(json_path.resolve()),
        "coverage": str(coverage_path.resolve()),
        "figures": figures,
        "run_status": coverage["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export manuscript-reported tables/figures and compare any available "
            "evaluation runs. Missing runs are reported as skipped."
        )
    )
    parser.add_argument("--results", default="configs/paper_results.yaml")
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument("--runs-root", help="Optional root containing eval_<backend>_*_paper run directories.")
    parser.add_argument("--backend", choices=("offline", "airsim"), default="airsim")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Absolute comparison tolerance in metres for NE and percentage points for other metrics.",
    )
    parser.add_argument(
        "--strict-runs",
        action="store_true",
        help="Fail if a run is absent or differs from the manuscript-reported values.",
    )
    args = parser.parse_args()

    summary = compile_results(
        args.results,
        args.output_dir,
        runs_root=args.runs_root,
        backend=args.backend,
        tolerance=args.tolerance,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict_runs and summary["run_status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
