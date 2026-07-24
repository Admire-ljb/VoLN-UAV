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
        raise ValueError(f"Experiment results must be a mapping: {path}")
    if payload.get("source") != "arxiv_reported":
        raise ValueError("experiment_results.yaml must identify its source as arxiv_reported")
    difficulties = list(payload.get("difficulties", []))
    if difficulties != ["Easy", "Normal", "Hard"]:
        raise ValueError("Experiment result difficulty order must be Easy, Normal, Hard")
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


def wide_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    difficulties = list(results["difficulties"])
    for split_name, methods in results["main_results"].items():
        for method_name, metrics in methods.items():
            row: dict[str, Any] = {
                "source": results["source"],
                "split": split_name,
                "method": method_name,
            }
            for metric in METRICS:
                for index, difficulty in enumerate(difficulties):
                    row[f"{metric}_{difficulty}"] = float(metrics[metric][index])
            rows.append(row)
    return rows


def ablation_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    ablations = results["ablation_results"]
    return [
        {
            "source": results["source"],
            "split": ablations["split"],
            "variant": variant,
            **{key: float(value) for key, value in metrics.items()},
        }
        for variant, metrics in ablations["rows"].items()
    ]


def comparison_rows(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, run in coverage.get("runs", {}).items():
        split_name, method_name = key.split("/", 1)
        comparisons = run.get("comparisons")
        if not isinstance(comparisons, list):
            rows.append(
                {
                    "split": split_name,
                    "method": method_name,
                    "difficulty": "",
                    "metric": "",
                    "status": run.get("status", "unknown"),
                    "actual": "",
                    "reported": "",
                    "absolute_error": "",
                    "reason": "",
                }
            )
            continue
        for item in comparisons:
            rows.append(
                {
                    "split": split_name,
                    "method": method_name,
                    "difficulty": item.get("difficulty", ""),
                    "metric": item.get("metric", ""),
                    "status": item.get("status", "unknown"),
                    "actual": item.get("actual", ""),
                    "reported": item.get("reported", ""),
                    "absolute_error": item.get("absolute_error", ""),
                    "reason": item.get("reason", ""),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _intermediate_readme() -> str:
    return """# Result compilation intermediates

These files are regenerated by `scripts/compile_experiment_results.py`.

- `main_results_wide.csv`: one row per split and method before table rendering.
- `ablation_results.csv`: normalized Test-Unseen ablation rows.
- `run_comparison.csv`: per-metric comparison with available `metrics.json` files.
- `result_manifest.json`: provenance, units, dimensions, row counts, and output inventory.

`arxiv_reported` rows reproduce the values reported in the paper.
Available evaluation runs are summarized separately in `run_comparison.csv`.
"""


def _main_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Reported VoLN-UAV experimental results",
        "",
        "> Source: `arxiv_reported`. These values reproduce the table reported "
        "in the paper; run comparisons are summarized separately.",
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
    if metrics.get("status") != "complete":
        return {
            "status": "invalid_protocol_run",
            "reason": f"run status is {metrics.get('status', '<missing>')!r}",
            "comparisons": [],
        }
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
            elif comparison["status"] in {"mismatch", "invalid_protocol_run"}:
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
            "Reported experimental results",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#666666",
        )
        stem = f"test_unseen_{metric.casefold()}"
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            save_kwargs: dict[str, Any] = {}
            if suffix == "png":
                save_kwargs["dpi"] = 220
            else:
                save_kwargs["metadata"] = {
                    "Creator": "VoLN-UAV result compiler",
                    "CreationDate": None,
                    "ModDate": None,
                }
            fig.savefig(path, **save_kwargs)
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
    intermediate = ensure_dir(output / "intermediate")
    csv_path = output / "experiment_results_long.csv"
    markdown_path = output / "experiment_results.md"
    json_path = output / "experiment_results.json"
    coverage_path = output / "run_coverage.json"
    wide_path = intermediate / "main_results_wide.csv"
    ablation_path = intermediate / "ablation_results.csv"
    comparison_path = intermediate / "run_comparison.csv"
    manifest_path = intermediate / "result_manifest.json"
    intermediate_readme_path = intermediate / "README.md"
    _write_csv(
        rows,
        csv_path,
        ["source", "split", "method", "difficulty", *METRICS],
    )
    markdown_path.write_text(_main_markdown(results), encoding="utf-8")
    write_json(results, json_path)
    coverage = inspect_runs(
        results,
        Path(runs_root) if runs_root is not None else None,
        backend,
        tolerance,
    )
    write_json(coverage, coverage_path)
    wide = wide_rows(results)
    wide_fields = ["source", "split", "method"]
    wide_fields.extend(
        f"{metric}_{difficulty}"
        for metric in METRICS
        for difficulty in results["difficulties"]
    )
    _write_csv(wide, wide_path, wide_fields)
    ablations = ablation_rows(results)
    _write_csv(
        ablations,
        ablation_path,
        ["source", "split", "variant", "CT", "EER", "NE", "SR", "OSR", "nDTW", "SPL"],
    )
    comparisons = comparison_rows(coverage)
    _write_csv(
        comparisons,
        comparison_path,
        [
            "split",
            "method",
            "difficulty",
            "metric",
            "status",
            "actual",
            "reported",
            "absolute_error",
            "reason",
        ],
    )
    manifest = {
        "source": results["source"],
        "units": results["units"],
        "splits": list(results["main_results"]),
        "methods": list(next(iter(results["main_results"].values()))),
        "difficulties": list(results["difficulties"]),
        "rows": {
            "long": len(rows),
            "wide": len(wide),
            "ablations": len(ablations),
            "run_comparisons": len(comparisons),
        },
        "run_status": coverage["status"],
        "outputs": {
            "long_table": "../experiment_results_long.csv",
            "wide_table": "main_results_wide.csv",
            "ablation_table": "ablation_results.csv",
            "run_comparison": "run_comparison.csv",
            "rendered_markdown": "../experiment_results.md",
            "normalized_json": "../experiment_results.json",
            "run_coverage": "../run_coverage.json",
        },
    }
    write_json(manifest, manifest_path)
    intermediate_readme_path.write_text(_intermediate_readme(), encoding="utf-8")
    figures = _plot_test_unseen(results, output / "figures")
    return {
        "source": results["source"],
        "csv": str(csv_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "json": str(json_path.resolve()),
        "coverage": str(coverage_path.resolve()),
        "intermediate": {
            "wide": str(wide_path.resolve()),
            "ablations": str(ablation_path.resolve()),
            "comparison": str(comparison_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "readme": str(intermediate_readme_path.resolve()),
        },
        "figures": figures,
        "run_status": coverage["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export reported experiment tables/figures and compare any available "
            "evaluation runs. Missing runs are reported as skipped."
        )
    )
    parser.add_argument("--results", default="configs/experiment_results.yaml")
    parser.add_argument("--output-dir", default="results/experiments")
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
        help="Fail if a run is absent or differs from the arXiv-reported experimental values.",
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
