from __future__ import annotations

import csv
from pathlib import Path

from voln_uav.cli.compile_experiment_results import compile_results, load_reported_results


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "configs" / "experiment_results.yaml"


def test_reported_results_match_manuscript_values() -> None:
    results = load_reported_results(RESULTS)

    assert results["source"] == "arxiv_reported"
    assert results["main_results"]["Test-Unseen"]["VoLN-MLLM"]["SR"] == [7.4, 4.5, 1.8]
    assert results["main_results"]["Test-Unseen"]["Random"]["nDTW"] == [30.1, 22.7, 15.1]
    assert results["main_results"]["Validation-Seen"]["LAG-VG"]["NE"] == [118.7, 154.9, 203.6]
    assert results["ablation_results"]["rows"]["No-LoRA"]["EER"] == 5.8


def test_compile_results_exports_tables_and_skips_absent_runs(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    summary = compile_results(
        RESULTS,
        output,
        runs_root=tmp_path / "runs",
        backend="airsim",
    )

    assert summary["source"] == "arxiv_reported"
    assert summary["run_status"] == "partial"
    assert (output / "experiment_results.md").exists()
    assert (output / "experiment_results.json").exists()
    assert (output / "run_coverage.json").exists()
    assert (output / "intermediate" / "main_results_wide.csv").exists()
    assert (output / "intermediate" / "ablation_results.csv").exists()
    assert (output / "intermediate" / "run_comparison.csv").exists()
    assert (output / "intermediate" / "result_manifest.json").exists()
    assert (output / "intermediate" / "README.md").exists()

    with (output / "experiment_results_long.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    assert all(row["source"] == "arxiv_reported" for row in rows)

    with (output / "intermediate" / "main_results_wide.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        wide_rows = list(csv.DictReader(stream))
    assert len(wide_rows) == 10

    with (output / "intermediate" / "ablation_results.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        ablation_rows = list(csv.DictReader(stream))
    assert len(ablation_rows) == 4
