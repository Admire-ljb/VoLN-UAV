from __future__ import annotations

import csv
from pathlib import Path

from voln_uav.cli.compile_paper_results import compile_results, load_reported_results


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "configs" / "paper_results.yaml"


def test_reported_results_match_manuscript_values() -> None:
    results = load_reported_results(RESULTS)

    assert results["source"] == "manuscript_reported"
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

    assert summary["source"] == "manuscript_reported"
    assert summary["run_status"] == "partial"
    assert (output / "paper_results.md").exists()
    assert (output / "paper_results.json").exists()
    assert (output / "run_coverage.json").exists()

    with (output / "paper_results_long.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    assert all(row["source"] == "manuscript_reported" for row in rows)
