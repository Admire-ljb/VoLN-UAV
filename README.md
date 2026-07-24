<div align="center">

# VoLN: Vision-Only Long-Horizon Navigation—Paradigm, Benchmark, and Method

<p align="center">
  <a href="https://admire-ljb.github.io/VoLN-UAV/">🌐 <strong>Project Page</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://arxiv.org/pdf/2607.21400">📄 <strong>Paper</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Louj/VoLN-UAV-dataset">🤗 <strong>Dataset</strong></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Louj/VoLN-UAV-ENV">🧭 <strong>Simulator Environments</strong></a>
</p>

</div>

## Reproduction Scope

This README is the reproducibility guide for the released VoLN training and evaluation code. For the method overview, headline results, dataset visualizations, qualitative cases, and simulation/real-world videos, visit the [project page](https://admire-ljb.github.io/VoLN-UAV/).

The repository covers:

- VoLN-MLLM adapter and planner training;
- VoLN-adapted Seq2Seq-VG, CMA-VG, and LAG-VG baselines;
- AirSim closed-loop paper evaluation and offline route-replay diagnostics;
- the No-Align, No-LoRA, and CLIP-Input ablations;
- metric reporting for NE, SR, OSR, nDTW, SPL, CT, and EER.

The [navigation dataset](https://huggingface.co/datasets/Louj/VoLN-UAV-dataset) and [simulator environments](https://huggingface.co/datasets/Louj/VoLN-UAV-ENV) are released separately and are not duplicated in this repository.

## Experimental Protocol

### Dataset splits

The manuscript uses **7,210 episodes**. Validation-Seen contains disjoint
trajectories from five environments represented in the training pool, while
Test-Unseen contains five held-out environments:

| Split | Episodes | Ratio | Evaluation name |
|---|---:|---:|---|
| Train | 5,047 | 70% | Train |
| Validation-Seen | 1,082 | 15% | Validation-Seen |
| Test-Unseen | 1,081 | 15% | Test-Unseen |

Difficulty is defined by reference path length:

- **Easy:** less than 300 m
- **Normal:** 300–450 m
- **Hard:** at least 450 m

### Model stages

VoLN-MLLM has two stages:

1. **Visual-semantic alignment.** A lightweight adapter maps frozen DINOv3 ViT-B/16 features into the frozen CLIP ViT-B/16 image-embedding space using cosine distillation.
2. **Trajectory planning.** A frozen Vicuna-7B-v1.5 backbone jointly encodes aligned observation history, three terminal goal views, proprioception, and top-\(k\) category tokens retrieved from the fixed semantic bank. The released configuration uses \(k=8\). Rank-16 LoRA modules adapt its attention and feed-forward projections; learned heads predict eight body-frame relative 3D waypoints and a stop signal.

### Evaluation rules

The trained baselines are visual-goal adaptations of instruction-following navigation models. All methods receive the same VoLN observations and share waypoint supervision, action interface, stopping rule, and evaluation protocol.

The policy receives onboard RGB and deployable body-frame proprioception; world-frame poses remain supervision/evaluation metadata. Training samples use the final three consecutive RGB frames as the visual goal and body-frame relative waypoint targets. The paper protocol uses at most 128 decisions per episode and a 4 m three-dimensional goal region. SR and SPL require the policy to issue an explicit stop inside that region; OSR records whether the executed trajectory enters it at any time. The stop threshold is calibrated on Validation-Seen and stored in `planner_best.pt`.

### Configuration index

| Experiment | Configuration or launcher |
|---|---|
| Adapter training | `configs/train_adapter_dataset_release.yaml` |
| Planner training | `configs/train_planner_dataset_release.yaml` |
| Offline route-replay diagnostic | `configs/eval_offline_dataset_release.yaml` |
| AirSim evaluation | `configs/eval_airsim_dataset_release.yaml` |
| Paper ablations | `scripts/run_paper_ablations.py` |
| Paper evaluation suite | `scripts/run_paper_evaluation.py` |
| Paper protocol audit | `scripts/validate_paper_protocol.py` |
| Experiment tables and figures | `scripts/compile_experiment_results.py` |
| Seq2Seq-VG / CMA-VG / LAG-VG | [Baseline documentation](docs/voln_adapted_baselines.md) |

## Installation

~~~bash
conda create -n voln-uav python=3.10 -y
conda activate voln-uav
pip install -e .
~~~

Install the CUDA-specific PyTorch build first if the default wheel does not match your system. The default package installation includes the training, AirSim, real-world, and plotting dependencies used by the released scripts.

## Dataset Preparation

Download the [navigation dataset](https://huggingface.co/datasets/Louj/VoLN-UAV-dataset) and extract the metadata and required split shards into one directory. Download the [simulator environments](https://huggingface.co/datasets/Louj/VoLN-UAV-ENV) separately for online AirSim evaluation.

Audit the existing split manifests without changing the dataset:

~~~bash
python scripts/validate_paper_protocol.py \
  --benchmark-root D:/VoLN_dataset/VoLN-UAV-Dataset-release-full/benchmark \
  --protocol configs/paper_protocol.yaml \
  --out results/local_protocol_coverage.json
~~~

The complete internal benchmark has 7,210 episodes in 17 environments. A
public or locally selected subset may be smaller. The audit marks such a
release as `partial`, verifies the available split invariants, and reports
absent optional environments such as Campus, Park, Tunnel, and Ruins. Missing
environments are skipped; they are not converted into zero-valued episodes.
Add `--strict` when a complete paper-scale release is required.

Set <code>source_root</code>, <code>output_root</code>, and the episode-level <code>split_manifest</code> in <code>configs/benchmark_dataset_release.yaml</code>. The split manifest assigns each episode to Train, Validation-Seen, or Test-Unseen and stores shortest-path provenance for SPL. Then build the benchmark:

~~~bash
python -m voln_uav.cli.build_benchmark --config configs/benchmark_dataset_release.yaml
~~~

## Training

Run the complete real-data pipeline:

~~~bash
python scripts/run_dataset_release_pipeline.py --device cuda
~~~

Run selected stages when resuming or debugging:

~~~bash
python scripts/run_dataset_release_pipeline.py --stages build train-adapter train-planner --device cuda
~~~

The VoLN-adapted baselines have separate training entry points and checkpoints. See [baseline documentation](docs/voln_adapted_baselines.md) for Seq2Seq-VG, CMA-VG, and LAG-VG.

Run the three manuscript ablations (`No-Align`, `No-LoRA`, and `CLIP-Input`) with their independent checkpoints:

~~~bash
python scripts/run_paper_ablations.py --stages train airsim --device cuda
~~~

`No-Align` saves an untrained dimensional adapter without CLIP-teacher supervision, `No-LoRA` freezes Vicuna without inserting LoRA branches, and `CLIP-Input` feeds frozen CLIP ViT-B/16 image features directly to the planner.

## Evaluation

Offline route-replay diagnostic:

~~~bash
python -m voln_uav.cli.eval_offline --config configs/eval_offline_dataset_release.yaml --device cuda
~~~

This diagnostic replays observations from the recorded route. Manuscript tables use the AirSim closed-loop backend below.

AirSim preflight and closed-loop evaluation:

~~~bash
python -m voln_uav.cli.eval_airsim --config configs/eval_airsim_dataset_release.yaml --preflight
python -m voln_uav.cli.eval_airsim --config configs/eval_airsim_dataset_release.yaml --device cuda
~~~

Run all manuscript methods on Validation-Seen and Test-Unseen:

~~~bash
python scripts/run_paper_evaluation.py \
  --methods random seq2seq_vg cma lag voln_mllm \
  --splits validation_seen test_unseen \
  --device cuda
~~~

The manuscript launcher verifies the complete split before evaluation. For a
selected-scene diagnostic on a partial release, run:

~~~bash
python -m voln_uav.cli.eval_airsim \
  --config configs/eval_airsim_dataset_release.yaml \
  --split test_unseen \
  --scenes Campus Park Tunnel Ruins \
  --allow-partial-diagnostic \
  --device cuda
~~~

Each run writes `scene_coverage.json`. Add `--strict-scenes` to reject a missing
requested scene.

On Windows, the reference and random online baselines can be evaluated with the same launcher:

~~~powershell
cd D:\VoLN_dataset\github-VoLN-UAV
$env:BASELINE="reference"
$env:TRIALS="10"
.\scripts\run_online_baseline.cmd --episode-index 0 --episode-stride 1 --reference-stride 1 --control-mode teleport --fast-reset --settle-sec 0.0 --work-dir D:\VoLN_dataset\VoLN-UAV-runs\reference_test_10_fast
~~~

Use <code>scripts\report_metrics.cmd</code> to summarize a run directory with the paper metrics.

## Experimental Results and Consistency Checks

`configs/experiment_results.yaml` is the machine-readable source for the numbers
reported in the paper. Generated closed-loop logs are compared with this table
and summarized separately in `run_coverage.json`.

The committed result package is not limited to YAML. It includes normalized
JSON, long-form and wide-form CSV files, rendered Markdown tables, PNG/PDF
figures, run coverage, and per-metric comparison intermediates:

~~~text
results/experiments/
  experiment_results.json
  experiment_results.md
  experiment_results_long.csv
  run_coverage.json
  intermediate/
    README.md
    main_results_wide.csv
    ablation_results.csv
    run_comparison.csv
    result_manifest.json
  figures/
    test_unseen_sr.{png,pdf}
    test_unseen_ndtw.{png,pdf}
~~~

Export the experiment tables and plots:

~~~bash
python scripts/compile_experiment_results.py \
  --results configs/experiment_results.yaml \
  --output-dir results/experiments
~~~

Compare available closed-loop runs against the reported table:

~~~bash
python scripts/compile_experiment_results.py \
  --results configs/experiment_results.yaml \
  --output-dir results/experiments \
  --runs-root D:/VoLN_dataset/VoLN-UAV-runs \
  --backend airsim
~~~

Missing run directories are listed as `skipped_missing` in
`run_coverage.json`. Use `--strict-runs` for release verification that must
include every method and split.

| Test-Unseen SR | Test-Unseen nDTW |
|---|---|
| ![Test-Unseen SR](results/experiments/figures/test_unseen_sr.png) | ![Test-Unseen nDTW](results/experiments/figures/test_unseen_ndtw.png) |

## Repository Structure

~~~text
src/voln_uav/
  benchmark/      Benchmark construction, visual goals, and beacon augmentation
  data/           Dataset loaders and release packaging
  models/         DINO–CLIP adapter, semantic bank, planners, and LoRA modules
  training/       Adapter/planner training and DAgger-style collection
  evaluation/     Offline and online metrics
  simulators/     Route replay and AirSim interfaces
  cli/            Command-line entry points
configs/          Dataset, training, and evaluation configurations
scripts/          Reproducible training and evaluation launchers
airsim_plugin/    Unreal/AirSim scene utilities
docs/             Project page, demonstrations, and baseline documentation
~~~

## Citation

If you find this work helpful, please consider citing our paper:

~~~bibtex
@article{lou2026voln,
  title  = {VoLN: Vision-Only Long-Horizon Navigation---Paradigm, Benchmark, and Method},
  author = {Lou, Jiabin and Wang, Haopeng and Wang, Yuanshuai and Liu, Xinyu and Lv, Xuxin and Guo, Yuxin and Huang, Lei and Shi, Rongye and Wu, Wenjun},
  journal = {arXiv preprint arXiv:2607.21400},
  year   = {2026},
  eprint = {2607.21400},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url    = {https://arxiv.org/abs/2607.21400}
}
~~~

## Acknowledgement

We thank the authors of TravelUAV and AirVLN for releasing their codebases and providing useful engineering references for UAV navigation research.
