<div align="center">

# VoLN: Vision-Only Long-Horizon Navigation—Paradigm, Benchmark, and Method

### Official Training and Evaluation Repository

<p>
  Jiabin Lou &nbsp;·&nbsp; Haopeng Wang &nbsp;·&nbsp; Yuanshuai Wang &nbsp;·&nbsp; Xinyu Liu &nbsp;·&nbsp; Xuxin Lv &nbsp;·&nbsp; Yuxin Guo &nbsp;·&nbsp; Lei Huang &nbsp;·&nbsp; Rongye Shi &nbsp;·&nbsp; Wenjun Wu<sup>*</sup><br>
  Beihang University, Beijing, China<br>
  <sup>*</sup> Corresponding author
</p>

[Project Page](https://admire-ljb.github.io/VoLN-UAV/) ·
[Paper](paperv1.pdf) ·
[Dataset](https://huggingface.co/datasets/Louj/VoLN-UAV-dataset) ·
[Simulator Environments](https://huggingface.co/datasets/Louj/VoLN-UAV-ENV) ·
[Documentation](docs/voln_adapted_baselines.md)

</div>

## Reproduction Scope

This README is the reproducibility guide for the released VoLN training and evaluation code. For the method overview, headline results, dataset visualizations, qualitative cases, and simulation/real-world videos, visit the [project page](https://admire-ljb.github.io/VoLN-UAV/).

The repository covers:

- VoLN-MLLM adapter and planner training;
- VoLN-adapted Seq2Seq-VG, CMA-VG, and LAG-VG baselines;
- paper-protocol offline and AirSim closed-loop evaluation;
- the No-Align, No-LoRA, and CLIP-Input ablations;
- metric reporting for NE, SR, OSR, nDTW, SPL, CT, and EER.

The [navigation dataset](https://huggingface.co/datasets/Louj/VoLN-UAV-dataset) and [simulator environments](https://huggingface.co/datasets/Louj/VoLN-UAV-ENV) are released separately and are not duplicated in this repository.

## Experimental Protocol

### Dataset splits

The manuscript uses **7,210 episodes** with scene-level disjoint splits:

| Split | Episodes | Ratio | Evaluation name |
|---|---:|---:|---|
| Train | 5,047 | 70% | Train |
| Validation | 1,082 | 15% | Validation-Seen |
| Test | 1,081 | 15% | Test-Unseen |

Difficulty is defined by reference path length:

- **Easy:** less than 300 m
- **Normal:** 300–450 m
- **Hard:** at least 450 m

### Model stages

VoLN-MLLM has two stages:

1. **Visual-semantic alignment.** A lightweight adapter maps frozen DINOv3 ViT-B/16 features into the frozen CLIP ViT-B/16 image-embedding space using cosine distillation.
2. **Trajectory planning.** A frozen Vicuna-7B-v1.5 backbone jointly encodes aligned observation history, goal views, proprioception, and top-8 category tokens retrieved from the fixed semantic bank. Rank-16 LoRA modules adapt its attention and feed-forward projections; learned heads predict eight 3D waypoints and a stop signal.

### Evaluation rules

The trained baselines are visual-goal adaptations of instruction-following navigation models. All methods receive the same VoLN observations and share waypoint supervision, action interface, stopping rule, and evaluation protocol.

The paper protocol uses at most 128 decisions per episode and a 4 m three-dimensional goal region. SR and SPL require the policy to issue an explicit stop inside that region; OSR records whether the executed trajectory enters it at any time. The stop threshold is calibrated on Validation-Seen and stored in `planner_best.pt`.

### Configuration index

| Experiment | Configuration or launcher |
|---|---|
| Adapter training | `configs/train_adapter_dataset_release.yaml` |
| Planner training | `configs/train_planner_dataset_release.yaml` |
| Offline evaluation | `configs/eval_offline_dataset_release.yaml` |
| AirSim evaluation | `configs/eval_airsim_dataset_release.yaml` |
| Paper ablations | `scripts/run_paper_ablations.py` |
| Paper evaluation suite | `scripts/run_paper_evaluation.py` |
| Seq2Seq-VG / CMA-VG / LAG-VG | [Baseline documentation](docs/voln_adapted_baselines.md) |

## Installation

~~~bash
conda create -n voln-uav python=3.10 -y
conda activate voln-uav
pip install -r requirement.txt
pip install -e .[real]
~~~

Install the CUDA-specific PyTorch build first if the default wheel does not match your system.

## Dataset Preparation

Download the [navigation dataset](https://huggingface.co/datasets/Louj/VoLN-UAV-dataset) and extract the metadata and required split shards into one directory. Download the [simulator environments](https://huggingface.co/datasets/Louj/VoLN-UAV-ENV) separately for online AirSim evaluation.

Set <code>source_root</code> and <code>output_root</code> in <code>configs/benchmark_dataset_release.yaml</code>, then build the benchmark:

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
python scripts/run_dataset_release_pipeline.py --stages build train-adapter train-planner offline-eval --device cuda
~~~

The VoLN-adapted baselines have separate training entry points and checkpoints. See [baseline documentation](docs/voln_adapted_baselines.md) for Seq2Seq-VG, CMA-VG, and LAG-VG.

Run the three manuscript ablations (`No-Align`, `No-LoRA`, and `CLIP-Input`) with their independent checkpoints:

~~~bash
python scripts/run_paper_ablations.py --stages train offline --device cuda
~~~

`No-Align` saves an untrained dimensional adapter without CLIP-teacher supervision, `No-LoRA` freezes Vicuna without inserting LoRA branches, and `CLIP-Input` feeds frozen CLIP ViT-B/16 image features directly to the planner.

## Evaluation

Offline evaluation:

~~~bash
python -m voln_uav.cli.eval_offline --config configs/eval_offline_dataset_release.yaml --device cuda
~~~

AirSim preflight and closed-loop evaluation:

~~~bash
python -m voln_uav.cli.eval_airsim --config configs/eval_airsim_dataset_release.yaml --preflight
python -m voln_uav.cli.eval_airsim --config configs/eval_airsim_dataset_release.yaml --device cuda
~~~

On Windows, the reference and random online baselines can be evaluated with the same launcher:

~~~powershell
cd D:\VoLN_dataset\github-VoLN-UAV
$env:BASELINE="reference"
$env:TRIALS="10"
.\scripts\run_online_baseline.cmd --episode-index 0 --episode-stride 1 --reference-stride 1 --control-mode teleport --fast-reset --settle-sec 0.0 --work-dir D:\VoLN_dataset\VoLN-UAV-runs\reference_test_10_fast
~~~

Use <code>scripts\report_metrics.cmd</code> to summarize a run directory with the paper metrics.

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

Please cite the preprint as follows. The arXiv identifier will be added after registration.

~~~bibtex
@misc{lou2026voln,
  title  = {VoLN: Vision-Only Long-Horizon Navigation---Paradigm, Benchmark, and Method},
  author = {Lou, Jiabin and Wang, Haopeng and Wang, Yuanshuai and Liu, Xinyu and Lv, Xuxin and Guo, Yuxin and Huang, Lei and Shi, Rongye and Wu, Wenjun},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/Admire-ljb/VoLN-UAV}
}
~~~

## Acknowledgement

We thank the authors of TravelUAV and AirVLN for releasing their codebases and providing useful engineering references for UAV navigation research.
