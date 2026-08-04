# VoLN-adapted learned baselines

This repository includes three optional learned baselines adapted from common VLN-style navigation families:

- `Seq2Seq-VG` (`model.planner_variant: seq2seq`)
- `CMA-VG` (`model.planner_variant: cma`)
- `LAG-VG` (`model.planner_variant: lag`)

`VG` means visual-goal adapted. These are not direct language-instruction replicas. The original language tokens are replaced by the VoLN-UAV visual-goal inputs already used by the planner pipeline:

- current observation image embedding
- history image embeddings and proprioception
- visual goal image embedding
- retrieved semantic tokens from the semantic bank
- current proprioception / odometry vector

All three baselines use the same paper action interface as the main planner:

- eight body-frame relative three-dimensional waypoints
- one stop logit

## Methods

### Seq2Seq-VG

`Seq2Seq-VG` uses a GRU encoder over visual-proprioceptive history and a recurrent decoder conditioned on the current image, three goal views, retrieved semantic tokens, and current proprioception.

### CMA-VG

`CMA-VG` uses cross-modal attention. The current visual-goal state is the query; history tokens, goal token, semantic tokens, and proprioception token are the memory.

### LAG-VG

`LAG-VG` is a landmark/goal-guided attention baseline. It separately attends to history and semantic landmark tokens, then gates between history, semantic-landmark, and goal features before predicting waypoints.

## Training

Use the existing planner training entry point and switch only `CONFIG`.

```powershell
cd VoLN-UAV

$env:CONFIG="configs\train_seq2seq_dataset_release.yaml"
.\scripts\run_train_planner.cmd

$env:CONFIG="configs\train_cma_dataset_release.yaml"
.\scripts\run_train_planner.cmd

$env:CONFIG="configs\train_lag_dataset_release.yaml"
.\scripts\run_train_planner.cmd
```

Ubuntu:

```bash
cd VoLN-UAV

CONFIG=configs/train_seq2seq_dataset_release.yaml ./scripts/train_planner.sh
CONFIG=configs/train_cma_dataset_release.yaml ./scripts/train_planner.sh
CONFIG=configs/train_lag_dataset_release.yaml ./scripts/train_planner.sh
```

Each method writes to a separate run directory:

- `runs/planner_seq2seq_vg`
- `runs/planner_cma_vg`
- `runs/planner_lag_vg`

## Offline evaluation

```powershell
$env:CONFIG="configs\eval_offline_seq2seq_dataset_release.yaml"
.\scripts\run_eval_offline.cmd

$env:CONFIG="configs\eval_offline_cma_dataset_release.yaml"
.\scripts\run_eval_offline.cmd

$env:CONFIG="configs\eval_offline_lag_dataset_release.yaml"
.\scripts\run_eval_offline.cmd
```

Ubuntu:

```bash
CONFIG=configs/eval_offline_seq2seq_dataset_release.yaml ./scripts/eval_offline.sh
CONFIG=configs/eval_offline_cma_dataset_release.yaml ./scripts/eval_offline.sh
CONFIG=configs/eval_offline_lag_dataset_release.yaml ./scripts/eval_offline.sh
```

## AirSim evaluation

```powershell
$env:CONFIG="configs\eval_airsim_seq2seq_dataset_release.yaml"
.\scripts\run_airsim_eval.cmd

$env:CONFIG="configs\eval_airsim_cma_dataset_release.yaml"
.\scripts\run_airsim_eval.cmd

$env:CONFIG="configs\eval_airsim_lag_dataset_release.yaml"
.\scripts\run_airsim_eval.cmd
```

Ubuntu:

```bash
CONFIG=configs/eval_airsim_seq2seq_dataset_release.yaml ./scripts/run_airsim_eval.sh
CONFIG=configs/eval_airsim_cma_dataset_release.yaml ./scripts/run_airsim_eval.sh
CONFIG=configs/eval_airsim_lag_dataset_release.yaml ./scripts/run_airsim_eval.sh
```

Set `AIRSIM_IP` and `AIRSIM_PORT` when the Ubuntu evaluation client connects
to a simulator running on another machine.

The release evaluation configs use the same paper protocol as VoLN-MLLM: at most 128 decisions, a 4 m three-dimensional goal region, and SR/SPL counted only when the policy explicitly stops inside the region. Each `planner_best.pt` stores the stop threshold calibrated on Validation-Seen.

All AirSim methods inherit `configs/airsim_active_beacon_protocol.yaml`.
Active beacons are generated from reference-route motion with the original
fixed-count selector. The release configuration uses three active beacons with
deterministic candidate sampling, spacing checks, and route-based fallback
positions. The target asset is placed separately and does not contribute to the
active-beacon count.
