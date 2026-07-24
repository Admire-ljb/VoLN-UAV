# Reported VoLN-UAV experimental results

> Source: `arxiv_reported`. These values reproduce the arXiv table; they are not substituted for missing evaluation runs.

## Validation-Seen

| Method | Difficulty | NE (m) ↓ | SR (%) ↑ | OSR (%) ↑ | nDTW (%) ↑ | SPL (%) ↑ |
|---|---|---:|---:|---:|---:|---:|
| Random | Easy | 268.5 | 0.6 | 1.8 | 28.3 | 0.4 |
| Random | Normal | 308.7 | 0.0 | 0.8 | 20.0 | 0.0 |
| Random | Hard | 398.9 | 0.0 | 0.2 | 12.1 | 0.0 |
| Seq2Seq-VG | Easy | 205.8 | 1.2 | 5.4 | 30.1 | 0.9 |
| Seq2Seq-VG | Normal | 251.6 | 0.5 | 2.9 | 21.0 | 0.3 |
| Seq2Seq-VG | Hard | 307.4 | 0.1 | 1.0 | 11.8 | 0.1 |
| CMA-VG | Easy | 170.2 | 1.9 | 7.6 | 34.5 | 1.3 |
| CMA-VG | Normal | 211.9 | 0.9 | 4.3 | 25.2 | 0.6 |
| CMA-VG | Hard | 261.3 | 0.2 | 1.9 | 16.1 | 0.1 |
| LAG-VG | Easy | 118.7 | 2.8 | 7.8 | 29.7 | 1.9 |
| LAG-VG | Normal | 154.9 | 1.5 | 4.6 | 20.4 | 1.0 |
| LAG-VG | Hard | 203.6 | 0.5 | 1.9 | 12.7 | 0.3 |
| VoLN-MLLM | Easy | 92.4 | 8.7 | 13.4 | 54.8 | 6.5 |
| VoLN-MLLM | Normal | 126.8 | 5.4 | 10.6 | 40.9 | 3.8 |
| VoLN-MLLM | Hard | 171.5 | 2.1 | 4.1 | 25.7 | 1.4 |

## Test-Unseen

| Method | Difficulty | NE (m) ↓ | SR (%) ↑ | OSR (%) ↑ | nDTW (%) ↑ | SPL (%) ↑ |
|---|---|---:|---:|---:|---:|---:|
| Random | Easy | 270.1 | 0.4 | 1.4 | 30.1 | 0.3 |
| Random | Normal | 310.4 | 0.0 | 0.6 | 22.7 | 0.0 |
| Random | Hard | 395.2 | 0.0 | 0.2 | 15.1 | 0.0 |
| Seq2Seq-VG | Easy | 208.6 | 1.0 | 4.8 | 28.9 | 0.7 |
| Seq2Seq-VG | Normal | 254.8 | 0.4 | 2.5 | 21.4 | 0.3 |
| Seq2Seq-VG | Hard | 309.9 | 0.1 | 0.9 | 13.0 | 0.0 |
| CMA-VG | Easy | 174.5 | 1.6 | 6.5 | 33.2 | 1.1 |
| CMA-VG | Normal | 216.8 | 0.8 | 3.9 | 26.4 | 0.6 |
| CMA-VG | Hard | 266.1 | 0.2 | 1.7 | 18.5 | 0.1 |
| LAG-VG | Easy | 122.4 | 2.3 | 6.4 | 28.1 | 1.5 |
| LAG-VG | Normal | 158.3 | 1.2 | 3.8 | 20.5 | 0.7 |
| LAG-VG | Hard | 206.7 | 0.4 | 1.7 | 14.0 | 0.2 |
| VoLN-MLLM | Easy | 97.1 | 7.4 | 14.6 | 53.1 | 5.7 |
| VoLN-MLLM | Normal | 131.4 | 4.5 | 10.1 | 41.2 | 3.2 |
| VoLN-MLLM | Hard | 176.8 | 1.8 | 4.5 | 28.0 | 1.3 |

## Test-Unseen ablations

| Variant | CT (s) ↓ | EER (%) ↓ | NE (m) ↓ | SR (%) ↑ | OSR (%) ↑ | nDTW (%) ↑ | SPL (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VoLN-MLLM | 1.42 | 0.5 | 119.0 | 5.7 | 11.8 | 45.8 | 4.3 |
| No-Align | 1.36 | 0.9 | 162.8 | 2.3 | 7.1 | 29.6 | 1.2 |
| No-LoRA | 1.45 | 5.8 | 176.9 | 2.8 | 7.8 | 27.2 | 1.4 |
| CLIP-Input | 1.98 | 1.5 | 158.6 | 2.9 | 8.2 | 30.9 | 1.6 |
