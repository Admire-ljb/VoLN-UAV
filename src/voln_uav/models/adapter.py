from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class DINOToCLIPAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        # Follow the DINOv3-CLIP adapter design: optional bottleneck MLP + LayerNorm.
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return torch.nn.functional.normalize(y, dim=-1)



def cosine_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(student, teacher, dim=-1).mean()



def save_adapter(adapter: DINOToCLIPAdapter, path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "meta": meta}, path)



def load_adapter(
    path: str | Path,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    map_location: str = "cpu",
    expected_alignment_mode: str | None = None,
) -> DINOToCLIPAdapter:
    ckpt = torch.load(path, map_location=map_location)
    if expected_alignment_mode is not None:
        meta = ckpt.get("meta", {})
        trained_mode = str(
            meta.get(
                "alignment_mode",
                meta.get("config", {}).get("alignment_mode", "clip_distill"),
            )
        ).lower()
        if trained_mode != str(expected_alignment_mode).lower():
            raise ValueError(
                "Adapter checkpoint/alignment mismatch: "
                f"checkpoint={trained_mode!r}, expected={expected_alignment_mode!r}"
            )
    adapter = DINOToCLIPAdapter(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter
