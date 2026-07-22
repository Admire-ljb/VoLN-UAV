from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from voln_uav.models.adapter import load_adapter
from voln_uav.models.encoders import build_image_encoder


def build_planner_vision(
    model_cfg: dict[str, Any],
    adapter_ckpt: str | Path | None,
    map_location: str = "cpu",
) -> tuple[nn.Module, nn.Module, int]:
    """Build the paper visual input path without changing the dataset.

    ``dino_aligned`` is the main method and No-Align path. ``clip`` is the
    CLIP-Input ablation, where CLIP features enter the planner directly and no
    DINO-to-CLIP adapter is loaded.
    """
    embed_dim = int(model_cfg["embed_dim"])
    image_size = int(model_cfg.get("image_size", 224))
    vision_input = str(model_cfg.get("vision_input", "dino_aligned")).lower()
    if vision_input == "clip":
        encoder = build_image_encoder(model_cfg["clip_image_encoder"], out_dim=embed_dim, image_size=image_size)
        return encoder, nn.Identity(), embed_dim
    if vision_input != "dino_aligned":
        raise ValueError(f"Unsupported vision_input={vision_input!r}")
    if adapter_ckpt is None:
        raise ValueError("adapter_ckpt is required for vision_input=dino_aligned")
    dino_dim = int(model_cfg.get("dino_dim", embed_dim))
    encoder = build_image_encoder(model_cfg["dino_backbone"], out_dim=dino_dim, image_size=image_size)
    adapter = load_adapter(
        adapter_ckpt,
        in_dim=dino_dim,
        hidden_dim=int(model_cfg["adapter_hidden"]),
        out_dim=embed_dim,
        map_location=map_location,
        expected_alignment_mode="no_align" if model_cfg.get("ablation") == "no_align" else "clip_distill",
    )
    return encoder, adapter, dino_dim
