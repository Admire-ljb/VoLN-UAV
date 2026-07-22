from __future__ import annotations

import pytest
from torch import nn

from voln_uav.models.adapter import DINOToCLIPAdapter, save_adapter
from voln_uav.models.vision import build_planner_vision


def test_clip_input_builds_direct_encoder_without_adapter_checkpoint(monkeypatch) -> None:
    encoder = nn.Identity()
    monkeypatch.setattr("voln_uav.models.vision.build_image_encoder", lambda *_args, **_kwargs: encoder)

    actual_encoder, adapter, dim = build_planner_vision(
        {
            "vision_input": "clip",
            "clip_image_encoder": "open_clip:ViT-B-16:openai",
            "embed_dim": 512,
            "image_size": 224,
        },
        adapter_ckpt=None,
    )

    assert actual_encoder is encoder
    assert isinstance(adapter, nn.Identity)
    assert dim == 512


def test_no_align_rejects_distilled_adapter_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("voln_uav.models.vision.build_image_encoder", lambda *_args, **_kwargs: nn.Identity())
    checkpoint = tmp_path / "adapter.pt"
    save_adapter(
        DINOToCLIPAdapter(in_dim=8, hidden_dim=0, out_dim=8),
        checkpoint,
        meta={"alignment_mode": "clip_distill"},
    )

    with pytest.raises(ValueError, match="alignment mismatch"):
        build_planner_vision(
            {
                "vision_input": "dino_aligned",
                "ablation": "no_align",
                "dino_backbone": "dummy",
                "dino_dim": 8,
                "embed_dim": 8,
                "adapter_hidden": 0,
            },
            adapter_ckpt=checkpoint,
        )
