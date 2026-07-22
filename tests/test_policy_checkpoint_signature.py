from __future__ import annotations

import pytest
import torch

from voln_uav.models.policy import VoLNPolicy
from voln_uav.training.planner_trainer import PlannerTrainer


def test_policy_rejects_checkpoint_from_different_ablation(tmp_path) -> None:
    checkpoint = tmp_path / "planner.pt"
    torch.save(
        {
            "state_dict": {},
            "meta": {
                "config": {
                    "model": {
                        "planner_variant": "voln",
                        "vision_input": "dino_aligned",
                        "horizon": 8,
                    }
                }
            },
        },
        checkpoint,
    )
    config = {
        "benchmark_root": str(tmp_path),
        "model": {
            "planner_variant": "voln",
            "vision_input": "clip",
            "horizon": 8,
            "embed_dim": 512,
            "memory_len": 8,
        },
    }

    with pytest.raises(ValueError, match="vision_input"):
        VoLNPolicy(
            config=config,
            semantic_bank_path=tmp_path / "categories.txt",
            adapter_ckpt=None,
            planner_ckpt=checkpoint,
        )


def test_resume_reads_validation_loss_from_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "planner_best.pt"
    torch.save({"state_dict": {}, "meta": {"val_total": 1.25}}, checkpoint)
    trainer = PlannerTrainer.__new__(PlannerTrainer)

    assert trainer._load_meta_val(checkpoint) == 1.25
    assert trainer._load_meta_val(tmp_path / "missing.pt") == float("inf")
