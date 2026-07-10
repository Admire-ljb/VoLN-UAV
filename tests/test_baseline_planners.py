from __future__ import annotations

import torch
from torch import nn

from voln_uav.models.baseline_planners import CMAPlanner, LAGPlanner, Seq2SeqPlanner
from voln_uav.models.planner_factory import build_planner, normalize_planner_variant
from voln_uav.models.semantic_bank import SemanticBank


class DummyEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(start_dim=1)


class DummyAdapter(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def _semantic_bank(embed_dim: int) -> SemanticBank:
    categories = ["target", "building", "tree", "road", "beacon"]
    embeddings = torch.randn(len(categories), embed_dim)
    return SemanticBank(categories=categories, embeddings=embeddings)


def _model_cfg(variant: str, embed_dim: int = 16, hidden_dim: int = 32, horizon: int = 5) -> dict[str, object]:
    return {
        "planner_variant": variant,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_heads": 4,
        "num_layers": 2,
        "lora_rank": 4,
        "horizon": horizon,
        "top_k_semantic": 3,
    }


def _batch(batch_size: int = 2, memory_len: int = 4, goal_count: int = 2, embed_dim: int = 16) -> dict[str, torch.Tensor]:
    return {
        "image_embedding": torch.randn(batch_size, embed_dim),
        "history_image_embeddings": torch.randn(batch_size, memory_len, embed_dim),
        "goal_image_embeddings": torch.randn(batch_size, goal_count, embed_dim),
        "history_proprio": torch.randn(batch_size, memory_len, 9),
        "proprio": torch.randn(batch_size, 9),
    }


def _assert_forward(planner: nn.Module, horizon: int = 5) -> None:
    out = planner(_batch())
    assert out["anchor"].shape == (2, 3)
    assert out["waypoints"].shape == (2, horizon, 3)
    assert out["stop_logit"].shape == (2,)
    assert len(out["semantic_names"]) == 2


def test_seq2seq_cma_lag_forward_shapes() -> None:
    embed_dim = 16
    kwargs = {
        "dino_encoder": DummyEncoder(),
        "adapter": DummyAdapter(),
        "semantic_bank": _semantic_bank(embed_dim),
        "embed_dim": embed_dim,
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "lora_rank": 4,
        "horizon": 5,
        "top_k_semantic": 3,
    }

    for planner_cls in (Seq2SeqPlanner, CMAPlanner, LAGPlanner):
        _assert_forward(planner_cls(**kwargs))


def test_planner_factory_selects_voln_adapted_baselines() -> None:
    embed_dim = 16
    bank = _semantic_bank(embed_dim)

    for variant, expected_cls in {
        "seq2seq": Seq2SeqPlanner,
        "cma": CMAPlanner,
        "lag": LAGPlanner,
    }.items():
        planner = build_planner(
            model_cfg=_model_cfg(variant, embed_dim=embed_dim),
            dino_encoder=DummyEncoder(),
            adapter=DummyAdapter(),
            semantic_bank=bank,
        )
        assert isinstance(planner, expected_cls)
        _assert_forward(planner)


def test_planner_variant_aliases() -> None:
    assert normalize_planner_variant({"planner_variant": "seq"}) == "seq2seq"
    assert normalize_planner_variant({"planner_variant": "seq-to-seq"}) == "seq2seq"
    assert normalize_planner_variant({"planner_variant": "voln-uav"}) == "voln"
