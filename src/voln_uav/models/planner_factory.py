from __future__ import annotations

from typing import Any

from torch import nn

from voln_uav.models.adapter import DINOToCLIPAdapter
from voln_uav.models.baseline_planners import CMAPlanner, LAGPlanner, Seq2SeqPlanner
from voln_uav.models.planner import VoLNPlanner
from voln_uav.models.semantic_bank import SemanticBank


PLANNER_VARIANTS = {
    "voln": VoLNPlanner,
    "ours": VoLNPlanner,
    "seq2seq": Seq2SeqPlanner,
    "cma": CMAPlanner,
    "lag": LAGPlanner,
}


def normalize_planner_variant(model_cfg: dict[str, Any]) -> str:
    variant = str(model_cfg.get("planner_variant", model_cfg.get("variant", "voln"))).lower()
    aliases = {
        "voln_uav": "voln",
        "voln-uav": "voln",
        "seq": "seq2seq",
        "seq-to-seq": "seq2seq",
    }
    return aliases.get(variant, variant)


def build_planner(
    model_cfg: dict[str, Any],
    dino_encoder: nn.Module,
    adapter: DINOToCLIPAdapter,
    semantic_bank: SemanticBank,
    cache_image_embeddings: bool = False,
) -> nn.Module:
    variant = normalize_planner_variant(model_cfg)
    try:
        planner_cls = PLANNER_VARIANTS[variant]
    except KeyError as exc:
        known = ", ".join(sorted(PLANNER_VARIANTS))
        raise ValueError(f"Unknown planner_variant={variant!r}. Expected one of: {known}") from exc
    return planner_cls(
        dino_encoder=dino_encoder,
        adapter=adapter,
        semantic_bank=semantic_bank,
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        lora_rank=int(model_cfg["lora_rank"]),
        horizon=int(model_cfg["horizon"]),
        top_k_semantic=int(model_cfg["top_k_semantic"]),
        cache_image_embeddings=cache_image_embeddings,
    )
