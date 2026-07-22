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
    kwargs: dict[str, Any] = {
        "dino_encoder": dino_encoder,
        "adapter": adapter,
        "semantic_bank": semantic_bank,
        "embed_dim": int(model_cfg["embed_dim"]),
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "num_heads": int(model_cfg.get("num_heads", 8)),
        "num_layers": int(model_cfg.get("num_layers", 6)),
        "lora_rank": int(model_cfg["lora_rank"]),
        "horizon": int(model_cfg["horizon"]),
        "top_k_semantic": int(model_cfg["top_k_semantic"]),
        "cache_image_embeddings": cache_image_embeddings,
    }
    if planner_cls is VoLNPlanner:
        kwargs.update(
            planner_backbone=model_cfg.get("planner_backbone"),
            lora_enabled=bool(model_cfg.get("lora_enabled", True)),
            lora_alpha=model_cfg.get("lora_alpha"),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules"),
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        )
    return planner_cls(
        **kwargs,
    )
