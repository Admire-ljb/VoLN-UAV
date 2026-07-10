from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voln_uav.models.adapter import DINOToCLIPAdapter
from voln_uav.models.semantic_bank import SemanticBank


class BaselineVisionPlannerBase(nn.Module):
    """Shared visual-goal encoder used by the paper-style learned baselines."""

    def __init__(
        self,
        dino_encoder: nn.Module,
        adapter: DINOToCLIPAdapter,
        semantic_bank: SemanticBank,
        embed_dim: int,
        hidden_dim: int,
        horizon: int,
        top_k_semantic: int,
        cache_image_embeddings: bool = False,
        proprio_dim: int = 9,
    ) -> None:
        super().__init__()
        self.dino_encoder = dino_encoder
        self.adapter = adapter
        self.semantic_bank = semantic_bank
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.top_k_semantic = int(top_k_semantic)
        self.proprio_dim = int(proprio_dim)
        self.cache_image_embeddings = bool(cache_image_embeddings)
        self._image_embedding_cache: dict[str, torch.Tensor] = {}

        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = False
        self.dino_encoder.eval()
        self.adapter.eval()

    def train(self, mode: bool = True) -> "BaselineVisionPlannerBase":
        super().train(mode)
        self.dino_encoder.eval()
        self.adapter.eval()
        return self

    @staticmethod
    def _flatten_image_paths(paths: Any) -> list[str] | None:
        if paths is None:
            return None
        if isinstance(paths, (str, Path)):
            return [str(paths)]
        flat: list[str] = []
        for item in paths:
            if isinstance(item, (list, tuple)):
                flat.extend(str(p) for p in item)
            else:
                flat.append(str(item))
        return flat

    def _encode_uncached_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.adapter(self.dino_encoder(images))

    def _encode_flat_images(self, images: torch.Tensor, paths: list[str] | None) -> torch.Tensor:
        if not self.cache_image_embeddings or paths is None or len(paths) != images.shape[0]:
            return self._encode_uncached_images(images)

        outputs: list[torch.Tensor | None] = [None] * len(paths)
        pending: dict[str, list[int]] = {}
        for idx, path in enumerate(paths):
            cached = self._image_embedding_cache.get(path)
            if cached is None:
                pending.setdefault(path, []).append(idx)
            else:
                outputs[idx] = cached.to(device=images.device)

        if pending:
            missing_keys = list(pending.keys())
            first_indices = torch.tensor([pending[key][0] for key in missing_keys], device=images.device)
            encoded = self._encode_uncached_images(images.index_select(0, first_indices))
            for key, embedding in zip(missing_keys, encoded):
                detached = embedding.detach()
                self._image_embedding_cache[key] = detached.cpu()
                for idx in pending[key]:
                    outputs[idx] = detached

        if any(item is None for item in outputs):
            raise RuntimeError("Image embedding cache returned an incomplete batch")
        return torch.stack([item.to(device=images.device) for item in outputs if item is not None], dim=0)

    def encode_images(self, images: torch.Tensor, paths: Any = None) -> torch.Tensor:
        flat_paths = self._flatten_image_paths(paths)
        if images.ndim == 5:
            b, n, c, h, w = images.shape
            flat = images.reshape(b * n, c, h, w)
            aligned = self._encode_flat_images(flat, flat_paths)
            return aligned.view(b, n, -1)
        if images.ndim == 4:
            return self._encode_flat_images(images, flat_paths)
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    def common_embeddings(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "image_embedding" in batch:
            history_img_emb = batch["history_image_embeddings"]
            current_emb = batch["image_embedding"]
            goal_emb = batch["goal_image_embeddings"].mean(dim=1)
        else:
            history_img_emb = self.encode_images(batch["history_images"], batch.get("history_image_paths"))
            current_emb = self.encode_images(batch["image"], batch.get("image_path"))
            goal_emb = self.encode_images(batch["goal_images"], batch.get("goal_image_paths")).mean(dim=1)

        semantic_tokens = []
        semantic_names: list[list[str]] = []
        for q in current_emb:
            result = self.semantic_bank.retrieve(q, top_k=self.top_k_semantic)
            semantic_tokens.append(result.embeddings * result.scores.unsqueeze(-1))
            semantic_names.append(result.categories)

        return {
            "history_img_emb": history_img_emb,
            "current_emb": current_emb,
            "goal_emb": goal_emb,
            "semantic_embeds": torch.stack(semantic_tokens, dim=0),
            "semantic_names": semantic_names,
        }


class Seq2SeqPlanner(BaselineVisionPlannerBase):
    """GRU encoder-decoder baseline over visual history and visual goal."""

    def __init__(
        self,
        dino_encoder: nn.Module,
        adapter: DINOToCLIPAdapter,
        semantic_bank: SemanticBank,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        lora_rank: int,
        horizon: int,
        top_k_semantic: int,
        cache_image_embeddings: bool = False,
        proprio_dim: int = 9,
    ) -> None:
        super().__init__(dino_encoder, adapter, semantic_bank, embed_dim, hidden_dim, horizon, top_k_semantic, cache_image_embeddings, proprio_dim)
        self.history_proj = nn.Linear(embed_dim + proprio_dim, hidden_dim)
        self.context_proj = nn.Linear(embed_dim * 2 + proprio_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=max(int(num_layers), 1), batch_first=True)
        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.step_embed = nn.Embedding(horizon, hidden_dim)
        self.anchor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.waypoint_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        emb = self.common_embeddings(batch)
        hist_inputs = torch.cat([emb["history_img_emb"], batch["history_proprio"]], dim=-1)
        hist_tokens = self.history_proj(hist_inputs)
        _encoded, h = self.encoder(hist_tokens)
        context = self.context_proj(torch.cat([emb["current_emb"], emb["goal_emb"], batch["proprio"]], dim=-1))
        state = torch.tanh(h[-1] + context)
        waypoints = []
        for step in range(self.horizon):
            step_token = self.step_embed.weight[step].unsqueeze(0).expand(state.shape[0], -1)
            state = self.decoder(step_token, state)
            waypoints.append(self.waypoint_head(state))
        waypoints_t = torch.stack(waypoints, dim=1)
        return {
            "anchor": self.anchor_head(state),
            "waypoints": waypoints_t,
            "stop_logit": self.stop_head(state).squeeze(-1),
            "semantic_names": emb["semantic_names"],
        }


class CMAPlanner(BaselineVisionPlannerBase):
    """Cross-modal attention baseline over history, goal, semantics, and proprioception."""

    def __init__(
        self,
        dino_encoder: nn.Module,
        adapter: DINOToCLIPAdapter,
        semantic_bank: SemanticBank,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        lora_rank: int,
        horizon: int,
        top_k_semantic: int,
        cache_image_embeddings: bool = False,
        proprio_dim: int = 9,
    ) -> None:
        super().__init__(dino_encoder, adapter, semantic_bank, embed_dim, hidden_dim, horizon, top_k_semantic, cache_image_embeddings, proprio_dim)
        self.history_proj = nn.Linear(embed_dim + proprio_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.query_proj = nn.Linear(embed_dim * 2 + proprio_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True),
                        "ln1": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)),
                        "ln2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(max(int(num_layers), 1))
            ]
        )
        self.anchor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.refine_head = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * 3))
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        emb = self.common_embeddings(batch)
        hist_tokens = self.history_proj(torch.cat([emb["history_img_emb"], batch["history_proprio"]], dim=-1))
        memory = torch.cat(
            [
                hist_tokens,
                self.image_proj(emb["goal_emb"]).unsqueeze(1),
                self.image_proj(emb["semantic_embeds"]),
                self.proprio_proj(batch["proprio"]).unsqueeze(1),
            ],
            dim=1,
        )
        x = self.query_proj(torch.cat([emb["current_emb"], emb["goal_emb"], batch["proprio"]], dim=-1)).unsqueeze(1)
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, memory, memory)
            x = layer["ln1"](x + attn_out)
            x = layer["ln2"](x + layer["ffn"](x))
        state = x[:, 0]
        anchor = self.anchor_head(state)
        waypoints = self.refine_head(torch.cat([state, anchor], dim=-1)).view(state.shape[0], self.horizon, 3)
        return {
            "anchor": anchor,
            "waypoints": waypoints,
            "stop_logit": self.stop_head(state).squeeze(-1),
            "semantic_names": emb["semantic_names"],
        }


class LAGPlanner(BaselineVisionPlannerBase):
    """Landmark/goal-guided attention baseline for visual-goal navigation."""

    def __init__(
        self,
        dino_encoder: nn.Module,
        adapter: DINOToCLIPAdapter,
        semantic_bank: SemanticBank,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        lora_rank: int,
        horizon: int,
        top_k_semantic: int,
        cache_image_embeddings: bool = False,
        proprio_dim: int = 9,
    ) -> None:
        super().__init__(dino_encoder, adapter, semantic_bank, embed_dim, hidden_dim, horizon, top_k_semantic, cache_image_embeddings, proprio_dim)
        self.history_proj = nn.Linear(embed_dim + proprio_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.goal_query = nn.Linear(embed_dim * 2 + proprio_dim, hidden_dim)
        self.history_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.landmark_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(max(int(num_layers), 1))])
        self.anchor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.refine_head = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * 3))
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        emb = self.common_embeddings(batch)
        hist_tokens = self.history_proj(torch.cat([emb["history_img_emb"], batch["history_proprio"]], dim=-1))
        semantic_tokens = self.image_proj(emb["semantic_embeds"])
        goal_token = self.image_proj(emb["goal_emb"])
        proprio_token = self.proprio_proj(batch["proprio"])
        query = self.goal_query(torch.cat([emb["current_emb"], emb["goal_emb"], batch["proprio"]], dim=-1)).unsqueeze(1)
        hist_ctx, _ = self.history_attn(query, hist_tokens, hist_tokens)
        landmark_ctx, _ = self.landmark_attn(query, semantic_tokens, semantic_tokens)
        hist_ctx = hist_ctx[:, 0]
        landmark_ctx = landmark_ctx[:, 0]
        gate_logits = self.gate(torch.cat([query[:, 0], hist_ctx, landmark_ctx, goal_token], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        state = gate[:, 0:1] * hist_ctx + gate[:, 1:2] * landmark_ctx + gate[:, 2:3] * goal_token + proprio_token
        for layer in self.layers:
            state = state + layer(state)
        anchor = self.anchor_head(state)
        waypoints = self.refine_head(torch.cat([state, anchor], dim=-1)).view(state.shape[0], self.horizon, 3)
        return {
            "anchor": anchor,
            "waypoints": waypoints,
            "stop_logit": self.stop_head(state).squeeze(-1),
            "semantic_names": emb["semantic_names"],
        }
