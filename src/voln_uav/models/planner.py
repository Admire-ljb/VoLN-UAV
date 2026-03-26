from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voln_uav.models.adapter import DINOToCLIPAdapter
from voln_uav.models.lora import LoRALinear
from voln_uav.models.semantic_bank import SemanticBank


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, lora_rank: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.q_proj = LoRALinear(dim, dim, rank=lora_rank)
        self.k_proj = LoRALinear(dim, dim, rank=lora_rank)
        self.v_proj = LoRALinear(dim, dim, rank=lora_rank)
        self.o_proj = LoRALinear(dim, dim, rank=lora_rank)
        self.ff1 = LoRALinear(dim, dim * 4, rank=lora_rank)
        self.ff2 = LoRALinear(dim * 4, dim, rank=lora_rank)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, hd = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, h * hd)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._merge_heads(out)
        x = x + self.o_proj(out)
        h2 = self.ln2(x)
        x = x + self.ff2(torch.nn.functional.gelu(self.ff1(h2)))
        return x


class VoLNPlanner(nn.Module):
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
        proprio_dim: int = 9,
    ) -> None:
        super().__init__()
        self.dino_encoder = dino_encoder
        self.adapter = adapter
        self.semantic_bank = semantic_bank
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.top_k_semantic = top_k_semantic
        self.proprio_dim = proprio_dim

        self.history_proj = nn.Linear(embed_dim + proprio_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.plan_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList([SelfAttentionBlock(hidden_dim, num_heads=num_heads, lora_rank=lora_rank) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.anchor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.refine_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * 3),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)

        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = False
        self.dino_encoder.eval()
        self.adapter.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            b, n, c, h, w = images.shape
            flat = images.view(b * n, c, h, w)
            with torch.no_grad():
                dino = self.dino_encoder(flat)
                aligned = self.adapter(dino)
            return aligned.view(b, n, -1)
        if images.ndim == 4:
            with torch.no_grad():
                dino = self.dino_encoder(images)
                aligned = self.adapter(dino)
            return aligned
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    def _retrieve_semantic_tokens(self, query: torch.Tensor) -> tuple[torch.Tensor, list[list[str]]]:
        batch_embeds = []
        batch_names: list[list[str]] = []
        for q in query:
            result = self.semantic_bank.retrieve(q, top_k=self.top_k_semantic)
            scores = result.scores.unsqueeze(-1)
            batch_embeds.append(result.embeddings * scores)
            batch_names.append(result.categories)
        return torch.stack(batch_embeds, dim=0), batch_names

    def build_token_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        history_img_emb = self.encode_images(batch["history_images"])
        current_emb = self.encode_images(batch["image"])
        goal_emb = self.encode_images(batch["goal_images"]).mean(dim=1)
        subgoal_emb = self.encode_images(batch["subgoal_images"]).mean(dim=1)
        beacon_emb = self.encode_images(batch["beacon_images"]).mean(dim=1)
        semantic_embeds, semantic_names = self._retrieve_semantic_tokens(current_emb)

        hist_inputs = torch.cat([history_img_emb, batch["history_proprio"]], dim=-1)
        hist_tokens = self.history_proj(hist_inputs)
        goal_token = self.image_proj(goal_emb).unsqueeze(1)
        subgoal_token = self.image_proj(subgoal_emb).unsqueeze(1)
        beacon_token = self.image_proj(beacon_emb).unsqueeze(1)
        semantic_tokens = self.image_proj(semantic_embeds)
        proprio_token = self.proprio_proj(batch["proprio"]).unsqueeze(1)
        plan_token = self.plan_token.expand(batch["image"].shape[0], -1, -1)

        seq = torch.cat([plan_token, hist_tokens, goal_token, subgoal_token, beacon_token, semantic_tokens, proprio_token], dim=1)
        aux = {
            "semantic_names": semantic_names,
            "current_embedding": current_emb,
            "goal_embedding": goal_emb,
        }
        return seq, aux

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[list[str]]]:
        seq, aux = self.build_token_sequence(batch)
        x = seq
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        plan_state = x[:, 0]
        anchor = self.anchor_head(plan_state)
        refined = self.refine_head(torch.cat([plan_state, anchor], dim=-1)).view(plan_state.shape[0], self.horizon, 3)
        stop = self.stop_head(plan_state).squeeze(-1)
        return {
            "anchor": anchor,
            "waypoints": refined,
            "stop_logit": stop,
            "semantic_names": aux["semantic_names"],
        }



def save_planner(planner: VoLNPlanner, path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": planner.state_dict(), "meta": meta}, path)
