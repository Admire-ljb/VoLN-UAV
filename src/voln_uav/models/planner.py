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
        cache_image_embeddings: bool = False,
        proprio_dim: int = 9,
        planner_backbone: str | None = None,
        lora_enabled: bool = True,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        torch_dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
        language_model: nn.Module | None = None,
        tokenizer: Any | None = None,
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
        self.cache_image_embeddings = bool(cache_image_embeddings)
        self._image_embedding_cache: dict[str, torch.Tensor] = {}
        self.planner_backbone = planner_backbone
        self.uses_language_model = bool(planner_backbone or language_model is not None)

        if self.uses_language_model:
            self.tokenizer, self.language_model = self._build_language_model(
                planner_backbone=planner_backbone,
                lora_enabled=lora_enabled,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                torch_dtype=torch_dtype,
                gradient_checkpointing=gradient_checkpointing,
                language_model=language_model,
                tokenizer=tokenizer,
            )
            lm_hidden = int(self.language_model.config.hidden_size)
            self.hidden_dim = lm_hidden
            self.visual_projector = nn.Sequential(
                nn.Linear(embed_dim, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, lm_hidden),
            )
            self.state_projector = nn.Sequential(
                nn.Linear(proprio_dim, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, lm_hidden),
            )
            self.confidence_projector = nn.Sequential(
                nn.Linear(1, lm_hidden),
                nn.Tanh(),
                nn.Linear(lm_hidden, lm_hidden),
            )
            # GOAL, HISTORY, SEMANTICS, STATE, and PLAN structural markers.
            self.field_tokens = nn.Parameter(torch.empty(5, lm_hidden))
            nn.init.normal_(self.field_tokens, mean=0.0, std=0.02)
            self.trajectory_head = nn.Sequential(
                nn.Linear(lm_hidden, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, horizon * 3),
            )
            self.stop_head = nn.Linear(lm_hidden, 1)
            self._semantic_token_ids = self._tokenize_semantic_bank()
        else:
            # Compact non-language-model fallback retained for focused unit tests.
            # Paper release configs always use lmsys/vicuna-7b-v1.5.
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

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            return aliases[str(name).lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch_dtype={name!r}") from exc

    @classmethod
    def _build_language_model(
        cls,
        planner_backbone: str | None,
        lora_enabled: bool,
        lora_rank: int,
        lora_alpha: int | None,
        lora_dropout: float,
        lora_target_modules: list[str] | None,
        torch_dtype: str,
        gradient_checkpointing: bool,
        language_model: nn.Module | None,
        tokenizer: Any | None,
    ) -> tuple[Any, nn.Module]:
        if language_model is not None:
            if tokenizer is None:
                raise ValueError("An injected language_model requires a tokenizer")
            return tokenizer, language_model
        if not planner_backbone:
            raise ValueError("planner_backbone is required for the language-model planner")
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in real-model runs
            raise ImportError("Install the real-model dependencies with: pip install -e .[real]") from exc

        tokenizer = AutoTokenizer.from_pretrained(planner_backbone, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            planner_backbone,
            torch_dtype=cls._torch_dtype(torch_dtype),
            low_cpu_mem_usage=True,
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.config.use_cache = False
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if not lora_enabled:
            return tokenizer, model
        targets = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora_rank),
            lora_alpha=int(lora_alpha or (2 * lora_rank)),
            lora_dropout=float(lora_dropout),
            target_modules=targets,
            bias="none",
        )
        return tokenizer, get_peft_model(model, peft_config)

    def _tokenize_semantic_bank(self) -> dict[str, list[int]]:
        token_ids: dict[str, list[int]] = {}
        fallback = getattr(self.tokenizer, "unk_token_id", None)
        if fallback is None:
            fallback = getattr(self.tokenizer, "eos_token_id", 0)
        for category in self.semantic_bank.categories:
            encoded = self.tokenizer(category, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            token_ids[category] = [int(item) for item in ids] or [int(fallback)]
        return token_ids

    def train(self, mode: bool = True) -> "VoLNPlanner":
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
            dino = self.dino_encoder(images)
            return self.adapter(dino)

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

    def _retrieve_semantic_tokens(self, query: torch.Tensor) -> tuple[torch.Tensor, list[list[str]]]:
        batch_embeds = []
        batch_names: list[list[str]] = []
        for q in query:
            result = self.semantic_bank.retrieve(q, top_k=self.top_k_semantic)
            scores = result.scores.unsqueeze(-1)
            batch_embeds.append(result.embeddings * scores)
            batch_names.append(result.categories)
        return torch.stack(batch_embeds, dim=0), batch_names

    def _retrieve_semantic_entries(self, query: torch.Tensor) -> tuple[list[list[str]], torch.Tensor]:
        names: list[list[str]] = []
        scores: list[torch.Tensor] = []
        for item in query:
            result = self.semantic_bank.retrieve(item, top_k=self.top_k_semantic)
            names.append(result.categories)
            scores.append(result.scores)
        return names, torch.stack(scores, dim=0)

    def _category_embeddings(self, semantic_names: list[list[str]], device: torch.device) -> torch.Tensor:
        embedding_layer = self.language_model.get_input_embeddings()
        rows = []
        with torch.no_grad():
            for names in semantic_names:
                row = []
                for name in names:
                    ids = torch.tensor(self._semantic_token_ids[name], dtype=torch.long, device=device)
                    row.append(embedding_layer(ids).mean(dim=0).detach())
                rows.append(torch.stack(row, dim=0))
        return torch.stack(rows, dim=0).float()

    def _build_language_model_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        if "image_embedding" in batch:
            history_img_emb = batch["history_image_embeddings"]
            current_emb = batch["image_embedding"]
            goal_embeds = batch["goal_image_embeddings"]
        else:
            history_img_emb = self.encode_images(batch["history_images"], batch.get("history_image_paths"))
            current_emb = self.encode_images(batch["image"], batch.get("image_path"))
            goal_embeds = self.encode_images(batch["goal_images"], batch.get("goal_image_paths"))

        semantic_names, semantic_scores = self._retrieve_semantic_entries(current_emb)
        history_tokens = self.visual_projector(history_img_emb)
        goal_tokens = self.visual_projector(goal_embeds)
        category_tokens = self._category_embeddings(semantic_names, device=current_emb.device)
        semantic_tokens = category_tokens + self.confidence_projector(semantic_scores.unsqueeze(-1).float())
        state_token = self.state_projector(batch["proprio"].float()).unsqueeze(1)
        batch_size = current_emb.shape[0]

        def marker(index: int) -> torch.Tensor:
            return self.field_tokens[index].view(1, 1, -1).expand(batch_size, -1, -1)

        # PLAN is last because Vicuna is causal; its hidden state can attend to all
        # permitted task inputs while no future token can leak into the prediction.
        sequence = torch.cat(
            [
                marker(0),
                goal_tokens,
                marker(1),
                history_tokens,
                marker(2),
                semantic_tokens,
                marker(3),
                state_token,
                marker(4),
            ],
            dim=1,
        )
        return sequence, {"semantic_names": semantic_names}

    def build_token_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        if "image_embedding" in batch:
            history_img_emb = batch["history_image_embeddings"]
            current_emb = batch["image_embedding"]
            goal_emb = batch["goal_image_embeddings"].mean(dim=1)
        else:
            history_img_emb = self.encode_images(batch["history_images"], batch.get("history_image_paths"))
            current_emb = self.encode_images(batch["image"], batch.get("image_path"))
            goal_emb = self.encode_images(batch["goal_images"], batch.get("goal_image_paths")).mean(dim=1)
        semantic_embeds, semantic_names = self._retrieve_semantic_tokens(current_emb)

        hist_inputs = torch.cat([history_img_emb, batch["history_proprio"]], dim=-1)
        hist_tokens = self.history_proj(hist_inputs)
        goal_token = self.image_proj(goal_emb).unsqueeze(1)
        semantic_tokens = self.image_proj(semantic_embeds)
        proprio_token = self.proprio_proj(batch["proprio"]).unsqueeze(1)
        plan_token = self.plan_token.expand(current_emb.shape[0], -1, -1)
        seq = torch.cat([plan_token, hist_tokens, goal_token, semantic_tokens, proprio_token], dim=1)
        aux = {
            "semantic_names": semantic_names,
            "current_embedding": current_emb,
            "goal_embedding": goal_emb,
        }
        return seq, aux

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[list[str]]]:
        if self.uses_language_model:
            sequence, aux = self._build_language_model_sequence(batch)
            embedding_weight = self.language_model.get_input_embeddings().weight
            inputs = sequence.to(dtype=embedding_weight.dtype)
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
            outputs = self.language_model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            plan_state = outputs.hidden_states[-1][:, -1].float()
            waypoints = self.trajectory_head(plan_state).view(plan_state.shape[0], self.horizon, 3)
            stop = self.stop_head(plan_state).squeeze(-1)
            return {
                "anchor": waypoints[:, -1],
                "waypoints": waypoints,
                "stop_logit": stop,
                "semantic_names": aux["semantic_names"],
            }

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



def trainable_state_dict(planner: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }


def load_planner_state(planner: nn.Module, checkpoint: dict[str, Any]) -> None:
    state_dict = checkpoint["state_dict"]
    checkpoint_format = checkpoint.get("meta", {}).get("checkpoint_format")
    if checkpoint_format == "trainable_state_dict_v1":
        expected = {name for name, parameter in planner.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(expected.difference(state_dict))
        if missing_trainable:
            raise RuntimeError(f"Planner checkpoint is missing trainable keys: {missing_trainable}")
        incompatible = planner.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected planner checkpoint keys: {incompatible.unexpected_keys}")
        return
    planner.load_state_dict(state_dict, strict=True)


def save_planner(planner: nn.Module, path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_meta = dict(meta)
    payload_meta["checkpoint_format"] = "trainable_state_dict_v1"
    torch.save({"state_dict": trainable_state_dict(planner), "meta": payload_meta}, path)
