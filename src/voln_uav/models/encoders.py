from __future__ import annotations

from typing import Iterable

import hashlib
import numpy as np
import torch
from torch import nn


def parse_open_clip_spec(spec: str) -> tuple[str, str]:
    # format: open_clip:<model_name>[:<pretrained_tag>]
    parts = spec.split(":")
    if len(parts) < 2 or not parts[1]:
        return "ViT-B-32", "laion2b_s34b_b79k"
    model_name = parts[1]
    pretrained = parts[2] if len(parts) >= 3 and parts[2] else "laion2b_s34b_b79k"
    return model_name, pretrained


class FrozenModule(nn.Module):
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


class ToyImageEncoder(FrozenModule):
    def __init__(self, out_dim: int, variant: str = "toy", image_size: int = 64) -> None:
        super().__init__()
        seed = {
            "toy_dino": 13,
            "toy_clip": 29,
            "toy_aux": 41,
        }.get(variant, 7)
        g = torch.Generator().manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)
        self._reset(g)
        self.freeze()

    def _reset(self, g: torch.Generator) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02, generator=g)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        out = self.proj(feat)
        return torch.nn.functional.normalize(out, dim=-1)


class TimmImageEncoder(FrozenModule):
    def __init__(self, model_name: str, out_dim: int) -> None:
        super().__init__()
        try:
            import timm
        except Exception as e:  # pragma: no cover
            raise ImportError("Install timm to use real backbones.") from e
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", out_dim)
        self.proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.proj(feat)
        return torch.nn.functional.normalize(out, dim=-1)


class OpenCLIPImageEncoder(FrozenModule):
    def __init__(self, model_name: str, pretrained: str, out_dim: int) -> None:
        super().__init__()
        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise ImportError("Install open-clip-torch to use OpenCLIP image encoders.") from e
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # OpenCLIP image head width differs across variants; run a tiny dry forward to infer.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.model.encode_image(dummy)
            feat_dim = int(feat.shape[-1])
        self.proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.model.encode_image(x)
        out = self.proj(feat)
        return torch.nn.functional.normalize(out, dim=-1)


class HFVisionEncoder(FrozenModule):
    def __init__(self, model_name: str, out_dim: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise ImportError("Install transformers to use HuggingFace vision encoders.") from e
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The dataset already outputs normalized float tensors in CHW format.
        out = self.backbone(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        feat = self.proj(feat)
        return torch.nn.functional.normalize(feat, dim=-1)



def build_image_encoder(name: str, out_dim: int, image_size: int = 64) -> nn.Module:
    if name in {"toy_dino", "toy_clip", "toy_aux"}:
        return ToyImageEncoder(out_dim=out_dim, variant=name, image_size=image_size)
    if name.startswith("open_clip:"):
        model_name, pretrained = parse_open_clip_spec(name)
        return OpenCLIPImageEncoder(model_name=model_name, pretrained=pretrained, out_dim=out_dim)
    if name.startswith("hf:"):
        # format: hf:<model_name>
        model_name = name.split(":", 1)[1]
        return HFVisionEncoder(model_name=model_name, out_dim=out_dim)
    return TimmImageEncoder(model_name=name, out_dim=out_dim)



def encode_texts_toy(texts: Iterable[str], dim: int) -> torch.Tensor:
    vectors = []
    for text in texts:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = torch.tensor(rng.standard_normal(dim), dtype=torch.float32)
        vec = vec / (vec.norm() + 1e-6)
        vectors.append(vec)
    return torch.stack(vectors, dim=0)



def encode_texts(texts: Iterable[str], encoder_name: str, dim: int) -> torch.Tensor:
    texts = list(texts)
    if encoder_name == "toy_text":
        return encode_texts_toy(texts, dim=dim)
    if encoder_name.startswith("open_clip"):
        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise ImportError("Install open-clip-torch to use open_clip text encoding.") from e
        model_name, pretrained = parse_open_clip_spec(encoder_name)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        with torch.no_grad():
            tokens = tokenizer(texts)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float()
    raise ValueError(f"Unsupported text encoder: {encoder_name}")
