from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from voln_uav.models.encoders import encode_texts


@dataclass
class RetrievalResult:
    categories: list[str]
    scores: torch.Tensor
    embeddings: torch.Tensor


class SemanticBank:
    def __init__(self, categories: list[str], embeddings: torch.Tensor) -> None:
        self.categories = categories
        self.embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)

    @classmethod
    def from_file(cls, path: str | Path, encoder_name: str, dim: int) -> "SemanticBank":
        with Path(path).open("r", encoding="utf-8") as f:
            categories = [line.strip() for line in f if line.strip()]
        embeddings = encode_texts(categories, encoder_name=encoder_name, dim=dim)
        return cls(categories=categories, embeddings=embeddings)

    def retrieve(self, query: torch.Tensor, top_k: int) -> RetrievalResult:
        query = torch.nn.functional.normalize(query.float(), dim=-1)
        embeddings = self.embeddings.to(device=query.device, dtype=query.dtype)
        sims = query @ embeddings.T
        k = min(top_k, self.embeddings.shape[0])
        scores, idx = torch.topk(sims, k=k, dim=-1)
        if query.ndim == 1:
            categories = [self.categories[i] for i in idx.tolist()]
            return RetrievalResult(categories=categories, scores=scores, embeddings=embeddings[idx])
        # batched path
        # flattening category names is only used for logging; not needed in training.
        categories = [self.categories[i] for i in idx[0].tolist()]
        return RetrievalResult(categories=categories, scores=scores, embeddings=embeddings[idx])

    def signature(self, encoder_name: str) -> dict[str, Any]:
        payload = "\n".join(self.categories).encode("utf-8")
        return {
            "categories": list(self.categories),
            "encoder": str(encoder_name),
            "dim": int(self.embeddings.shape[-1]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
