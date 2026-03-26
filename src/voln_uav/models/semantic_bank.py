from __future__ import annotations

from dataclasses import dataclass
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
        sims = query @ self.embeddings.T
        k = min(top_k, self.embeddings.shape[0])
        scores, idx = torch.topk(sims, k=k, dim=-1)
        if query.ndim == 1:
            categories = [self.categories[i] for i in idx.tolist()]
            embeddings = self.embeddings[idx]
            return RetrievalResult(categories=categories, scores=scores, embeddings=embeddings)
        # batched path
        # flattening category names is only used for logging; not needed in training.
        categories = [self.categories[i] for i in idx[0].tolist()]
        embeddings = self.embeddings[idx]
        return RetrievalResult(categories=categories, scores=scores, embeddings=embeddings)
