from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 0, bias: bool = True, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else max(rank, 1))
        if self.rank > 0:
            self.lora_a = nn.Linear(in_features, self.rank, bias=False)
            self.lora_b = nn.Linear(self.rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.rank > 0 and self.lora_a is not None and self.lora_b is not None:
            out = out + self.lora_b(self.lora_a(x)) * (self.alpha / self.rank)
        return out
