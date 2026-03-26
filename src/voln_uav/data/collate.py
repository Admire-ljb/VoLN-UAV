from __future__ import annotations

from typing import Any

import torch



def default_collate_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if torch.is_tensor(first):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out
