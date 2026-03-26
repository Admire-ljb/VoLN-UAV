from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image



def _pil_to_tensor(img: Image.Image, image_size: int = 64) -> torch.Tensor:
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)



def load_image_tensor(path: str | Path, image_size: int = 64) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _pil_to_tensor(img, image_size=image_size)



def stack_images(paths: Iterable[str | Path], image_size: int = 64) -> torch.Tensor:
    items = [load_image_tensor(p, image_size=image_size) for p in paths]
    if not items:
        return torch.zeros(1, 3, image_size, image_size)
    return torch.stack(items, dim=0)
