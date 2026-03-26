import torch

from voln_uav.models.lora import LoRALinear


def test_lora_shapes():
    layer = LoRALinear(8, 4, rank=2)
    x = torch.randn(3, 8)
    y = layer(x)
    assert y.shape == (3, 4)
