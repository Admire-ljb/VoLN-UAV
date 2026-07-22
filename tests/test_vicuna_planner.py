from types import SimpleNamespace

import torch
from torch import nn

from voln_uav.models.adapter import DINOToCLIPAdapter
from voln_uav.models.planner import VoLNPlanner, load_planner_state, save_planner
from voln_uav.models.semantic_bank import SemanticBank


class DummyTokenizer:
    unk_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [1 + (ord(char) % 31) for char in text[:4]]}


class DummyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=False)
        self.embedding = nn.Embedding(64, hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        # Cumulative context makes the last (PLAN) token depend on all prior fields.
        hidden = torch.cumsum(inputs_embeds, dim=1)
        return SimpleNamespace(hidden_states=(hidden,))


def _planner() -> VoLNPlanner:
    encoder = nn.Identity()
    adapter = DINOToCLIPAdapter(in_dim=8, hidden_dim=0, out_dim=8)
    bank = SemanticBank(
        categories=["turn left", "turn right", "goal"],
        embeddings=torch.randn(3, 8),
    )
    return VoLNPlanner(
        dino_encoder=encoder,
        adapter=adapter,
        semantic_bank=bank,
        embed_dim=8,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        lora_rank=2,
        horizon=2,
        top_k_semantic=2,
        planner_backbone="dummy-vicuna",
        language_model=DummyCausalLM(),
        tokenizer=DummyTokenizer(),
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "history_image_embeddings": torch.randn(1, 3, 8),
        "image_embedding": torch.randn(1, 8),
        "goal_image_embeddings": torch.randn(1, 2, 8),
        "proprio": torch.randn(1, 9),
    }


def test_vicuna_planner_uses_all_structured_fields_and_plan_token_last():
    planner = _planner()
    sequence, aux = planner._build_language_model_sequence(_batch())

    # 5 field markers + 2 goal + 3 history + 2 semantic + 1 state.
    assert sequence.shape == (1, 13, 16)
    assert len(aux["semantic_names"][0]) == 2
    assert torch.allclose(sequence[:, -1], planner.field_tokens[4].unsqueeze(0))

    output = planner(_batch())
    assert output["waypoints"].shape == (1, 2, 3)
    assert output["stop_logit"].shape == (1,)


def test_planner_checkpoint_contains_only_trainable_parameters(tmp_path):
    planner = _planner()
    checkpoint_path = tmp_path / "planner.pt"
    save_planner(planner, checkpoint_path, meta={"epoch": 1})
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["meta"]["checkpoint_format"] == "trainable_state_dict_v1"
    assert not any(name.startswith("dino_encoder") for name in checkpoint["state_dict"])
    assert not any(name.startswith("language_model.embedding") for name in checkpoint["state_dict"])

    restored = _planner()
    load_planner_state(restored, checkpoint)
