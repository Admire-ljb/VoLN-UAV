from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from voln_uav.common.image import load_image_tensor
from voln_uav.models.adapter import load_adapter
from voln_uav.models.encoders import build_image_encoder
from voln_uav.models.planner import VoLNPlanner
from voln_uav.models.semantic_bank import SemanticBank


class VoLNPolicy:
    def __init__(self, config: dict[str, Any], semantic_bank_path: str | Path, adapter_ckpt: str | Path, planner_ckpt: str | Path, device: str = "cpu") -> None:
        self.cfg = config
        model_cfg = config["model"]
        self.device = torch.device(device)
        self.benchmark_root = Path(config["benchmark_root"]).resolve() if "benchmark_root" in config else Path.cwd()
        self.repo_root = self.benchmark_root.parent
        self._image_cache: dict[str, torch.Tensor] = {}
        embed_dim = int(model_cfg["embed_dim"])
        image_size = int(model_cfg.get("image_size", 64))
        hidden_dim = int(model_cfg["hidden_dim"])
        num_heads = int(model_cfg["num_heads"])
        num_layers = int(model_cfg["num_layers"])
        lora_rank = int(model_cfg["lora_rank"])
        horizon = int(model_cfg["horizon"])
        top_k_semantic = int(model_cfg["top_k_semantic"])
        self.memory_len = int(model_cfg["memory_len"])
        self.image_size = image_size

        dino_encoder = build_image_encoder(model_cfg["dino_backbone"], out_dim=embed_dim, image_size=image_size)
        adapter = load_adapter(adapter_ckpt, in_dim=embed_dim, hidden_dim=int(model_cfg["adapter_hidden"]), out_dim=embed_dim)
        semantic_bank = SemanticBank.from_file(semantic_bank_path, encoder_name=model_cfg["text_encoder"], dim=embed_dim)
        self.planner = VoLNPlanner(
            dino_encoder=dino_encoder,
            adapter=adapter,
            semantic_bank=semantic_bank,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            lora_rank=lora_rank,
            horizon=horizon,
            top_k_semantic=top_k_semantic,
        )
        ckpt = torch.load(planner_ckpt, map_location=self.device)
        self.planner.load_state_dict(ckpt["state_dict"], strict=True)
        self.planner.to(self.device)
        self.planner.eval()

    def _resolve(self, path_like: str) -> str:
        path = Path(path_like)
        if path.exists():
            return str(path)
        candidate = self.repo_root / path_like
        if candidate.exists():
            return str(candidate)
        candidate2 = self.benchmark_root / path_like
        if candidate2.exists():
            return str(candidate2)
        raise FileNotFoundError(f"Could not resolve path: {path_like}")

    def _load_cached_image(self, path_like: str) -> torch.Tensor:
        resolved = self._resolve(path_like)
        key = f"{resolved}|{self.image_size}"
        item = self._image_cache.get(key)
        if item is None:
            item = load_image_tensor(resolved, image_size=self.image_size)
            self._image_cache[key] = item
        return item

    def _stack_cached_images(self, paths: list[str]) -> torch.Tensor:
        if not paths:
            return torch.zeros(1, 3, self.image_size, self.image_size)
        return torch.stack([self._load_cached_image(p) for p in paths], dim=0)

    def prepare_batch(self, state: dict[str, Any], history_states: list[dict[str, Any]], visual_goal: dict[str, Any]) -> dict[str, torch.Tensor]:
        history_images = self._stack_cached_images([s["image"] for s in history_states]).unsqueeze(0)
        history_proprio = torch.tensor([list(s.get("imu", [])) + list(s.get("odometry", [])) for s in history_states], dtype=torch.float32).unsqueeze(0)
        cur_image = self._load_cached_image(state["image"]).unsqueeze(0)
        goal_images = self._stack_cached_images(list(visual_goal["V_goal"])).unsqueeze(0)
        subgoal_images = self._stack_cached_images(list(visual_goal["V_sub"])).unsqueeze(0)
        beacon_images = self._stack_cached_images(list(visual_goal["V_beacon"])).unsqueeze(0)
        proprio = torch.tensor(list(state.get("imu", [])) + list(state.get("odometry", [])), dtype=torch.float32).unsqueeze(0)
        return {
            "history_images": history_images.to(self.device),
            "history_proprio": history_proprio.to(self.device),
            "image": cur_image.to(self.device),
            "goal_images": goal_images.to(self.device),
            "subgoal_images": subgoal_images.to(self.device),
            "beacon_images": beacon_images.to(self.device),
            "proprio": proprio.to(self.device),
        }

    @torch.no_grad()
    def act(self, state: dict[str, Any], history_states: list[dict[str, Any]], visual_goal: dict[str, Any]) -> dict[str, Any]:
        batch = self.prepare_batch(state, history_states, visual_goal)
        out = self.planner(batch)
        waypoints = out["waypoints"][0].cpu()
        anchor = out["anchor"][0].cpu()
        stop_prob = torch.sigmoid(out["stop_logit"])[0].item()
        return {
            "waypoints": waypoints,
            "anchor": anchor,
            "stop_prob": stop_prob,
            "semantic_names": out["semantic_names"][0],
        }
