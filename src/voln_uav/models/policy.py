from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from voln_uav.common.image import load_image_tensor
from voln_uav.models.planner import load_planner_state
from voln_uav.models.planner_factory import build_planner
from voln_uav.models.semantic_bank import SemanticBank
from voln_uav.models.vision import build_planner_vision


class VoLNPolicy:
    def __init__(self, config: dict[str, Any], semantic_bank_path: str | Path, adapter_ckpt: str | Path | None, planner_ckpt: str | Path, device: str = "cpu") -> None:
        self.cfg = config
        model_cfg = config["model"]
        self.device = torch.device(device)
        self.benchmark_root = Path(config["benchmark_root"]).resolve() if "benchmark_root" in config else Path.cwd()
        self.repo_root = self.benchmark_root.parent
        self._image_cache: dict[str, torch.Tensor] = {}
        embed_dim = int(model_cfg["embed_dim"])
        image_size = int(model_cfg.get("image_size", 64))
        self.memory_len = int(model_cfg["memory_len"])
        self.image_size = image_size

        ckpt = torch.load(planner_ckpt, map_location=self.device)
        self.stop_threshold = float(ckpt.get("meta", {}).get("stop_threshold", 0.5))
        ckpt_model_cfg = ckpt.get("meta", {}).get("config", {}).get("model", {})
        if "planner_variant" not in model_cfg and "planner_variant" in ckpt_model_cfg:
            model_cfg = dict(model_cfg)
            model_cfg["planner_variant"] = ckpt_model_cfg["planner_variant"]
        defaults: dict[str, Any] = {
            "vision_input": "dino_aligned",
            "lora_enabled": True,
            "planner_variant": "voln",
            "ablation": None,
            "image_size": 224,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "lora_alpha": None,
            "lora_dropout": 0.05,
            "lora_target_modules": None,
        }
        signature_keys = (
            "planner_variant",
            "ablation",
            "vision_input",
            "dino_backbone",
            "dino_dim",
            "clip_image_encoder",
            "text_encoder",
            "embed_dim",
            "adapter_hidden",
            "image_size",
            "planner_backbone",
            "hidden_dim",
            "num_heads",
            "num_layers",
            "lora_enabled",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "horizon",
            "top_k_semantic",
            "memory_len",
        )
        mismatches = []
        for key in signature_keys:
            current = model_cfg.get(key, defaults.get(key))
            trained = ckpt_model_cfg.get(key, defaults.get(key))
            if current != trained:
                mismatches.append(f"{key}: checkpoint={trained!r}, evaluation={current!r}")
        if mismatches:
            raise ValueError("Planner checkpoint/config mismatch:\n- " + "\n- ".join(mismatches))

        dino_encoder, adapter, _vision_dim = build_planner_vision(
            model_cfg,
            adapter_ckpt=adapter_ckpt,
            map_location=str(self.device),
        )
        semantic_bank = SemanticBank.from_file(semantic_bank_path, encoder_name=model_cfg["text_encoder"], dim=embed_dim)
        self.planner = build_planner(
            model_cfg=model_cfg,
            dino_encoder=dino_encoder,
            adapter=adapter,
            semantic_bank=semantic_bank,
        )
        load_planner_state(self.planner, ckpt)
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
        proprio = torch.tensor(list(state.get("imu", [])) + list(state.get("odometry", [])), dtype=torch.float32).unsqueeze(0)
        return {
            "history_images": history_images.to(self.device),
            "history_proprio": history_proprio.to(self.device),
            "image": cur_image.to(self.device),
            "goal_images": goal_images.to(self.device),
            "proprio": proprio.to(self.device),
        }

    @torch.no_grad()
    def act(self, state: dict[str, Any], history_states: list[dict[str, Any]], visual_goal: dict[str, Any]) -> dict[str, Any]:
        batch = self.prepare_batch(state, history_states, visual_goal)
        out = self.planner(batch)
        waypoints = out["waypoints"][0].cpu()
        anchor = out["anchor"][0].cpu()
        stop_prob = torch.sigmoid(out["stop_logit"])[0].item()
        if waypoints.shape != (int(self.cfg["model"]["horizon"]), 3):
            raise ValueError(f"Invalid waypoint output shape: {tuple(waypoints.shape)}")
        if not torch.isfinite(waypoints).all() or not torch.isfinite(anchor).all():
            raise ValueError("Planner produced non-finite waypoint output")
        if not math.isfinite(stop_prob):
            raise ValueError("Planner produced a non-finite stop probability")
        return {
            "waypoints": waypoints,
            "anchor": anchor,
            "stop_prob": stop_prob,
            "semantic_names": out["semantic_names"][0],
        }
