from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from voln_uav.common.io import ensure_dir, write_json
from voln_uav.data.collate import default_collate_dict
from voln_uav.data.episode_dataset import PlannerDataset
from voln_uav.models.adapter import load_adapter
from voln_uav.models.encoders import build_image_encoder
from voln_uav.models.planner import VoLNPlanner, save_planner
from voln_uav.models.semantic_bank import SemanticBank
from voln_uav.training.losses import planner_loss


class PlannerTrainer:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        model_cfg = config["model"]
        self.embed_dim = int(model_cfg["embed_dim"])
        self.work_dir = ensure_dir(config["work_dir"])
        self.train_dataset = PlannerDataset(
            benchmark_root=config["benchmark_root"],
            records_file=config["train_records"],
            image_size=int(model_cfg.get("image_size", 64)),
            memory_len=int(model_cfg.get("memory_len", 4)),
        )
        self.val_dataset = PlannerDataset(
            benchmark_root=config["benchmark_root"],
            records_file=config["val_records"],
            image_size=int(model_cfg.get("image_size", 64)),
            memory_len=int(model_cfg.get("memory_len", 4)),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=default_collate_dict,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=default_collate_dict,
        )
        dino_encoder = build_image_encoder(model_cfg["dino_backbone"], out_dim=self.embed_dim, image_size=int(model_cfg.get("image_size", 64)))
        adapter = load_adapter(config["adapter_ckpt"], in_dim=self.embed_dim, hidden_dim=int(model_cfg["adapter_hidden"]), out_dim=self.embed_dim)
        semantic_bank = SemanticBank.from_file(config["benchmark_root"] + "/" + config["semantic_bank"], encoder_name=model_cfg["text_encoder"], dim=self.embed_dim)
        self.planner = VoLNPlanner(
            dino_encoder=dino_encoder,
            adapter=adapter,
            semantic_bank=semantic_bank,
            embed_dim=self.embed_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            lora_rank=int(model_cfg["lora_rank"]),
            horizon=int(model_cfg["horizon"]),
            top_k_semantic=int(model_cfg["top_k_semantic"]),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.planner.parameters() if p.requires_grad],
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.planner.train(mode=train)
        running = {"total": 0.0, "waypoint_l1": 0.0, "anchor_l1": 0.0, "stop_bce": 0.0}
        count = 0
        iterator = tqdm(loader, desc="planner-train" if train else "planner-val")
        for batch in iterator:
            batch = self._move_batch(batch)
            out = self.planner(batch)
            loss, loss_items = planner_loss(
                pred_waypoints=out["waypoints"],
                target_waypoints=batch["future_waypoints"],
                pred_anchor=out["anchor"],
                target_anchor=batch["anchor_waypoint"],
                pred_stop_logit=out["stop_logit"],
                target_stop=batch["stop"],
                waypoint_l1_weight=float(self.cfg["loss"]["waypoint_l1_weight"]),
                anchor_weight=float(self.cfg["loss"]["anchor_weight"]),
                stop_weight=float(self.cfg["loss"]["stop_weight"]),
            )
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            for k in running:
                running[k] += loss_items[k]
            count += 1
        return {k: v / max(count, 1) for k, v in running.items()}

    def train(self) -> dict[str, Any]:
        history = []
        best_val = float("inf")
        best_path = self.work_dir / "planner_best.pt"
        last_path = self.work_dir / "planner_last.pt"
        for epoch in range(1, int(self.cfg["epochs"]) + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(entry)
            save_planner(self.planner, last_path, meta={"epoch": epoch, "config": self.cfg, "val_total": val_metrics["total"]})
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                save_planner(self.planner, best_path, meta={"epoch": epoch, "config": self.cfg, "val_total": val_metrics["total"]})
        summary = {"history": history, "best_val": best_val, "best_ckpt": str(best_path), "last_ckpt": str(last_path)}
        write_json(summary, self.work_dir / "metrics.json")
        return summary
