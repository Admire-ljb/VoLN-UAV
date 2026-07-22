from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from voln_uav.common.io import ensure_dir, write_json
from voln_uav.data.collate import default_collate_dict
from voln_uav.data.episode_dataset import AdapterDistillDataset
from voln_uav.models.adapter import DINOToCLIPAdapter, cosine_distill_loss, save_adapter
from voln_uav.models.encoders import build_image_encoder


class AdapterTrainer:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        model_cfg = config["model"]
        self.alignment_mode = str(config.get("alignment_mode", "clip_distill")).lower()
        if self.alignment_mode not in {"clip_distill", "no_align"}:
            raise ValueError(f"Unsupported alignment_mode={self.alignment_mode!r}")
        self.embed_dim = int(model_cfg["embed_dim"])
        self.dino_dim = int(model_cfg.get("dino_dim", self.embed_dim))
        self.work_dir = ensure_dir(config["work_dir"])
        self.dataset = AdapterDistillDataset(
            benchmark_root=config["benchmark_root"],
            records_file=config["records_file"],
            image_size=int(model_cfg.get("image_size", 64)),
        )
        max_records = config.get("max_records")
        if max_records is not None:
            self.dataset = Subset(self.dataset, range(min(int(max_records), len(self.dataset))))
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=default_collate_dict,
        )
        self.dino = build_image_encoder(model_cfg["dino_backbone"], out_dim=self.dino_dim, image_size=int(model_cfg.get("image_size", 64))).to(self.device)
        self.teacher = None
        if self.alignment_mode == "clip_distill":
            self.teacher = build_image_encoder(model_cfg["clip_image_encoder"], out_dim=self.embed_dim, image_size=int(model_cfg.get("image_size", 64))).to(self.device)
        self.dino.eval()
        if self.teacher is not None:
            self.teacher.eval()
        for module in (self.dino, self.teacher):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False
        self.adapter = DINOToCLIPAdapter(
            in_dim=self.dino_dim,
            hidden_dim=int(model_cfg.get("adapter_hidden", 0)),
            out_dim=self.embed_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    def train(self) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_path = self.work_dir / "adapter_best.pt"
        last_path = self.work_dir / "adapter_last.pt"
        if self.alignment_mode == "no_align":
            meta = {"epoch": 0, "loss": None, "alignment_mode": self.alignment_mode, "config": self.cfg}
            save_adapter(self.adapter, last_path, meta=meta)
            save_adapter(self.adapter, best_path, meta=meta)
            summary = {
                "alignment_mode": self.alignment_mode,
                "best_loss": None,
                "history": history,
                "best_ckpt": str(best_path),
                "last_ckpt": str(last_path),
            }
            write_json(summary, self.work_dir / "metrics.json")
            return summary
        for epoch in range(1, int(self.cfg["epochs"]) + 1):
            self.adapter.train()
            running = 0.0
            count = 0
            for batch in tqdm(self.loader, desc=f"adapter-epoch-{epoch}"):
                images = batch["image"].to(self.device)
                with torch.no_grad():
                    dino_emb = self.dino(images)
                    assert self.teacher is not None
                    teacher_emb = self.teacher(images)
                student_emb = self.adapter(dino_emb)
                loss = cosine_distill_loss(student_emb, teacher_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += float(loss.detach().cpu())
                count += 1
            epoch_loss = running / max(count, 1)
            history.append({"epoch": epoch, "loss": epoch_loss})
            checkpoint_meta = {
                "epoch": epoch,
                "loss": epoch_loss,
                "alignment_mode": self.alignment_mode,
                "config": self.cfg,
            }
            save_adapter(self.adapter, last_path, meta=checkpoint_meta)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_adapter(self.adapter, best_path, meta=checkpoint_meta)
        summary = {"best_loss": best_loss, "history": history, "best_ckpt": str(best_path), "last_ckpt": str(last_path)}
        write_json(summary, self.work_dir / "metrics.json")
        return summary
