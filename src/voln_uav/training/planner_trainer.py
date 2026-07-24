from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from voln_uav.common.geometry import l2
from voln_uav.common.image import load_image_tensor
from voln_uav.common.io import ensure_dir, read_json, write_json
from voln_uav.common.navigation_frames import PROPRIO_SCHEMA
from voln_uav.data.collate import default_collate_dict
from voln_uav.data.episode_dataset import PlannerDataset
from voln_uav.models.planner import load_planner_state, save_planner
from voln_uav.models.planner_factory import build_planner, normalize_planner_variant
from voln_uav.models.semantic_bank import SemanticBank
from voln_uav.models.vision import build_planner_vision
from voln_uav.training.losses import planner_loss


class PlannerTrainer:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        model_cfg = config["model"]
        self.embed_dim = int(model_cfg["embed_dim"])
        self.dino_dim = int(model_cfg.get("dino_dim", self.embed_dim))
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
        max_train_records = config.get("max_train_records", config.get("max_records"))
        if max_train_records is not None:
            self.train_dataset = Subset(self.train_dataset, range(min(int(max_train_records), len(self.train_dataset))))
        max_val_records = config.get("max_val_records", config.get("max_records"))
        if max_val_records is not None:
            self.val_dataset = Subset(self.val_dataset, range(min(int(max_val_records), len(self.val_dataset))))
        self.stop_targets = self._build_stop_targets(
            (self.train_dataset, self.val_dataset),
            success_radius=float(config["success_radius"]),
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
        dino_encoder, adapter, self.dino_dim = build_planner_vision(
            model_cfg,
            adapter_ckpt=config.get("adapter_ckpt"),
            map_location=str(self.device),
        )
        self._precompute_image_embeddings(
            dino_encoder=dino_encoder,
            adapter=adapter,
            image_size=int(model_cfg.get("image_size", 64)),
        )
        semantic_bank = SemanticBank.from_file(config["benchmark_root"] + "/" + config["semantic_bank"], encoder_name=model_cfg["text_encoder"], dim=self.embed_dim)
        self.semantic_bank_signature = semantic_bank.signature(model_cfg["text_encoder"])
        self.planner = build_planner(
            model_cfg=model_cfg,
            dino_encoder=dino_encoder,
            adapter=adapter,
            semantic_bank=semantic_bank,
            cache_image_embeddings=bool(config.get("cache_image_embeddings", model_cfg.get("cache_image_embeddings", False))),
        ).to(self.device)
        if bool(config.get("precompute_image_embeddings", False)):
            # Phase-II batches already contain aligned embeddings, so keeping the
            # frozen DINO and adapter on CPU preserves GPU memory for Vicuna.
            self.planner.dino_encoder.to("cpu")
            self.planner.adapter.to("cpu")
        self.optimizer = torch.optim.AdamW(
            [p for p in self.planner.parameters() if p.requires_grad],
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    @classmethod
    def _build_stop_targets(
        cls,
        datasets: tuple[PlannerDataset | Subset, ...],
        success_radius: float,
    ) -> dict[str, float]:
        targets: dict[str, float] = {}
        for dataset in datasets:
            root, indices = cls._root_dataset_and_indices(dataset)
            for idx in indices:
                record = root.records[idx]
                episode = root.episodes[record["episode_id"]]
                current = episode["states"][int(record["step"])]["position"]
                goal = episode["states"][-1]["position"]
                targets[str(record["record_id"])] = float(l2(current, goal) <= success_radius)
        return targets

    @staticmethod
    def _root_dataset_and_indices(dataset: PlannerDataset | Subset) -> tuple[PlannerDataset, list[int]]:
        if isinstance(dataset, Subset):
            if not isinstance(dataset.dataset, PlannerDataset):
                raise TypeError("Planner Subset must wrap PlannerDataset")
            return dataset.dataset, [int(i) for i in dataset.indices]
        return dataset, list(range(len(dataset.records)))

    def _collect_image_paths(self) -> list[Path]:
        seen: set[str] = set()
        paths: list[Path] = []
        for dataset in (self.train_dataset, self.val_dataset):
            root, indices = self._root_dataset_and_indices(dataset)
            for idx in indices:
                record = root.records[idx]
                for path in root.image_paths_for_record(record):
                    key = str(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    paths.append(path)
        return paths

    def _set_image_embeddings(self, embeddings: dict[str, torch.Tensor]) -> None:
        for dataset in (self.train_dataset, self.val_dataset):
            root, _ = self._root_dataset_and_indices(dataset)
            root.image_embeddings = embeddings

    @staticmethod
    def _load_embedding_cache(path: Path) -> dict[str, torch.Tensor]:
        if not path.exists():
            return {}
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "embeddings" in payload:
            embeddings = payload["embeddings"]
        else:
            embeddings = payload
        if not isinstance(embeddings, dict):
            raise ValueError(f"Invalid image embedding cache: {path}")
        return {str(k): v.float().cpu() for k, v in embeddings.items() if torch.is_tensor(v)}

    def _precompute_image_embeddings(self, dino_encoder: torch.nn.Module, adapter: torch.nn.Module, image_size: int) -> None:
        if not bool(self.cfg.get("precompute_image_embeddings", False)):
            return

        cache_path = Path(self.cfg.get("image_embedding_cache_path", self.work_dir / "image_embeddings.pt"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self._load_embedding_cache(cache_path)
        paths = self._collect_image_paths()
        missing = [path for path in paths if str(path) not in embeddings]

        dino_encoder.to(self.device)
        adapter.to(self.device)
        dino_encoder.eval()
        adapter.eval()

        batch_size = int(self.cfg.get("image_embedding_batch_size", 64))
        save_every = int(self.cfg.get("image_embedding_save_every", 1000))
        if missing:
            progress = tqdm(range(0, len(missing), batch_size), desc="precompute-image-emb")
            computed = 0
            with torch.no_grad():
                for start in progress:
                    batch_paths = missing[start : start + batch_size]
                    images = torch.stack([load_image_tensor(path, image_size=image_size) for path in batch_paths], dim=0).to(self.device)
                    encoded = adapter(dino_encoder(images)).detach().cpu()
                    for path, embedding in zip(batch_paths, encoded):
                        embeddings[str(path)] = embedding.float()
                    computed += len(batch_paths)
                    if save_every > 0 and computed % save_every == 0:
                        torch.save({"embeddings": embeddings, "meta": {"count": len(embeddings)}}, cache_path)
            torch.save({"embeddings": embeddings, "meta": {"count": len(embeddings)}}, cache_path)
        self._set_image_embeddings(embeddings)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.planner.train(mode=train)
        running = {"total": 0.0, "waypoint_l1": 0.0, "stop_bce": 0.0}
        count = 0
        iterator = tqdm(loader, desc="planner-train" if train else "planner-val")
        accumulation_steps = max(int(self.cfg.get("gradient_accumulation_steps", 1)), 1)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(iterator, start=1):
            batch = self._move_batch(batch)
            target_stop = torch.tensor(
                [self.stop_targets[str(record_id)] for record_id in batch["record_id"]],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.set_grad_enabled(train):
                out = self.planner(batch)
                loss, loss_items = planner_loss(
                    pred_waypoints=out["waypoints"],
                    target_waypoints=batch["future_waypoints"],
                    pred_stop_logit=out["stop_logit"],
                    target_stop=target_stop,
                    waypoint_l1_weight=float(self.cfg["loss"]["waypoint_l1_weight"]),
                    stop_weight=float(self.cfg["loss"]["stop_weight"]),
                )
            if train:
                (loss / accumulation_steps).backward()
                should_step = batch_idx % accumulation_steps == 0 or batch_idx == len(loader)
                if should_step:
                    max_grad_norm = float(self.cfg.get("max_grad_norm", 1.0))
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.planner.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            for k in running:
                running[k] += loss_items[k]
            count += 1
        return {k: v / max(count, 1) for k, v in running.items()}

    def _load_history(self) -> list[dict[str, Any]]:
        metrics_path = self.work_dir / "metrics.json"
        if not metrics_path.exists():
            return []
        metrics = read_json(metrics_path)
        history = metrics.get("history", [])
        return history if isinstance(history, list) else []

    def _load_meta_val(self, path: Path) -> float:
        if not path.exists():
            return float("inf")
        checkpoint = torch.load(path, map_location="cpu")
        meta = checkpoint.get("meta", {})
        try:
            return float(meta["val_total"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    def _validate_checkpoint_compatibility(
        self,
        checkpoint: dict[str, Any],
        path: Path,
    ) -> None:
        checkpoint_model = checkpoint.get("meta", {}).get("config", {}).get("model", {})
        checkpoint_schema = checkpoint_model.get("proprio_schema")
        configured_schema = self.cfg.get("model", {}).get(
            "proprio_schema",
            PROPRIO_SCHEMA,
        )
        if checkpoint_schema != configured_schema:
            raise ValueError(
                f"Cannot use {path} with a different proprioception schema; "
                "retrain from a fresh checkpoint"
            )
        if checkpoint.get("meta", {}).get("semantic_bank") != self.semantic_bank_signature:
            raise ValueError(
                f"Cannot use {path} with a different semantic-bank signature"
            )

    @torch.no_grad()
    def _calibrate_stop_threshold(self) -> tuple[float, dict[str, float]]:
        self.planner.eval()
        probabilities: list[float] = []
        labels: list[int] = []
        for batch in tqdm(self.val_loader, desc="calibrate-stop"):
            batch = self._move_batch(batch)
            output = self.planner(batch)
            probabilities.extend(torch.sigmoid(output["stop_logit"]).detach().float().cpu().tolist())
            labels.extend(int(self.stop_targets[str(record_id)]) for record_id in batch["record_id"])

        candidates = self.cfg.get("stop_threshold_candidates")
        if candidates is None:
            candidates = [index / 100.0 for index in range(5, 100, 5)]
        best_threshold = 0.5
        best_stats = {"f1": -1.0, "precision": 0.0, "recall": 0.0}
        for candidate in candidates:
            threshold = float(candidate)
            predictions = [probability >= threshold for probability in probabilities]
            tp = sum(pred and label == 1 for pred, label in zip(predictions, labels))
            fp = sum(pred and label == 0 for pred, label in zip(predictions, labels))
            fn = sum((not pred) and label == 1 for pred, label in zip(predictions, labels))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            stats = {"f1": f1, "precision": precision, "recall": recall}
            if (f1, -abs(threshold - 0.5)) > (best_stats["f1"], -abs(best_threshold - 0.5)):
                best_threshold = threshold
                best_stats = stats
        return best_threshold, best_stats

    def _maybe_resume(self, last_path: Path, best_path: Path) -> tuple[int, list[dict[str, Any]], float]:
        history: list[dict[str, Any]] = []
        if not bool(self.cfg.get("resume", True)) or not last_path.exists():
            return 1, history, float("inf")

        checkpoint = torch.load(last_path, map_location=self.device)
        self._validate_checkpoint_compatibility(checkpoint, last_path)
        load_planner_state(self.planner, checkpoint)
        meta = checkpoint.get("meta", {})
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            self._validate_checkpoint_compatibility(best_checkpoint, best_path)
        best_val = self._load_meta_val(best_path)
        start_epoch = int(meta.get("epoch", 0)) + 1
        history = self._load_history()
        try:
            last_val = float(meta["val_total"])
        except (KeyError, TypeError, ValueError):
            last_val = float("inf")
        best_val = min(best_val, last_val)
        if not best_path.exists():
            save_planner(self.planner, best_path, meta=meta)
        return start_epoch, history, best_val

    def train(self) -> dict[str, Any]:
        best_path = self.work_dir / "planner_best.pt"
        last_path = self.work_dir / "planner_last.pt"
        start_epoch, history, best_val = self._maybe_resume(last_path, best_path)
        for epoch in range(start_epoch, int(self.cfg["epochs"]) + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(entry)
            checkpoint_meta = {
                "epoch": epoch,
                "config": self.cfg,
                "planner_variant": normalize_planner_variant(self.cfg["model"]),
                "val_total": val_metrics["total"],
                "semantic_bank": self.semantic_bank_signature,
            }
            save_planner(self.planner, last_path, meta=checkpoint_meta)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                save_planner(self.planner, best_path, meta=checkpoint_meta)
            write_json({"history": history, "best_val": best_val, "best_ckpt": str(best_path), "last_ckpt": str(last_path)}, self.work_dir / "metrics.json")
        best_checkpoint = torch.load(best_path, map_location=self.device)
        load_planner_state(self.planner, best_checkpoint)
        stop_threshold, stop_calibration = self._calibrate_stop_threshold()
        best_meta = dict(best_checkpoint.get("meta", {}))
        best_meta.update({"stop_threshold": stop_threshold, "stop_calibration": stop_calibration})
        save_planner(self.planner, best_path, meta=best_meta)
        summary = {
            "history": history,
            "best_val": best_val,
            "best_ckpt": str(best_path),
            "last_ckpt": str(last_path),
            "stop_threshold": stop_threshold,
            "stop_calibration": stop_calibration,
        }
        write_json(summary, self.work_dir / "metrics.json")
        return summary
