"""Train CDCN++ for face anti-spoofing.

Expected image layout:
  <data_dir>/train/real/*.jpg
  <data_dir>/train/spoof/*.jpg
  <data_dir>/val/real/*.jpg
  <data_dir>/val/spoof/*.jpg

Optional depth maps (grayscale):
  <depth_dir>/train/real/<same_name>.png
  <depth_dir>/train/spoof/<same_name>.png
  <depth_dir>/val/real/<same_name>.png
  <depth_dir>/val/spoof/<same_name>.png

If a depth map is missing, pseudo supervision is used:
  real -> depth map of ones
  spoof -> depth map of zeros
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets.folder import default_loader


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from CDCNPP import CDCNpp, build_cdcnpp, save_cdcnpp  # noqa: E402


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class FaceDepthDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        split: str,
        depth_root: Optional[str] = None,
        image_size: int = 256,
    ) -> None:
        self.image_root = Path(image_root)
        self.depth_root = Path(depth_root) if depth_root else None
        self.split = split
        self.image_size = int(image_size)

        self.samples: List[Tuple[Path, int, Path]] = []
        for class_name, spoof_label in (("real", 0), ("spoof", 1)):
            class_dir = self.image_root / split / class_name
            if not class_dir.is_dir():
                continue
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    rel = p.relative_to(class_dir)
                    self.samples.append((p, spoof_label, Path(class_name) / rel))

        if not self.samples:
            raise RuntimeError(f"No samples found in {self.image_root / split}")

        self.img_tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.depth_tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_depth_or_pseudo(self, rel_path: Path, spoof_label: int) -> torch.Tensor:
        if self.depth_root is not None:
            base = (self.depth_root / self.split / rel_path).with_suffix("")
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                candidate = base.with_suffix(ext)
                if candidate.is_file():
                    dimg = default_loader(str(candidate)).convert("L")
                    return self.depth_tfm(dimg)

        # Pseudo depth supervision fallback.
        val = 0.0 if spoof_label == 1 else 1.0
        return torch.full((1, self.image_size, self.image_size), fill_value=val, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, spoof_label, rel_path = self.samples[idx]
        img = default_loader(str(img_path)).convert("RGB")
        image_tensor = self.img_tfm(img)
        depth_tensor = self._load_depth_or_pseudo(rel_path, spoof_label)
        spoof_target = torch.tensor([float(spoof_label)], dtype=torch.float32)
        return image_tensor, depth_tensor, spoof_target


def train_one_epoch(
    model: CDCNpp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mse_loss: nn.Module,
    bce_loss: nn.Module,
    device: torch.device,
    lambda_depth: float,
    lambda_cls: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    loss_sum, depth_sum, cls_sum, correct, total = 0.0, 0.0, 0.0, 0, 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [train]",
        leave=False,
        dynamic_ncols=True,
    )
    for images, depth_targets, spoof_targets in pbar:
        images = images.to(device, non_blocking=True)
        depth_targets = depth_targets.to(device, non_blocking=True)
        spoof_targets = spoof_targets.to(device, non_blocking=True)

        out = model(images)
        depth_map = out["depth_map"]
        spoof_logit = out["spoof_logit"]

        loss_depth = mse_loss(depth_map, depth_targets)
        loss_cls = bce_loss(spoof_logit, spoof_targets)
        loss = lambda_depth * loss_depth + lambda_cls * loss_cls

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = images.size(0)
        loss_sum += loss.item() * batch
        depth_sum += loss_depth.item() * batch
        cls_sum += loss_cls.item() * batch

        preds = (torch.sigmoid(spoof_logit) >= 0.5).float()
        correct += (preds == spoof_targets).sum().item()
        total += batch
        pbar.set_postfix(
            loss=f"{(loss_sum / max(total, 1)):.4f}",
            acc=f"{(correct / max(total, 1)):.3f}",
        )

    return {
        "loss": loss_sum / max(total, 1),
        "loss_depth": depth_sum / max(total, 1),
        "loss_cls": cls_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model: CDCNpp,
    loader: DataLoader,
    mse_loss: nn.Module,
    bce_loss: nn.Module,
    device: torch.device,
    lambda_depth: float,
    lambda_cls: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    loss_sum, depth_sum, cls_sum, correct, total = 0.0, 0.0, 0.0, 0, 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [val]  ",
        leave=False,
        dynamic_ncols=True,
    )
    for images, depth_targets, spoof_targets in pbar:
        images = images.to(device, non_blocking=True)
        depth_targets = depth_targets.to(device, non_blocking=True)
        spoof_targets = spoof_targets.to(device, non_blocking=True)

        out = model(images)
        depth_map = out["depth_map"]
        spoof_logit = out["spoof_logit"]

        loss_depth = mse_loss(depth_map, depth_targets)
        loss_cls = bce_loss(spoof_logit, spoof_targets)
        loss = lambda_depth * loss_depth + lambda_cls * loss_cls

        batch = images.size(0)
        loss_sum += loss.item() * batch
        depth_sum += loss_depth.item() * batch
        cls_sum += loss_cls.item() * batch

        preds = (torch.sigmoid(spoof_logit) >= 0.5).float()
        correct += (preds == spoof_targets).sum().item()
        total += batch
        pbar.set_postfix(
            loss=f"{(loss_sum / max(total, 1)):.4f}",
            acc=f"{(correct / max(total, 1)):.3f}",
        )

    return {
        "loss": loss_sum / max(total, 1),
        "loss_depth": depth_sum / max(total, 1),
        "loss_cls": cls_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def _default_resume_path(save_path: str) -> str:
    save = Path(save_path)
    ckpt_dir = save.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return str(ckpt_dir / f"{save.stem}_resume.pth")


def _resume_meta(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "data_dir": str(Path(args.data_dir).resolve()),
        "depth_dir": str(Path(args.depth_dir).resolve()) if args.depth_dir else "",
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lambda_depth": float(args.lambda_depth),
        "lambda_cls": float(args.lambda_cls),
        "epochs": int(args.epochs),
    }


def _save_resume_checkpoint(
    path: str,
    epoch: int,
    model: CDCNpp,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": float(best_val),
            "resume_meta": _resume_meta(args),
        },
        path,
    )


def run_training(args: argparse.Namespace) -> str:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = FaceDepthDataset(args.data_dir, split="train", depth_root=args.depth_dir, image_size=args.image_size)
    val_ds = FaceDepthDataset(args.data_dir, split="val", depth_root=args.depth_dir, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_cdcnpp(device=device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    start_epoch = 1
    save_path = args.save_path
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    resume_path = args.resume_path if args.resume_path else _default_resume_path(save_path)

    if not args.no_resume and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        saved_meta = ckpt.get("resume_meta")
        if saved_meta is not None and saved_meta != _resume_meta(args):
            print("Resume checkpoint config mismatch. Starting fresh training.")
        else:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val = float(ckpt.get("best_val_loss", best_val))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if start_epoch > args.epochs:
                start_epoch = args.epochs + 1
            print(f"[resume] loaded {resume_path}")
            print(f"[resume] best_val_loss={best_val:.6f}, next_epoch={start_epoch}")

    print(f"Training CDCN++ on {device}")
    print(f"train: {len(train_ds)} samples | val: {len(val_ds)} samples")
    print(f"resume checkpoint: {resume_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            mse_loss,
            bce_loss,
            device,
            args.lambda_depth,
            args.lambda_cls,
            epoch,
            args.epochs,
        )
        vl = validate_one_epoch(
            model,
            val_loader,
            mse_loss,
            bce_loss,
            device,
            args.lambda_depth,
            args.lambda_cls,
            epoch,
            args.epochs,
        )
        scheduler.step()

        print(
            f"ep {epoch:02d}/{args.epochs} | "
            f"train loss {tr['loss']:.4f} (d {tr['loss_depth']:.4f} c {tr['loss_cls']:.4f}) acc {tr['acc']:.3f} | "
            f"val loss {vl['loss']:.4f} (d {vl['loss_depth']:.4f} c {vl['loss_cls']:.4f}) acc {vl['acc']:.3f}"
        )

        if vl["loss"] < best_val:
            best_val = vl["loss"]
            save_cdcnpp(
                model,
                save_path,
                extra={
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
            )
            print(f"  [best] saved -> {save_path}")

        _save_resume_checkpoint(
            resume_path,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val,
            args,
        )

    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CDCN++ for face anti-spoofing")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root with train/ and val/")
    parser.add_argument("--depth_dir", type=str, default=None, help="Optional depth map root")
    parser.add_argument("--save_path", type=str, default="weights/cdcnpp.pth", help="Output .pth path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_depth", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional manual path for resume checkpoint")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto resume from checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
