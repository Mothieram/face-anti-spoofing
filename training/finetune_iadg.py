"""
finetune_iadg.py
================
Fine-tunes ICM2O or IOM2C (Framework model from IADG.py).

Supports automatic resume checkpoints per model.
"""

import os
import sys
import argparse
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from IADG import Framework, _load_checkpoint

DATA_DIR = os.path.join(REPO_ROOT, "data")
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
SAVE_DIR = os.path.join(REPO_ROOT, "finetuned_weights")
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 16
LR_HEAD = 1e-4
LR_FULL = 1e-5
EPOCHS_HEAD = 5
EPOCHS_FULL = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TQDM = os.environ.get("FT_TQDM", "1") == "1"

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_loaders():
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=build_transforms(True))
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=build_transforms(False))

    targets = train_ds.targets
    class_counts = [targets.count(c) for c in range(len(train_ds.classes))]
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  train: {len(train_ds)} images  ({class_counts[0]} real, {class_counts[1]} spoof)")
    print(f"  val  : {len(val_ds)} images")
    print(f"  classes: {train_ds.class_to_idx}")
    return train_dl, val_dl


def load_model(model_name):
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print(f"  loading checkpoint: {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path, map_location="cpu")
    model_defs = ckpt["args"].model
    state_dict = ckpt["state_dict"]

    model = Framework(**model_defs["params"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys : {missing[:5]} ...")
    if unexpected:
        print(f"  [warn] unexpected   : {unexpected[:5]} ...")
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = ("Classifier.fc" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [phase 1] trainable params: {trainable:,}")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [phase 2] trainable params: {trainable:,}")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    batches = tqdm(loader, desc="train", leave=False) if USE_TQDM else loader
    for imgs, labels in batches:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)["out"]
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    batches = tqdm(loader, desc="val", leave=False) if USE_TQDM else loader
    for imgs, labels in batches:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)["out"]
        loss = criterion(out, labels)
        preds = out.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total
    apcer = fp / max(fp + tn, 1)
    bpcer = fn / max(fn + tp, 1)
    acer = (apcer + bpcer) / 2
    return total_loss / total, acc, acer, apcer, bpcer


def _ckpt_path(model_name):
    return os.path.join(CKPT_DIR, f"{model_name}_resume.pth")


def _save_resume_checkpoint(model_name, phase, epoch, model, optimizer, scheduler,
                            best_acer, best_state, epochs_head, epochs_full):
    torch.save(
        {
            "phase": phase,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acer": float(best_acer),
            "best_state": best_state,
            "epochs_head": int(epochs_head),
            "epochs_full": int(epochs_full),
        },
        _ckpt_path(model_name),
    )


def run_training(model_name, epochs_head, epochs_full, resume=True):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {model_name}")
    print(f"{'='*60}")

    train_dl, val_dl = build_loaders()
    model = load_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_acer = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    resume_phase = "head"
    resume_epoch = 0
    resume_opt_state = None
    resume_sch_state = None

    if resume:
        ckpt_file = _ckpt_path(model_name)
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=DEVICE)
            resume_phase = ckpt.get("phase", "head")
            resume_epoch = int(ckpt.get("epoch", 0))
            model.load_state_dict(ckpt["model_state"])
            best_acer = float(ckpt.get("best_acer", best_acer))
            best_state = ckpt.get("best_state", best_state)
            resume_opt_state = ckpt.get("optimizer_state")
            resume_sch_state = ckpt.get("scheduler_state")
            print(f"  [resume] loaded: {ckpt_file}")
            print(f"  [resume] phase={resume_phase} epoch={resume_epoch}")
            if resume_phase == "done":
                save_path = os.path.join(SAVE_DIR, f"{model_name}_finetuned.pth")
                torch.save(best_state, save_path)
                print(f"  [resume] already finished. Best ACER: {best_acer:.4f}")
                print(f"  [resume] Saved best weights -> {save_path}")
                return save_path

    print(f"\n--- Phase 1: classifier head only ({epochs_head} epochs) ---")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_head)

    start_head = 1
    if resume_phase == "head":
        if resume_opt_state is not None:
            optimizer.load_state_dict(resume_opt_state)
        if resume_sch_state is not None:
            scheduler.load_state_dict(resume_sch_state)
        start_head = max(1, resume_epoch + 1)
        if start_head > epochs_head:
            start_head = epochs_head + 1
        if start_head > 1:
            print(f"  [resume] continuing phase 1 from epoch {start_head}/{epochs_head}")

    for epoch in range(start_head, epochs_head + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, criterion)
        scheduler.step()
        print(
            f"  ep {epoch:02d}/{epochs_head}  "
            f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
            f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
            f"{time.time()-t0:.1f}s"
        )
        if acer < best_acer:
            best_acer = acer
            best_state = copy.deepcopy(model.state_dict())
        _save_resume_checkpoint(
            model_name, "head", epoch, model, optimizer, scheduler,
            best_acer, best_state, epochs_head, epochs_full
        )

    print(f"\n--- Phase 2: full model ({epochs_full} epochs) ---")
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_full)

    start_full = 1
    if resume_phase == "full":
        if resume_opt_state is not None:
            optimizer.load_state_dict(resume_opt_state)
        if resume_sch_state is not None:
            scheduler.load_state_dict(resume_sch_state)
        start_full = max(1, resume_epoch + 1)
        if start_full > epochs_full:
            start_full = epochs_full + 1
        if start_full > 1:
            print(f"  [resume] continuing phase 2 from epoch {start_full}/{epochs_full}")

    for epoch in range(start_full, epochs_full + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, criterion)
        scheduler.step()
        print(
            f"  ep {epoch:02d}/{epochs_full}  "
            f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
            f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
            f"{time.time()-t0:.1f}s"
        )
        if acer < best_acer:
            best_acer = acer
            best_state = copy.deepcopy(model.state_dict())
        _save_resume_checkpoint(
            model_name, "full", epoch, model, optimizer, scheduler,
            best_acer, best_state, epochs_head, epochs_full
        )

    save_path = os.path.join(SAVE_DIR, f"{model_name}_finetuned.pth")
    torch.save(best_state, save_path)
    _save_resume_checkpoint(
        model_name, "done", epochs_full, model, None, None,
        best_acer, best_state, epochs_head, epochs_full
    )
    print(f"\n  [ok] Best ACER: {best_acer:.4f}")
    print(f"  [ok] Saved -> {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ICM2O", choices=["ICM2O", "IOM2C"], help="Which IADG model to fine-tune")
    parser.add_argument("--epochs_head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs_full", type=int, default=EPOCHS_FULL)
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic resume from checkpoint")
    args = parser.parse_args()
    run_training(args.model, args.epochs_head, args.epochs_full, resume=not args.no_resume)


if __name__ == "__main__":
    main()
