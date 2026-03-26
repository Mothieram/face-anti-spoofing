"""
finetune_iadg.py
================
Fine-tunes ICM2O or IOM2C (the Framework / AENet models from IADG.py)
on your own real + spoof dataset.

The Framework model's final classifier (FeatEmbedder.fc) outputs 2 logits:
  index 0 = real,  index 1 = spoof
so labels are:  real=0, spoof=1

Usage:
  # fine-tune ICM2O
  python finetune_iadg.py --model ICM2O

  # fine-tune IOM2C
  python finetune_iadg.py --model IOM2C

  # fine-tune both sequentially
  python finetune_iadg.py --model ICM2O --epochs 15
  python finetune_iadg.py --model IOM2C --epochs 15
"""

import os, sys, argparse, copy, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# ── make sure your repo root is on the path ───────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from IADG import Framework, _load_checkpoint          # your IADG.py

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(REPO_ROOT, "data")
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
SAVE_DIR    = os.path.join(REPO_ROOT, "finetuned_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE  = 256          # matches the checkpoint's transform.image_size
BATCH_SIZE  = 16
LR_HEAD     = 1e-4         # learning rate for classifier head
LR_FULL     = 1e-5         # learning rate after unfreezing backbone
EPOCHS_HEAD = 5            # phase 1: head only
EPOCHS_FULL = 10           # phase 2: full unfrozen
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

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
    """
    ImageFolder expects:
        data/train/real/  data/train/spoof/
        data/val/real/    data/val/spoof/
    Class indices: real=0, spoof=1  (alphabetical)
    """
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                    transform=build_transforms(True))
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                    transform=build_transforms(False))

    # ── balanced sampler (handles class imbalance automatically) ──────────────
    targets      = train_ds.targets
    class_counts = [targets.count(c) for c in range(len(train_ds.classes))]
    weights      = [1.0 / class_counts[t] for t in targets]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    print(f"  train: {len(train_ds)} images  ({class_counts[0]} real, {class_counts[1]} spoof)")
    print(f"  val  : {len(val_ds)} images")
    print(f"  classes: {train_ds.class_to_idx}")   # should be {'real': 0, 'spoof': 1}
    return train_dl, val_dl


def load_model(model_name):
    """Load the Framework model from the original checkpoint."""
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print(f"  loading checkpoint: {ckpt_path}")
    ckpt      = _load_checkpoint(ckpt_path, map_location="cpu")
    model_defs = ckpt["args"].model
    state_dict = ckpt["state_dict"]

    model = Framework(**model_defs["params"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys : {missing[:5]} …")
    if unexpected:
        print(f"  [warn] unexpected   : {unexpected[:5]} …")
    return model


def freeze_backbone(model):
    """Freeze everything except the final fc layer."""
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
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)["out"]           # shape: (B, 2)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds       = out.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out   = model(imgs)["out"]
        loss  = criterion(out, labels)
        preds = out.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        # spoof = class 1
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
    acc   = correct / total
    apcer = fp / max(fp + tn, 1)   # attack presentation classification error rate
    bpcer = fn / max(fn + tp, 1)   # bonafide presentation classification error rate
    acer  = (apcer + bpcer) / 2    # avg classification error rate (main metric)
    return total_loss / total, acc, acer, apcer, bpcer


def run_training(model_name, epochs_head, epochs_full):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {model_name}")
    print(f"{'='*60}")

    train_dl, val_dl = build_loaders()
    model = load_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_acer = 1.0
    best_state = None

    # ── PHASE 1: head only ────────────────────────────────────────────────────
    print(f"\n--- Phase 1: classifier head only ({epochs_head} epochs) ---")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_head)

    for epoch in range(1, epochs_head + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, criterion)
        scheduler.step()
        print(f"  ep {epoch:02d}/{epochs_head}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
              f"{time.time()-t0:.1f}s")
        if acer < best_acer:
            best_acer  = acer
            best_state = copy.deepcopy(model.state_dict())

    # ── PHASE 2: full model ───────────────────────────────────────────────────
    print(f"\n--- Phase 2: full model ({epochs_full} epochs) ---")
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_full)

    for epoch in range(1, epochs_full + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, criterion)
        scheduler.step()
        print(f"  ep {epoch:02d}/{epochs_full}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
              f"{time.time()-t0:.1f}s")
        if acer < best_acer:
            best_acer  = acer
            best_state = copy.deepcopy(model.state_dict())

    # ── SAVE BEST ─────────────────────────────────────────────────────────────
    save_path = os.path.join(SAVE_DIR, f"{model_name}_finetuned.pth")
    torch.save(best_state, save_path)
    print(f"\n  ✓ Best ACER: {best_acer:.4f}")
    print(f"  ✓ Saved → {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="ICM2O",
                        choices=["ICM2O", "IOM2C"], help="Which IADG model to fine-tune")
    parser.add_argument("--epochs_head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs_full", type=int, default=EPOCHS_FULL)
    args = parser.parse_args()
    run_training(args.model, args.epochs_head, args.epochs_full)


if __name__ == "__main__":
    main()
