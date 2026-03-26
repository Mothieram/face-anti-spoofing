"""
finetune_sasf.py
================
Fine-tunes the two MiniFASNet models used by SASF.py:
  - 2.7_80x80_MiniFASNetV2.pth
  - 4_0_0_80x80_MiniFASNetV1SE.pth

SASF outputs 3 classes: [spoof_low, real, spoof_high]
So the label mapping is:  real=1,  spoof=0  (class index of "real" in the 3-way head)
We fine-tune using a 2-class loss that fuses spoof_low + spoof_high vs real.

Usage:
  python finetune_sasf.py
"""

import os, sys, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
DATA_DIR    = os.path.join(REPO_ROOT, "data")
SAVE_DIR    = os.path.join(REPO_ROOT, "finetuned_weights")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
LR_HEAD     = 1e-4
LR_FULL     = 5e-5
EPOCHS_HEAD = 5
EPOCHS_FULL = 10

# SASF model names → input sizes (from parse_model_name)
SASF_MODELS = {
    "2.7_80x80_MiniFASNetV2.pth":    (80, 80),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (80, 80),
}
# ──────────────────────────────────────────────────────────────────────────────

def build_transforms(img_size=80, train=True):
    base = [transforms.Resize((img_size, img_size))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.15),
            transforms.RandomGrayscale(p=0.05),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]
    return transforms.Compose(base)


def build_loaders(img_size=80):
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                    transform=build_transforms(img_size, True))
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                    transform=build_transforms(img_size, False))

    targets      = train_ds.targets
    class_counts = [targets.count(c) for c in range(len(train_ds.classes))]
    weights      = [1.0 / class_counts[t] for t in targets]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    print(f"  classes: {train_ds.class_to_idx}")   # real=0, spoof=1
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")
    return train_dl, val_dl, train_ds.class_to_idx


def load_sasf_model_and_predict(model_name):
    """
    Load a MiniFASNet via the src.anti_spoof_predict module the same way SASF.py does,
    then expose the underlying PyTorch nn.Module so we can fine-tune it.
    """
    from src.anti_spoof_predict import AntiSpoofPredict
    predictor = AntiSpoofPredict(device_id=0 if torch.cuda.is_available() else -1)

    model_path = os.path.join(WEIGHTS_DIR, model_name)
    # AntiSpoofPredict._load_model returns the nn.Module
    model = predictor._load_model(model_path)
    return model


# ── training helpers ──────────────────────────────────────────────────────────

def sasf_loss(logits, labels, class_to_idx):
    """
    SASF outputs 3 logits: [spoof_low, real, spoof_high]
    We merge spoof_low + spoof_high into a single spoof score:
        real_score  = logits[:, 1]
        spoof_score = logits[:, 0] + logits[:, 2]
    Then apply binary cross entropy.
    ImageFolder labels: real=class_to_idx['real'], spoof=class_to_idx['spoof']
    """
    real_idx  = class_to_idx.get("real",  0)
    # Convert: real_idx → binary label 1 (real), else 0 (spoof)
    bin_labels = (labels == real_idx).float()    # 1 = real, 0 = spoof

    probs       = torch.softmax(logits, dim=1)
    real_prob   = probs[:, 1]                    # "real" class in 3-way head
    # binary cross entropy: target 1 = real
    loss = F.binary_cross_entropy(real_prob.clamp(1e-6, 1-1e-6), bin_labels)
    return loss, real_prob, bin_labels


def train_one_epoch(model, loader, optimizer, class_to_idx):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits        = model(imgs)           # (B, 3)
        loss, real_prob, bin_labels = sasf_loss(logits, labels, class_to_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds       = (real_prob > 0.5).long()
        correct    += (preds == bin_labels.long()).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, class_to_idx):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits        = model(imgs)
        loss, real_prob, bin_labels = sasf_loss(logits, labels, class_to_idx)
        total_loss   += loss.item() * imgs.size(0)
        preds         = (real_prob > 0.5).long()
        correct      += (preds == bin_labels.long()).sum().item()
        total        += imgs.size(0)
        tp += ((preds == 1) & (bin_labels == 1)).sum().item()
        tn += ((preds == 0) & (bin_labels == 0)).sum().item()
        fp += ((preds == 1) & (bin_labels == 0)).sum().item()
        fn += ((preds == 0) & (bin_labels == 1)).sum().item()
    acc   = correct / total
    apcer = fp / max(fp + tn, 1)
    bpcer = fn / max(fn + tp, 1)
    acer  = (apcer + bpcer) / 2
    return total_loss / total, acc, acer, apcer, bpcer


def finetune_one_model(model_name, img_size):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning SASF model: {model_name}")
    print(f"{'='*60}")

    train_dl, val_dl, class_to_idx = build_loaders(img_size)

    model = load_sasf_model_and_predict(model_name)
    model = model.to(DEVICE)

    best_acer  = 1.0
    best_state = None

    # ── phase 1: freeze everything except classifier ──────────────────────────
    print(f"\n--- Phase 1: head only ({EPOCHS_HEAD} epochs) ---")
    for name, p in model.named_parameters():
        # MiniFASNet typically ends with 'classifier' or 'fc' layer
        p.requires_grad = any(kw in name for kw in ["classifier", "fc", "last"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {n_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_HEAD)

    for epoch in range(1, EPOCHS_HEAD + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, class_to_idx)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, class_to_idx)
        scheduler.step()
        print(f"  ep {epoch:02d}/{EPOCHS_HEAD}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"ACER {acer:.4f}  {time.time()-t0:.1f}s")
        if acer < best_acer:
            best_acer  = acer
            best_state = copy.deepcopy(model.state_dict())

    # ── phase 2: full model ───────────────────────────────────────────────────
    print(f"\n--- Phase 2: full model ({EPOCHS_FULL} epochs) ---")
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_FULL)

    for epoch in range(1, EPOCHS_FULL + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, class_to_idx)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, class_to_idx)
        scheduler.step()
        print(f"  ep {epoch:02d}/{EPOCHS_FULL}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
              f"ACER {acer:.4f}  {time.time()-t0:.1f}s")
        if acer < best_acer:
            best_acer  = acer
            best_state = copy.deepcopy(model.state_dict())

    # ── save ──────────────────────────────────────────────────────────────────
    save_name = model_name.replace(".pth", "_finetuned.pth")
    save_path = os.path.join(SAVE_DIR, save_name)
    torch.save(best_state, save_path)
    print(f"\n  ✓ Best ACER: {best_acer:.4f}")
    print(f"  ✓ Saved → {save_path}")
    return save_path


def main():
    for model_name, (h, w) in SASF_MODELS.items():
        finetune_one_model(model_name, img_size=w)


if __name__ == "__main__":
    main()
