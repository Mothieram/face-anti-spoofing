"""Inference utilities for CDCN++ and ensemble min-fusion.

Example:
    from infer_cdcnpp import load, infer_image, ensemble_min_fusion

    model = load("cdcnpp.pth")
    cdcn_score = infer_image(model, "sample_face.jpg")
    final_score = ensemble_min_fusion(iadg_score=0.82, safas_score=0.77, cdcn_score=cdcn_score)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from CDCNPP import CDCNpp, load_cdcnpp


def _preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    img = default_loader(image_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(img).unsqueeze(0)


def load(path: str, device: Optional[str] = None) -> CDCNpp:
    """Load CDCN++ weights from .pth, handling DataParallel checkpoints."""
    d = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CDCNpp().to(d)

    try:
        ckpt = torch.load(path, map_location=d, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=d)
    state_dict = ckpt.get("state_dict", ckpt)

    # Remove DataParallel prefix when present.
    first_key = next(iter(state_dict), "")
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def infer_tensor(model: CDCNpp, image_tensor: torch.Tensor) -> float:
    """Return scalar spoof probability in [0, 1]."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    out = model(image_tensor)
    return float(out["spoof_prob"].flatten()[0].detach().cpu())


@torch.no_grad()
def infer_image(model: CDCNpp, image_path: str, image_size: int = 256) -> float:
    """Load image file and return scalar spoof score."""
    t = _preprocess_image(image_path, image_size=image_size)
    return infer_tensor(model, t)


def ensemble_min_fusion(iadg_score: float, safas_score: float, cdcn_score: float) -> float:
    """Final ensemble score per requirement.

    final_score = min(iadg_score, safas_score, cdcn_score)
    """
    return float(min(float(iadg_score), float(safas_score), float(cdcn_score)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDCN++ inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to cdcnpp .pth")
    parser.add_argument("--image", type=str, required=True, help="Path to RGB face image")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--iadg_score", type=float, default=None, help="Optional IADG score for fusion")
    parser.add_argument("--safas_score", type=float, default=None, help="Optional SAFAS score for fusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load(args.weights, device=args.device)
    cdcn_score = infer_image(model, args.image)
    print(f"CDCN spoof score: {cdcn_score:.6f}")

    if args.iadg_score is not None and args.safas_score is not None:
        final_score = ensemble_min_fusion(args.iadg_score, args.safas_score, cdcn_score)
        print(f"Ensemble final score (min): {final_score:.6f}")


if __name__ == "__main__":
    main()
