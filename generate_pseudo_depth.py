"""
generate_pseudo_depth.py
========================
Generate aligned pseudo-depth maps for CDCN++ training.

Input structure (default):
  data/train/real, data/train/spoof
  data/val/real,   data/val/spoof

Output structure (default):
  data/depth/train/real, data/depth/train/spoof
  data/depth/val/real,   data/depth/val/spoof

Depth files keep the same relative path as source images and are saved as .png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize01(arr: np.ndarray) -> np.ndarray:
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_max - a_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - a_min) / (a_max - a_min)).astype(np.float32)


def _radial_prior(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    rx = (xx - cx) / max(cx, 1.0)
    ry = (yy - cy) / max(cy, 1.0)
    r2 = rx * rx + ry * ry
    prior = np.exp(-2.8 * r2)  # center high, border low
    return _normalize01(prior)


def _edge_inverse(gray01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = _normalize01(mag)
    return 1.0 - mag


def _pseudo_depth_real(gray01: np.ndarray, radial: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray01, (0, 0), sigmaX=2.0, sigmaY=2.0)
    smooth = _normalize01(smooth)
    edge_inv = _edge_inverse(gray01)
    depth = 0.62 * radial + 0.23 * smooth + 0.15 * edge_inv
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return np.clip(depth, 0.0, 1.0)


def _pseudo_depth_spoof(gray01: np.ndarray, radial: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray01, (0, 0), sigmaX=2.5, sigmaY=2.5)
    smooth = _normalize01(smooth)
    edge_inv = _edge_inverse(gray01)
    # Flatter map for spoof classes (print/replay/mask projected as weaker depth)
    depth = 0.20 + 0.18 * smooth + 0.10 * radial + 0.08 * edge_inv
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return np.clip(depth, 0.0, 1.0)


def generate_depth_map(image_bgr: np.ndarray, class_name: str, out_size: int) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid input image for depth generation.")

    img = cv2.resize(image_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radial = _radial_prior(out_size, out_size)

    if class_name.lower() == "real":
        depth = _pseudo_depth_real(gray, radial)
    else:
        depth = _pseudo_depth_spoof(gray, radial)
    return (depth * 255.0).astype(np.uint8)


def iter_images(root: Path) -> Iterable[Tuple[Path, str, str, Path]]:
    for split in ("train", "val"):
        for cls in ("real", "spoof"):
            base = root / split / cls
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    rel = p.relative_to(base)
                    yield p, split, cls, rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aligned pseudo-depth maps.")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root containing train/ and val/")
    parser.add_argument("--out_dir", type=str, default="data/depth", help="Output depth root")
    parser.add_argument("--size", type=int, default=256, help="Depth map size (square)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth maps")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_images(data_dir))
    if not items:
        raise RuntimeError(f"No images found under: {data_dir}")

    written, skipped = 0, 0
    for src_path, split, cls, rel_path in tqdm(items, desc="Generating depth"):
        dst_path = (out_dir / split / cls / rel_path).with_suffix(".png")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        depth_u8 = generate_depth_map(img, cls, args.size)
        ok = cv2.imwrite(str(dst_path), depth_u8)
        if not ok:
            raise RuntimeError(f"Failed writing depth map: {dst_path}")
        written += 1

    print(f"Done. Wrote: {written} | Skipped existing: {skipped}")
    print(f"Depth root: {out_dir.resolve()}")
    print("Structure: depth/train/real|spoof and depth/val/real|spoof")


if __name__ == "__main__":
    main()

