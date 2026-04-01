"""
liveness_finetuned.py
=====================
Drop-in replacement for your existing liveness.py in the attendance system.
Loads fine-tuned weights when available, falls back to originals automatically.

Copy this file into e:/django/bookstore/ as liveness.py (after fine-tuning).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
_ROOT          = os.path.dirname(_HERE)


def _pick_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


_FINETUNED_DIR = _pick_existing_dir([
    os.path.join(_HERE, "finetuned_weights"),
    os.path.join(_ROOT, "finetuned_weights"),
    os.path.join(_HERE, "training", "finetuned_weights"),
])
_WEIGHTS_DIR   = _pick_existing_dir([
    os.path.join(_HERE, "weights"),
    os.path.join(_ROOT, "weights"),
])


def _ft_path(name):
    """Return fine-tuned path if it exists, else original weights path."""
    ft = os.path.join(_FINETUNED_DIR, name)
    return ft if os.path.exists(ft) else None


# Lazy imports — models are loaded once at module level
_models_loaded = False
_aSpoof_ICM2O  = None
_aSpoof_IOM2C  = None
_aSpoofONNX    = None
_aSASF         = None

# ── ensemble weights (sum to 1.0) ─────────────────────────────────────────────
_DEFAULT_WEIGHTS = {
    "ICM2O": 0.35,
    "IOM2C": 0.35,
    "FLRGB":  0.15,
    "SASF":   0.15,
}


def _load_models():
    global _models_loaded, _aSpoof_ICM2O, _aSpoof_IOM2C, _aSpoofONNX, _aSASF

    if _models_loaded:
        return

    from IADG import aSpoof as _aSpoof, aSpoofONNX as _aSpoofONNX_cls
    from SASF import aSASF as _aSASF_cls

    errors = {}

    # ── ICM2O ────────────────────────────────────────────────────────────────
    try:
        _aSpoof_ICM2O = _aSpoof("ICM2O", threshold=0.9980)
        ft = _ft_path("ICM2O_finetuned.pth")
        if ft:
            import torch
            _aSpoof_ICM2O.model.load_state_dict(
                torch.load(ft, map_location="cpu"), strict=False
            )
            print(f"[liveness] ICM2O: loaded fine-tuned weights from {ft}")
        else:
            print("[liveness] ICM2O: using original weights")
    except Exception as e:
        errors["ICM2O"] = str(e)
        _aSpoof_ICM2O = None

    # ── IOM2C ────────────────────────────────────────────────────────────────
    try:
        _aSpoof_IOM2C = _aSpoof("IOM2C", threshold=0.9944)
        ft = _ft_path("IOM2C_finetuned.pth")
        if ft:
            import torch
            _aSpoof_IOM2C.model.load_state_dict(
                torch.load(ft, map_location="cpu"), strict=False
            )
            print(f"[liveness] IOM2C: loaded fine-tuned weights from {ft}")
        else:
            print("[liveness] IOM2C: using original weights")
    except Exception as e:
        errors["IOM2C"] = str(e)
        _aSpoof_IOM2C = None

    # ── FLRGB (ONNX — not fine-tuned, used as-is) ────────────────────────────
    try:
        _aSpoofONNX = _aSpoofONNX_cls("modelrgb", threshold=0.2808)
        print("[liveness] FLRGB: loaded (ONNX, not fine-tuned)")
    except Exception as e:
        errors["FLRGB"] = str(e)
        _aSpoofONNX = None

    # ── SASF ─────────────────────────────────────────────────────────────────
    try:
        _aSASF = _aSASF_cls(threshold=0.0094)
        # SASF uses AntiSpoofPredict which loads weights internally.
        # Fine-tuned state-dicts for SASF would need to be swapped at a lower level.
        # For now, it uses the originals.  See finetune_sasf.py comments.
        print("[liveness] SASF: loaded (original weights)")
    except Exception as e:
        errors["SASF"] = str(e)
        _aSASF = None

    if errors:
        print(f"[liveness] WARNING: some models failed to load: {errors}")

    _models_loaded = True


# ── public API ────────────────────────────────────────────────────────────────

def predict(image_rgb, bbox, landmarks, weights: dict = None):
    """
    Run 4-model ensemble and return (is_spoof, spoof_score, per_model_scores).

    Args:
        image_rgb  : np.ndarray  H×W×3 RGB
        bbox       : face bounding box from detector
        landmarks  : 5-point landmarks from detector, shape (5, 2)
        weights    : optional dict override e.g. {"ICM2O": 0.4, "IOM2C": 0.4, ...}

    Returns:
        is_spoof (bool), spoof_score (float 0–1), scores (dict)
    """
    _load_models()

    w = weights or _DEFAULT_WEIGHTS.copy()

    model_map = {
        "ICM2O": _aSpoof_ICM2O,
        "IOM2C": _aSpoof_IOM2C,
        "FLRGB": _aSpoofONNX,
        "SASF":  _aSASF,
    }

    # remove failed models and redistribute weights
    active = {k: v for k, v in model_map.items() if v is not None}
    if not active:
        raise RuntimeError("No liveness models loaded.")

    missing_weight = sum(w[k] for k in model_map if k not in active)
    if missing_weight > 0 and active:
        scale = 1.0 / (1.0 - missing_weight)
        w = {k: w[k] * scale for k in active}

    def _run(key):
        model = active[key]
        _, prob, _ = model(image_rgb, bbox, landmarks)
        return key, float(prob)

    scores = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {pool.submit(_run, k): k for k in active}
        for fut in as_completed(futures):
            key, prob = fut.result()
            scores[key] = prob

    spoof_score = sum(scores[k] * w[k] for k in scores)
    is_spoof    = spoof_score > 0.5

    return is_spoof, spoof_score, scores
