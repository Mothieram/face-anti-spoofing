"""
convert_onnx_to_pth.py
======================
Convert an ONNX model to a PyTorch artifact (.pth) using onnx2torch.

Usage:
  python training/convert_onnx_to_pth.py --onnx weights/modelrgb.onnx
  python training/convert_onnx_to_pth.py --onnx weights/modelrgb.onnx --out finetuned_weights/modelrgb_from_onnx.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ONNX model to .pth (PyTorch)")
    parser.add_argument("--onnx", type=str, required=True, help="Input .onnx file path")
    parser.add_argument("--out", type=str, default=None, help="Output .pth file path")
    parser.add_argument(
        "--save_state_dict",
        action="store_true",
        help="Save state_dict bundle instead of pickled module object",
    )
    return parser


def _resolve_output_path(onnx_path: Path, out_path: str | None) -> Path:
    if out_path:
        return Path(out_path)
    return onnx_path.with_suffix(".pth")


def _import_deps():
    try:
        import onnx  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'onnx'. Install with: pip install onnx"
        ) from exc

    try:
        from onnx2torch import convert  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'onnx2torch'. Install with: pip install onnx2torch"
        ) from exc
    return onnx, convert


def _save_state_dict_bundle(out_path: Path, onnx_path: Path, model: torch.nn.Module) -> None:
    payload: Dict[str, Any] = {
        "format": "onnx2torch_state_dict_bundle",
        "source_onnx": str(onnx_path.resolve()),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, str(out_path))


def _save_pickled_module(out_path: Path, model: torch.nn.Module) -> None:
    torch.save(model, str(out_path))


def main() -> None:
    args = _build_parser().parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    out_path = _resolve_output_path(onnx_path, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    onnx, convert = _import_deps()
    onnx_model = onnx.load(str(onnx_path))
    model = convert(onnx_model)
    model.eval()

    if args.save_state_dict:
        _save_state_dict_bundle(out_path, onnx_path, model)
        print(f"[ok] Saved state_dict bundle -> {out_path}")
    else:
        _save_pickled_module(out_path, model)
        print(f"[ok] Saved converted module -> {out_path}")
        print("[note] Loading this .pth later requires 'onnx2torch' installed.")


if __name__ == "__main__":
    main()

