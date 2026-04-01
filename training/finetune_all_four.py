"""
finetune_all_four.py
====================
One-command runner to fine-tune all 4 models:
  - ICM2O
  - IOM2C
  - 2.7_80x80_MiniFASNetV2
  - 4_0_0_80x80_MiniFASNetV1SE
"""

import os
import sys
import time
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import finetune_iadg
import finetune_sasf


IADG_MODELS = ["ICM2O", "IOM2C"]


def _build_sasf_aliases():
    aliases = {}
    for model_name in finetune_sasf.SASF_MODELS.keys():
        aliases[model_name] = model_name
        aliases[os.path.splitext(model_name)[0]] = model_name
    return aliases


def _resolve_models(requested_models):
    sasf_aliases = _build_sasf_aliases()
    all_models = IADG_MODELS + list(finetune_sasf.SASF_MODELS.keys())

    if not requested_models:
        return IADG_MODELS, list(finetune_sasf.SASF_MODELS.keys())

    iadg_selected = []
    sasf_selected = []
    unknown = []

    for raw_name in requested_models:
        name = raw_name.strip()
        if name in IADG_MODELS:
            if name not in iadg_selected:
                iadg_selected.append(name)
            continue
        canonical_sasf = sasf_aliases.get(name)
        if canonical_sasf:
            if canonical_sasf not in sasf_selected:
                sasf_selected.append(canonical_sasf)
            continue
        unknown.append(raw_name)

    if unknown:
        valid_iadg = ", ".join(IADG_MODELS)
        valid_sasf = ", ".join(all_models[2:])
        valid_sasf_short = ", ".join(
            os.path.splitext(n)[0] for n in all_models[2:]
        )
        raise ValueError(
            "Unknown model(s): "
            + ", ".join(unknown)
            + f"\nValid IADG: {valid_iadg}"
            + f"\nValid SASF (full): {valid_sasf}"
            + f"\nValid SASF (short): {valid_sasf_short}"
        )

    return iadg_selected, sasf_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_head", type=int, default=5, help="Head-only epochs for all models")
    parser.add_argument("--epochs_full", type=int, default=10, help="Full-model epochs for all models")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable automatic resume from per-model checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional subset of models to train. "
            "Examples: IOM2C or 2.7_80x80_MiniFASNetV2(.pth). "
            "Default: train all 4."
        ),
    )
    args = parser.parse_args()

    iadg_models, sasf_models = _resolve_models(args.models)

    t0 = time.time()
    selected = iadg_models + sasf_models
    print(f"\n=== Fine-tuning {len(selected)} model(s) ===")
    print("Selected:", ", ".join(selected))

    # IADG models
    for model_name in iadg_models:
        finetune_iadg.run_training(
            model_name, args.epochs_head, args.epochs_full, resume=not args.no_resume
        )

    # SASF models
    finetune_sasf.EPOCHS_HEAD = args.epochs_head
    finetune_sasf.EPOCHS_FULL = args.epochs_full
    for model_name in sasf_models:
        (_h, w) = finetune_sasf.SASF_MODELS[model_name]
        finetune_sasf.finetune_one_model(model_name, img_size=w, resume=not args.no_resume)

    print(f"\nDone. Total time: {(time.time() - t0) / 60.0:.1f} minutes")
    print("Saved weights under: finetuned_weights/")


if __name__ == "__main__":
    main()
