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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_head", type=int, default=5, help="Head-only epochs for all models")
    parser.add_argument("--epochs_full", type=int, default=10, help="Full-model epochs for all models")
    args = parser.parse_args()

    t0 = time.time()
    print("\n=== Fine-tuning all 4 models ===")

    # IADG models
    finetune_iadg.run_training("ICM2O", args.epochs_head, args.epochs_full)
    finetune_iadg.run_training("IOM2C", args.epochs_head, args.epochs_full)

    # SASF models
    finetune_sasf.EPOCHS_HEAD = args.epochs_head
    finetune_sasf.EPOCHS_FULL = args.epochs_full
    for model_name, (_h, w) in finetune_sasf.SASF_MODELS.items():
        finetune_sasf.finetune_one_model(model_name, img_size=w)

    print(f"\nDone. Total time: {(time.time() - t0) / 60.0:.1f} minutes")
    print("Saved weights under: finetuned_weights/")


if __name__ == "__main__":
    main()

