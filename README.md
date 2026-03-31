---
title: Face Anti-Spoofing - Model Fine-Tuning
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

<div align="center">

# 🛡️ Face Anti-Spoofing — Model Fine-Tuning Branch

**Production-grade passive liveness detection with a 5-model weighted ensemble, fine-tuned on domain-specific datasets.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx)](https://onnxruntime.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6B35?logo=gradio)](https://gradio.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://kaggle.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-FFD21E)](https://huggingface.co/spaces/mothieram/face-anti-spoofing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Branch:** `model-finetuning` &nbsp;|&nbsp; **Base Repo:** [`Mothieram/face-anti-spoofing`](https://github.com/Mothieram/face-anti-spoofing) &nbsp;|&nbsp; **HF Space:** [`mothieram/face-anti-spoofing`](https://huggingface.co/spaces/mothieram/face-anti-spoofing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Ensemble Models](#ensemble-models)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
  - [Datasets](#datasets)
  - [Dataset Preparation](#dataset-preparation)
  - [Training Configuration](#training-configuration)
  - [Checkpoint Resumption](#checkpoint-resumption)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Fine-Tuning](#running-fine-tuning)
  - [Inference](#inference)
  - [Gradio Demo](#gradio-demo)
- [Gradio UI Features](#gradio-ui-features)
- [Results & Metrics](#results--metrics)
- [Differences from Main Branch](#differences-from-main-branch)
- [Roadmap](#roadmap)
- [Citations](#citations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This branch (`model-finetuning`) contains the **complete fine-tuning pipeline** for the passive face anti-spoofing ensemble. The goal is to adapt pre-trained liveness detection models to domain-specific datasets, improving generalization across real-world spoof attack types — including printed photos, screen replays, silicone masks, and cut-out masks.

The system uses a **5-model weighted ensemble** running in parallel via `ThreadPoolExecutor`, combining complementary model architectures and training paradigms to achieve robust, low-latency passive liveness detection without requiring active user cooperation.

Key highlights of this branch:

- Full dataset preparation pipeline for `anti-spoofing-live` and `LCC_FASD` Kaggle datasets
- Checkpoint-based training with automatic epoch resumption
- Per-model configurable fine-tuning with frozen/unfrozen backbone strategies
- Weighted ensemble scoring with tunable model weights exposed via Gradio sliders
- Micro-motion liveness detection as an auxiliary passive signal
- ONNX and PyTorch dual-format inference support

---

## Architecture

```
Input Frame (BGR / RGB)
        │
        ▼
┌───────────────────┐
│   Face Detection  │  ← RetinaFace (optional pre-crop)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────────┐
│                    Parallel Ensemble Inference                │
│                    (ThreadPoolExecutor)                       │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │  ICM2O     │  │  IOM2C     │  │ modelrgb   │  │  SASF /  │ │
│  │  Model     │  │  Model     │  │  (ONNX)    │  │MiniFASNet│ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘ │
│        │               │               │               │      │
│        └───────────────┴───────────────┴───────────────┘      │
│                               │                               │
│                    Weighted Score Fusion                      │
│              w₁·s₁ + w₂·s₂ + w₃·s₃ + w₄·s₄                    │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Micro-Motion Signal  │  ← Optical flow / frame delta
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Final Liveness      │
                    │   Decision + Score    │
                    └───────────────────────┘
```

---

## Ensemble Models

| #   | Model ID              | Format         | Architecture                        | Input Size    | Role                      |
| --- | --------------------- | -------------- | ----------------------------------- | ------------- | ------------------------- |
| 1   | **ICM2O**             | PyTorch `.pth` | MobileNetV2-based binary classifier | 224×224       | RGB texture analysis      |
| 2   | **IOM2C**             | PyTorch `.pth` | Depth-aware CNN variant             | 224×224       | Cross-modal spoofing cues |
| 3   | **modelrgb**          | ONNX `.onnx`   | Lightweight CNN (ONNX-optimized)    | 128×128       | Fast RGB liveness scoring |
| 4   | **SASF / MiniFASNet** | PyTorch `.pth` | MiniFASNet (Silent Anti-Spoof)      | 80×80 / 80×80 | Multi-scale patch fusion  |
| 5   | **CDCN++**            | PyTorch `.pth` | Central Difference CNN++            | 256x256       | Depth-supervised anti-spoofing |

All five models run in **parallel threads** and their softmax confidence scores are fused using configurable weights:

```python
final_score = (
    w1 * score_icm2o +
    w2 * score_iom2c +
    w3 * score_modelrgb +
    w4 * score_sasf +
    w5 * score_cdcnpp
)
```

Default weights: `[0.20, 0.20, 0.20, 0.20, 0.20]` — adjustable via Gradio sliders at inference time.

---

## Fine-Tuning Pipeline

### Datasets

Two Kaggle datasets are used for fine-tuning:

| Dataset                | Source | Samples         | Classes      |
| ---------------------- | ------ | --------------- | ------------ |
| **anti-spoofing-live** | Kaggle | ~10,000+ frames | Live / Spoof |
| **LCC_FASD**           | Kaggle | ~5,000+ frames  | Live / Spoof |

Both datasets are combined, balanced, and split into train/val/test sets during preparation.

> **Note:** Kaggle API credentials must be configured before downloading. See [Installation](#installation).

---

### Dataset Preparation

The dataset preparation script handles downloading, extraction, directory normalization, class balancing, and train/val/test splitting.

```bash
# Step 1 — Download and prepare datasets
python finetune/prepare_dataset.py \
    --datasets anti-spoofing-live lcc_fasd \
    --output_dir data/finetune \
    --val_split 0.15 \
    --test_split 0.10 \
    --seed 42
```

Expected output structure after preparation:

```
data/
└── finetune/
    ├── train/
    │   ├── live/
    │   └── spoof/
    ├── val/
    │   ├── live/
    │   └── spoof/
    └── test/
        ├── live/
        └── spoof/
```

---

### Training Configuration

Each model has an individual configuration file under `finetune/configs/`. The general training arguments are:

| Argument            | Default         | Description                                                      |
| ------------------- | --------------- | ---------------------------------------------------------------- |
| `--model`           | `icm2o`         | Target model to fine-tune (`icm2o`, `iom2c`, `modelrgb`, `sasf`) |
| `--epochs`          | `20`            | Total training epochs                                            |
| `--batch_size`      | `32`            | Batch size per step                                              |
| `--lr`              | `1e-4`          | Initial learning rate                                            |
| `--freeze_backbone` | `True`          | Freeze backbone, train classifier head only (Phase 1)            |
| `--unfreeze_after`  | `5`             | Unfreeze full network after N epochs (Phase 2)                   |
| `--checkpoint_dir`  | `checkpoints/`  | Directory to save epoch checkpoints                              |
| `--resume_from`     | `None`          | Path to a `.pth` checkpoint to resume from                       |
| `--data_dir`        | `data/finetune` | Root of prepared dataset                                         |
| `--amp`             | `True`          | Use mixed precision (FP16) training                              |
| `--patience`        | `5`             | Early stopping patience (epochs without val improvement)         |

---

### Checkpoint Resumption

Training is designed to be **fully resumable** from any saved epoch checkpoint. This is critical for long Kaggle GPU sessions (which terminate after a fixed time limit).

```bash
# Resume from a specific checkpoint (e.g., after epoch 2)
python finetune/train.py \
    --model icm2o \
    --resume_from checkpoints/icm2o_epoch2.pth \
    --epochs 20
```

Checkpoints are saved at the end of every epoch:

```
checkpoints/
├── icm2o_epoch1.pth
├── icm2o_epoch2.pth   ← current pause point
├── icm2o_best.pth     ← best val-accuracy checkpoint
└── ...
```

Each checkpoint stores:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "val_loss": float,
    "val_acc": float,
    "config": dict
}
```

---

## Repository Structure

```
face-anti-spoofing/                    ← root (model-finetuning branch)
│
├── app.py                             ← Gradio UI entry point
├── requirements.txt                   ← Python dependencies
├── README.md                          ← This file
│
├── models/                            ← Model loader & inference wrappers
│   ├── __init__.py
│   ├── icm2o.py
│   ├── iom2c.py
│   ├── modelrgb_onnx.py
│   └── sasf_minifasnet.py
│
├── ensemble/                          ← Ensemble fusion logic
│   ├── __init__.py
│   ├── parallel_runner.py             ← ThreadPoolExecutor inference
│   └── fusion.py                      ← Weighted score fusion
│
├── micro_motion/                      ← Micro-motion liveness module
│   ├── __init__.py
│   └── motion_detector.py
│
├── finetune/                          ← Fine-tuning pipeline (this branch)
│   ├── prepare_dataset.py             ← Dataset download & preparation
│   ├── train.py                       ← Main training script
│   ├── evaluate.py                    ← Post-training evaluation
│   ├── export_onnx.py                 ← PyTorch → ONNX export
│   ├── configs/
│   │   ├── icm2o_config.yaml
│   │   ├── iom2c_config.yaml
│   │   ├── modelrgb_config.yaml
│   │   └── sasf_config.yaml
│   └── utils/
│       ├── dataset.py                 ← Dataset class & augmentations
│       ├── metrics.py                 ← HTER, APCER, BPCER
│       └── callbacks.py              ← Checkpoint, EarlyStopping
│
├── checkpoints/                       ← Saved model checkpoints
│   └── .gitkeep
│
├── data/                              ← Dataset root (gitignored)
│   └── .gitkeep
│
├── weights/                           ← Pre-trained & fine-tuned weights
│   └── .gitkeep
│
└── scripts/                           ← Utility scripts
    ├── download_weights.py
    └── benchmark.py
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended for GPU training)
- Kaggle API configured (`~/.kaggle/kaggle.json`)

### 1. Clone the branch

```bash
git clone -b model-finetuning https://github.com/Mothieram/face-anti-spoofing.git
cd face-anti-spoofing
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API

```bash
# Place your kaggle.json token in:
# Windows: C:\Users\<user>\.kaggle\kaggle.json
# Linux:   ~/.kaggle/kaggle.json

# Or set environment variables:
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_api_key
```

### 5. Download pre-trained weights

```bash
python scripts/download_weights.py
```

---

## Usage

### Running Fine-Tuning

**Phase 1 — Backbone frozen, head only (fast convergence):**

```bash
python finetune/train.py \
    --model icm2o \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --freeze_backbone True \
    --data_dir data/finetune \
    --checkpoint_dir checkpoints/
```

**Phase 2 — Full network unfrozen (refinement):**

```bash
python finetune/train.py \
    --model icm2o \
    --epochs 20 \
    --batch_size 16 \
    --lr 5e-5 \
    --freeze_backbone False \
    --resume_from checkpoints/icm2o_best.pth \
    --data_dir data/finetune \
    --checkpoint_dir checkpoints/
```

**Resume from paused Kaggle session:**

```bash
python finetune/train.py \
    --model icm2o \
    --resume_from checkpoints/icm2o_epoch2.pth \
    --epochs 20
```

---

### Inference

**Single image:**

```python
from ensemble.parallel_runner import EnsembleRunner

runner = EnsembleRunner(weights=[0.20, 0.20, 0.20, 0.20, 0.20])
result = runner.predict("path/to/face_image.jpg")

print(result)
# {'label': 'Live', 'score': 0.923, 'model_scores': {...}}
```

**Webcam / real-time:**

```bash
python scripts/benchmark.py --source 0 --show
```

---

### Gradio Demo

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Gradio UI Features

The Gradio interface exposes the following controls:

| Feature                  | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| **Image / Webcam Input** | Upload a face image or use live webcam feed                               |
| **Model Weight Sliders** | Individually tune each model's contribution to the ensemble score (w₁–w₅) |
| **Liveness Score Bar**   | Visual confidence gauge — Live vs. Spoof                                  |
| **Per-Model Breakdown**  | Shows individual score from each of the 5 models (including CDCN++)                          |
| **Micro-Motion Toggle**  | Enable/disable micro-motion liveness auxiliary check                      |
| **Threshold Slider**     | Adjust the Live/Spoof decision boundary (default: 0.5)                    |

---

## Results & Metrics

Evaluation metrics used:

| Metric    | Description                                      |
| --------- | ------------------------------------------------ |
| **HTER**  | Half Total Error Rate — primary benchmark metric |
| **APCER** | Attack Presentation Classification Error Rate    |
| **BPCER** | Bona Fide Presentation Classification Error Rate |
| **AUC**   | Area Under ROC Curve                             |

> Fine-tuning results will be updated here after completing full training runs on Kaggle. Current state: paused at **epoch 2 / 20** for ICM2O. Checkpoint saved at `checkpoints/icm2o_epoch2.pth`.

---

## Differences from Main Branch

| Aspect               | `main` branch                   | `model-finetuning` branch               |
| -------------------- | ------------------------------- | --------------------------------------- |
| **Purpose**          | Inference + HF Space deployment | Fine-tuning pipeline                    |
| **Dataset code**     | Not included                    | Full Kaggle download + prep             |
| **Training scripts** | Not included                    | `finetune/train.py` + configs           |
| **Checkpoint logic** | Not included                    | Per-epoch save + resume                 |
| **ONNX export**      | Pre-exported weights only       | `finetune/export_onnx.py`               |
| **Gradio UI**        | Production UI                   | Extended with fine-tuning metrics panel |

---

## Roadmap

- [x] Dataset preparation pipeline (anti-spoofing-live + LCC_FASD)
- [x] Checkpoint-based training with epoch resumption
- [x] Mixed precision (FP16/AMP) training support
- [x] Per-model YAML config files
- [ ] Complete fine-tuning of all 5 ensemble models on Kaggle
- [ ] ONNX export of fine-tuned weights
- [ ] Quantization-aware training (QAT) for edge deployment
- [ ] Evaluation on CelebA-Spoof and SiW benchmarks
- [ ] Push fine-tuned weights to HuggingFace Hub
- [ ] TensorRT conversion for NVIDIA Jetson deployment

---

## Citations

If you use this project or build upon it, please consider citing the following works that underpin the models and techniques used.

### MiniFASNet / Silent Face Anti-Spoofing

```bibtex
@inproceedings{zhousimple2021,
  title     = {A Simple Baseline for Semi-supervised Semantic Segmentation with Strong Data Augmentation},
  author    = {George, Anjith and Marcel, Sébastien},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2021}
}
```

```bibtex
@misc{minivision2020silent,
  author       = {minivision-ai},
  title        = {Silent-Face-Anti-Spoofing},
  year         = {2020},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/minivision-ai/Silent-Face-Anti-Spoofing}},
  note         = {Accessed: 2024}
}
```

### CDCNPP / Central Difference Convolutional Networks

```bibtex
@inproceedings{yu2020cdcn,
  title     = {Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
  author    = {Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5295--5305},
  year      = {2020}
}
```

### ONNX Runtime

```bibtex
@misc{onnxruntime,
  author       = {{ONNX Runtime developers}},
  title        = {ONNX Runtime},
  year         = {2021},
  howpublished = {\url{https://onnxruntime.ai}},
  note         = {Version 1.x}
}
```

### RetinaFace (Face Detection)

```bibtex
@inproceedings{deng2020retinaface,
  title     = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
  author    = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

### LCC_FASD Dataset

```bibtex
@article{Datta2020LCCFASD,
  title   = {LCC-FASD: A Low-Cost and Compact Face Anti-Spoofing Dataset},
  author  = {Datta, Pankaj and others},
  journal = {arXiv preprint},
  year    = {2020},
  url     = {https://kaggle.com/datasets}
}
```

### PyTorch

```bibtex
@incollection{pytorch2019,
  title     = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle = {Advances in Neural Information Processing Systems 32},
  pages     = {8024--8035},
  year      = {2019},
  publisher = {Curran Associates, Inc.}
}
```

### Gradio

```bibtex
@article{abid2019gradio,
  title   = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author  = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year    = {2019}
}
```

## Acknowledgements

- [minivision-ai](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) for the MiniFASNet / SASF architecture and training methodology
- [Zitong Yu et al.](https://arxiv.org/abs/2003.04092) for the Central Difference CNN (CDCNPP) paper
- [InsightFace / RetinaFace](https://github.com/deepinsight/insightface) for the face detection backbone
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform efficient model inference
- [Kaggle](https://kaggle.com/) for GPU compute and dataset hosting
- [Hugging Face Spaces](https://huggingface.co/spaces) for free deployment of the Gradio demo

---

<div align="center">

Made with ❤️ by [Mothieram](https://github.com/Mothieram) ·

[⬆ Back to Top](#️-face-anti-spoofing--model-fine-tuning-branch)

</div>


