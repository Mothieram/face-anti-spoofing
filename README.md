# 🛡️ Face Anti-Spoofing Detector

A production-ready **face liveness detection** system using a **4-model ensemble** with **micro-motion analysis**. Built with Gradio for the UI and deployed on HuggingFace Spaces.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/mothieram/face-anti-spoofing)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.9.0-orange)](https://gradio.app/)

---

## 🧠 How It Works

The system runs **4 spoof detection models in parallel** via `ThreadPoolExecutor`, fuses their scores using configurable weighted averaging, and optionally combines with a **micro-motion liveness check** from the webcam stream.

```
Input Image / Webcam Frame
        │
        ▼
  RetinaFace Detector
        │
   ┌────┴────┐
   │  bbox   │  landmarks
   └────┬────┘
        │
  ┌─────┴──────────────────────────────┐
  │        Parallel Inference          │
  │  SASF  │  FLRGB  │ ICM2O │ IOM2C  │
  └─────┬──────────────────────────────┘
        │
  Weighted Ensemble Score
        │
  (Live tab only)
  Micro-Motion Score  ──┐
  (nose displacement)   │
                        ▼
               Fused Final Verdict
               🟢 REAL  /  🔴 SPOOF
```

---

## 📦 Models

| Model     | Description                                    | Default Weight | Default Threshold |
| --------- | ---------------------------------------------- | -------------- | ----------------- |
| **SASF**  | Silent-Face-Anti-Spoofing (MobileNet)          | 15%            | 0.0094            |
| **FLRGB** | Face Liveness Detection Model-RGB (ONNX)       | 15%            | 0.0553            |
| **ICM2O** | Instance-Aware Domain Generalisation CVPR 2023 | 35%            | 0.9980            |
| **IOM2C** | Instance-Aware Domain Generalisation CVPR 2023 | 35%            | 0.9944            |

> Weights are auto-normalised — if a model fails to load, its weight is redistributed to the remaining models.

---

## 📁 Project Structure

```
face-anti-spoofing/
│
├── app.py                  # Gradio UI — two tabs (Image + Live)
├── liveness_temporal.py    # Micro-motion liveness checker
├── IADG.py                 # FLRGB + ICM2O + IOM2C model wrappers
├── SASF.py                 # Silent-Face-Anti-Spoofing wrapper
├── detector.py             # RetinaFace face detector wrapper
├── models.py               # AENet architecture (ResNet-18 based)
├── tsn_predict.py          # TSN prediction utilities
├── requirements.txt        # Python dependencies
│
├── src/                    # Source utilities
└── weights/                # Model weight files (.pth / .onnx)
    ├── modelrgb.onnx
    ├── ICM2O.pth
    ├── IOM2C.pth
    └── ...
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Mothieram/face-anti-spoofing.git
cd face-anti-spoofing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model weights

Place your model weight files inside the `weights/` folder:

```
weights/
├── modelrgb.onnx
├── ICM2O.pth
└── IOM2C.pth
```

### 4. Run locally

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## 🖥️ UI Tabs

### 📷 Image Tab

- Upload a photo or capture via webcam
- Runs all 4 models in parallel
- Shows per-model score + confidence + ensemble verdict
- Annotates bounding box on result image

### 🎥 Live Tab

- Streams webcam in real time
- Collects **15 frames** of facial landmark data
- Runs micro-motion analysis (nose displacement variance + FFT periodicity check)
- Fuses motion score with spoof ensemble for final verdict
- **Result locks on screen** after verdict — press 🔄 Reset to scan again

---

## ⚙️ Configuration

Both tabs share these controls in the UI:

**Model Thresholds** _(collapsed accordion)_
| Slider | Default | Effect |
|---|---|---|
| SASF threshold | 0.0094 | Score above this → spoof |
| FLRGB threshold | 0.2808 | Score above this → spoof |
| ICM2O threshold | 0.9980 | Score above this → spoof |
| IOM2C threshold | 0.9944 | Score above this → spoof |

**Model Weights** _(open accordion)_
| Slider | Default | Effect |
|---|---|---|
| SASF weight | 0.15 | Contribution to ensemble |
| FLRGB weight | 0.15 | Contribution to ensemble |
| ICM2O weight | 0.35 | Contribution to ensemble |
| IOM2C weight | 0.35 | Contribution to ensemble |

**Live tab only**
| Slider | Default | Effect |
|---|---|---|
| Motion weight | 0.35 | How much micro-motion contributes vs spoof models |

---

## 🔬 Micro-Motion Liveness (`liveness_temporal.py`)

Detects whether a face is real by analysing **involuntary micro-movements** across frames:

```
Real face  →  small random nose jitter  →  variance > threshold  →  LIVE
Photo      →  zero movement             →  variance ≈ 0          →  SPOOF
Replay     →  mechanical/periodic motion →  FFT dominant freq high → SPOOF
```

**Three sub-scores combined:**

| Sub-score              | Weight | What it checks                        |
| ---------------------- | ------ | ------------------------------------- |
| Displacement variance  | 50%    | Is there any natural movement?        |
| Naturalness            | 30%    | Fraction of frames with movement      |
| Anti-periodicity (FFT) | 20%    | Catches replay attacks (phone wobble) |

All motion is **normalised by inter-eye distance** — works regardless of face distance from camera.

### API

```python
from liveness_temporal import TemporalLivenessChecker, fuse_with_spoof_score

checker = TemporalLivenessChecker(min_frames=15)

# Feed landmarks per frame (RetinaFace 5-point, shape (5,2))
for landmarks in all_landmarks:
    checker.add_frame(landmarks)

# Evaluate once ready
if checker.ready:
    result = checker.evaluate()
    print(result.is_live)    # bool
    print(result.score)      # 0.0 → 1.0
    print(result.reason)     # detailed breakdown string

# Fuse with spoof ensemble score
is_live, fused_score, reason = fuse_with_spoof_score(
    spoof_score=0.85,         # from your ensemble (0=real, 1=spoof)
    temporal_result=result,
    spoof_weight=0.65,
    temporal_weight=0.35,
)
```

---

## 📋 Requirements

```
torch
torchvision
opencv-python-headless
onnxruntime
omegaconf
easydict
gradio
numpy
scipy
```

---

## 🔗 References

- **SASF** — [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **FLRGB** — [Face Liveness Detection Model-RGB, ModelScope IIC](https://modelscope.cn/models/iic/cv_manual_face-liveness_flrgb)
- **IADG (ICM2O / IOM2C)** — [Instance-Aware Domain Generalisation for Face Anti-Spoofing, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf)
- **RetinaFace** — Face detector providing 5-point landmarks

---

## 🗺️ Roadmap

- [ ] rPPG remote pulse detection (heartbeat from skin color)
- [ ] Meta-learner for learned ensemble fusion weights
- [ ] Fine-tune on domain-specific data (office lighting, phone cameras)
- [ ] Knowledge distillation — single fast model from ensemble
- [ ] Multi-face support

---

## 👤 Author

**Mothieram** — [HuggingFace](https://huggingface.co/mothieram) · [GitHub](https://github.com/Mothieram)
