# Fine-Tuning Pipeline for face-anti-spoofing
## Using: Kaggle anti-spoofing-live (real) + LCC_FASD (spoof)

---

## Folder Structure

Place these files inside your repo root:

```
face-anti-spoofing/
├── train_pipeline/
│   ├── prepare_dataset.py        ← Step 1: crop & organize data
│   ├── finetune_iadg.py          ← Step 2a: fine-tune ICM2O / IOM2C
│   ├── finetune_sasf.py          ← Step 2b: fine-tune SASF models
│   ├── recalibrate_thresholds.py ← Step 3: find new thresholds
│   ├── liveness_finetuned.py     ← Step 4: drop-in replacement liveness.py
│   └── data/
│       ├── train/real/   ← populated by prepare_dataset.py
│       ├── train/spoof/
│       ├── val/real/
│       └── val/spoof/
├── finetuned_weights/            ← created automatically
│   ├── ICM2O_finetuned.pth
│   ├── IOM2C_finetuned.pth
│   └── 2.7_80x80_MiniFASNetV2_finetuned.pth
├── IADG.py
├── SASF.py
├── models.py
└── weights/
```

---

## Step 0 – Download datasets

### Real (Kaggle anti-spoofing-live)
```bash
pip install kaggle
kaggle datasets download -d trainingdatapro/anti-spoofing-live
unzip anti-spoofing-live.zip -d train_pipeline/data/raw/anti-spoofing-live
```

### Spoof (LCC_FASD — free, publicly available)
```bash
# Download from: https://csit.am/datasets/lcc_fasd/
# Or Kaggle: https://www.kaggle.com/datasets/faber24/lcc-fasd
kaggle datasets download -d faber24/lcc-fasd
unzip lcc-fasd.zip -d train_pipeline/data/raw/LCC_FASD
```

LCC_FASD structure expected:
```
LCC_FASD/
  train/
    real/   (bonafide selfies)
    spoof/  (print + replay attacks)
  val/
    real/
    spoof/
```

---

## Step 1 – Prepare dataset (crop faces)

```bash
cd train_pipeline
python prepare_dataset.py
```

This will:
- Detect and crop faces from all selfie photos in the Kaggle real dataset
- Detect and crop faces from all spoof images in LCC_FASD
- Split 85% train / 15% val
- Output to `data/train/` and `data/val/`

Expected output:
```
data/train/real:   ~8500 images
data/train/spoof:  ~8500 images
data/val/real:     ~1500 images
data/val/spoof:    ~1500 images
```

---

## Step 2a – Fine-tune ICM2O and IOM2C

```bash
python finetune_iadg.py --model ICM2O --epochs_head 5 --epochs_full 10
python finetune_iadg.py --model IOM2C --epochs_head 5 --epochs_full 10
```

Training phases:
- **Phase 1 (5 epochs):** Only the FeatEmbedder.fc layer trains (LR=1e-4)
- **Phase 2 (10 epochs):** All layers unfreeze (LR=1e-5)

Output: `finetuned_weights/ICM2O_finetuned.pth`, `IOM2C_finetuned.pth`

Metric to watch: **ACER** (lower is better, 0 = perfect)

Typical results after fine-tuning on domain data:
- Original ACER on lab dataset: ~5–15%
- After fine-tuning on your camera/environment: ~1–5%

---

## Step 2b – Fine-tune SASF

```bash
python finetune_sasf.py
```

Fine-tunes both:
- `2.7_80x80_MiniFASNetV2.pth`
- `4_0_0_80x80_MiniFASNetV1SE.pth`

Output: `finetuned_weights/*_finetuned.pth`

---

## Step 3 – Recalibrate thresholds

After fine-tuning the model scores shift. Run this to find new optimal thresholds:

```bash
python recalibrate_thresholds.py
```

Output example:
```
--- Calibrating ICM2O ---
  Budget    APCER     BPCER    Threshold
      10%    3.21%    9.98%    0.987600
      20%    1.45%   20.01%    0.994200
      30%    0.88%   29.77%    0.997100
```

Choose the budget that fits your attendance use case:
- **10% budget** = only 10% of real faces incorrectly rejected → stricter anti-spoof
- **30% budget** = easier to pass → fewer false rejections

Update thresholds in your `liveness.py` accordingly.

---

## Step 4 – Use fine-tuned models

Copy `liveness_finetuned.py` to your repo root as the new `liveness.py`:

```bash
cp liveness_finetuned.py ../liveness.py
```

The file auto-detects fine-tuned weights and loads them.
Falls back to original weights if fine-tuned weights are not found.

In your FastAPI `api.py`, usage is unchanged:
```python
from liveness import predict
is_spoof, score, per_model = predict(image_rgb, bbox, landmarks)
```

---

## GPU vs CPU

- **With GPU (CUDA):** Each epoch ~30–60 seconds for ~17,000 images
- **CPU only:** ~5–10 minutes per epoch — consider reducing to 5 full epochs

Check GPU availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No faces detected` in prepare_dataset.py | Kaggle selfies may have low resolution — reduce `minSize` in Haar cascade to (40,40) |
| `KeyError: args` when loading checkpoint | Your .pth.tar uses OmegaConf — already handled by `_load_checkpoint()` in IADG.py |
| ACER not improving after phase 2 | Reduce `LR_FULL` to 1e-6, or skip phase 2 entirely |
| Class imbalance warning | WeightedRandomSampler handles this — ignore |
| `CUDA out of memory` | Reduce `BATCH_SIZE` from 16 to 8 in finetune_iadg.py |
