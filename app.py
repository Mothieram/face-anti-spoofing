import logging
import os
import time
import sys
import traceback
import cv2
import numpy as np
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "face-antispoofing")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import IADG
import SASF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_LOAD_ERROR = None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
try:
    ModelD  = IADG.aFaceDetect()
    Model1  = SASF.aSASF(threshold=0.0094)
    Model2  = IADG.aSpoofONNX('modelrgb', threshold=0.0553)
    Model3  = IADG.aSpoof('ICM2O', threshold=0.9980)
    Model4  = IADG.aSpoof('IOM2C', threshold=0.9944)
    MODELS_OK = True
except Exception as e:
    MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
    logger.error("Error loading models:\n%s", traceback.format_exc())
    ModelD = Model1 = Model2 = Model3 = Model4 = None
    MODELS_OK = False

# ---------------------------------------------------------------------------
# Weights for ensemble fusion
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {
    'sasf':  0.25,
    'flrgb': 0.25,
    'icm2o': 0.25,
    'iom2c': 0.25,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_label_index(value):
    """Normalise numpy/tensor/scalar booleans → plain Python int index."""
    try:
        idx = int(np.asarray(value).reshape(-1)[0])
    except Exception:
        idx = int(bool(value))
    return 1 if idx else 0


def _confidence(p, threshold):
    """Distance from threshold, normalised to [0, 1]."""
    if p < threshold:
        return (threshold - p) / threshold
    return (p - threshold) / (1 - threshold)


def _draw_result_on_image(image_rgb, bbox, label: str, color):
    """Draw bounding box + label on a copy of the image."""
    img = image_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font_scale = max(0.6, (x2 - x1) / 300)
    cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return img


def _run_single_model(model, image, bbox, landmark):
    """Run one model and return (spoof_label, spoof_prob, face_crop)."""
    spoof, prob, crop = model(image, bbox, landmark)
    return _to_label_index(spoof), float(prob), crop


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def run_image(input_image,
              thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c):
    """
    Returns: annotated_image, result_text
    """
    if input_image is None or not hasattr(input_image, "shape"):
        return None, "Please upload or capture an image first."

    if not MODELS_OK:
        return None, (
            "⚠️ Models failed to load. "
            "Check that `face-antispoofing/weights` exists and all dependencies are installed.\n"
            f"Startup error: {MODEL_LOAD_ERROR or 'unknown'}"
        )

    # Apply slider thresholds
    Model1.threshold = thr_sasf
    Model2.threshold = thr_flrgb
    Model3.threshold = thr_icm2o
    Model4.threshold = thr_iom2c

    # ── Face detection ──────────────────────────────────────────────────────
    bboxes, landmarks = ModelD(input_image)
    if len(landmarks) < 1:
        return input_image, "⚠️ No face detected in the image."

    # Use the first (largest) face only
    bbox, landmark = bboxes[0], landmarks[0]

    # ── Parallel inference ──────────────────────────────────────────────────
    tasks = {
        'sasf':  (Model1, input_image, bbox, landmark),
        'flrgb': (Model2, input_image, bbox, landmark),
        'icm2o': (Model3, input_image, bbox, landmark),
        'iom2c': (Model4, input_image, bbox, landmark),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_single_model, *args): key
            for key, args in tasks.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.error("Model %s failed: %s", key, exc)
                results[key] = None

    # ── Gather results ──────────────────────────────────────────────────────
    model_info = {
        'sasf':  {'label': 'SASF',  'threshold': thr_sasf},
        'flrgb': {'label': 'FLRGB', 'threshold': thr_flrgb},
        'icm2o': {'label': 'ICM2O', 'threshold': thr_icm2o},
        'iom2c': {'label': 'IOM2C', 'threshold': thr_iom2c},
    }
    names = ['Real', 'Spoof']
    active_weights = {}
    lines = []

    for key in ('sasf', 'flrgb', 'icm2o', 'iom2c'):
        res = results.get(key)
        info = model_info[key]
        if res is None:
            lines.append(f"{info['label']}:\t ❌ failed")
            continue
        spoof_label, spoof_prob, _ = res
        conf = _confidence(spoof_prob, info['threshold'])
        lines.append(
            f"{info['label']}:\t P={spoof_prob:.4f}  →  {names[spoof_label]}"
            f"  (conf: {conf:.2%})"
        )
        active_weights[key] = ENSEMBLE_WEIGHTS[key]

    # ── Weighted ensemble verdict ───────────────────────────────────────────
    if active_weights:
        total_w = sum(active_weights.values())
        norm_w  = {k: v / total_w for k, v in active_weights.items()}
        ensemble_score = sum(
            norm_w[k] * results[k][1]
            for k in active_weights
        )
        # Ensemble threshold: weighted average of individual thresholds
        ensemble_threshold = sum(
            norm_w[k] * model_info[k]['threshold']
            for k in active_weights
        )
        is_spoof         = ensemble_score >= ensemble_threshold
        ensemble_verdict = "🔴 SPOOF" if is_spoof else "🟢 REAL"
        ensemble_conf    = _confidence(ensemble_score, ensemble_threshold)
        lines.append("")
        lines.append("─" * 42)
        lines.append(f"Ensemble score : {ensemble_score:.4f}  (threshold: {ensemble_threshold:.4f})")
        lines.append(f"Final verdict  : {ensemble_verdict}  (conf: {ensemble_conf:.2%})")
    else:
        is_spoof         = False
        ensemble_verdict = "❓ Unknown"
        lines.append("\nAll models failed — no verdict.")

    # ── Annotated output image ──────────────────────────────────────────────
    color = (220, 50, 50) if is_spoof else (50, 200, 50)   # red / green
    annotated = _draw_result_on_image(input_image, bbox, ensemble_verdict, color)

    return annotated, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def demo():
    with gr.Blocks(title="Face Anti-Spoofing Detector", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
# 🛡️ Face Anti-Spoofing Detector
**4-model ensemble**: SASF · FLRGB · ICM2O · IOM2C  
Models run in **parallel**; results are fused via weighted average (ICM2O/IOM2C 35% · SASF/FLRGB 15%).
""")

        with gr.Row():
            # ── Left column: inputs ────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type='numpy',
                    sources=['upload', 'webcam'],
                    label='Input — upload or capture a face photo'
                )

                gr.Markdown("### ⚙️ Thresholds")
                thr_sasf  = gr.Slider(0.0, 1.0, value=0.0094, step=0.0001, label="SASF threshold")
                thr_flrgb = gr.Slider(0.0, 1.0, value=0.2808, step=0.0001, label="FLRGB threshold")
                thr_icm2o = gr.Slider(0.0, 1.0, value=0.9980, step=0.0001, label="ICM2O threshold")
                thr_iom2c = gr.Slider(0.0, 1.0, value=0.9944, step=0.0001, label="IOM2C threshold")

                with gr.Row():
                    run_btn   = gr.Button("▶ Run", variant="primary")
                    clear_btn = gr.Button("🗑 Clear")

            # ── Right column: outputs ──────────────────────────────────────
            with gr.Column(scale=1):
                annotated_out = gr.Image(type='numpy', label='Result — annotated image')
                result_text   = gr.TextArea(
                    label='Per-model scores + ensemble verdict',
                    lines=10,
                    value=''
                )

        # ── Model reference ────────────────────────────────────────────────
        with gr.Accordion("📚 Model references", open=False):
            gr.Markdown("""
- **SASF** — Silent-Face-Anti-Spoofing · [github.com/minivision-ai](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **FLRGB** — Face Liveness Detection Model-RGB · [ModelScope IIC](https://modelscope.cn/models/iic/cv_manual_face-liveness_flrgb)
- **ICM2O / IOM2C** — Instance-Aware Domain Generalisation (CVPR 2023) · [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf)

**Interpreting results:** P above threshold → spoof.  
The further P is from the threshold, the higher the confidence.
""")

        # ── Event wiring ───────────────────────────────────────────────────
        inputs  = [image_input, thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c]
        outputs = [annotated_out, result_text]

        run_btn.click(fn=run_image, inputs=inputs, outputs=outputs)
        clear_btn.click(
            fn=lambda: [None, None, ''],
            inputs=None,
            outputs=[image_input, annotated_out, result_text]
        )

    app.launch()


if __name__ == '__main__':
    demo()