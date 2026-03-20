"""
app.py  —  Face Anti-Spoofing Detector  (Gradio)
─────────────────────────────────────────────────
Two tabs:
  1. Image tab   — upload or single webcam snapshot → 4-model ensemble verdict
  2. Live tab    — streaming webcam → micro-motion check + spoof ensemble fused
"""

import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import gradio as gr

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "face-antispoofing")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import IADG
import SASF
from liveness_temporal import TemporalLivenessChecker, fuse_with_spoof_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_LOAD_ERROR = None
try:
    ModelD = IADG.aFaceDetect()
    Model1 = SASF.aSASF(threshold=0.0094)
    Model2 = IADG.aSpoofONNX('modelrgb', threshold=0.0553)
    Model3 = IADG.aSpoof('ICM2O',  threshold=0.9980)
    Model4 = IADG.aSpoof('IOM2C',  threshold=0.9944)
    MODELS_OK = True
except Exception as e:
    MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
    logger.error("Error loading models:\n%s", traceback.format_exc())
    ModelD = Model1 = Model2 = Model3 = Model4 = None
    MODELS_OK = False

# ── Ensemble weights ──────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {'sasf': 0.15, 'flrgb': 0.15, 'icm2o': 0.35, 'iom2c': 0.35}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_label_index(value):
    try:
        idx = int(np.asarray(value).reshape(-1)[0])
    except Exception:
        idx = int(bool(value))
    return 1 if idx else 0


def _confidence(p, threshold):
    if p < threshold:
        return (threshold - p) / threshold
    return (p - threshold) / (1 - threshold)


def _run_single_model(model, image, bbox, landmark):
    spoof, prob, crop = model(image, bbox, landmark)
    return _to_label_index(spoof), float(prob), crop


def _draw_on_image(image_rgb, bbox, label, color):
    img = image_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font_scale = max(0.6, (x2 - x1) / 300)
    cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return img


def _run_ensemble(image_rgb, bbox, landmark,
                  thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c,
                  w_sasf=0.15, w_flrgb=0.15, w_icm2o=0.35, w_iom2c=0.35):
    """
    Run 4 models in parallel with dynamic weights (raw, auto-normalised inside).
    Returns (ensemble_spoof_score, result_lines, is_spoof)
    """
    Model1.threshold = thr_sasf
    Model2.threshold = thr_flrgb
    Model3.threshold = thr_icm2o
    Model4.threshold = thr_iom2c

    tasks = {
        'sasf':  (Model1, image_rgb, bbox, landmark),
        'flrgb': (Model2, image_rgb, bbox, landmark),
        'icm2o': (Model3, image_rgb, bbox, landmark),
        'iom2c': (Model4, image_rgb, bbox, landmark),
    }
    model_thresholds  = {'sasf': thr_sasf, 'flrgb': thr_flrgb, 'icm2o': thr_icm2o, 'iom2c': thr_iom2c}
    model_raw_weights = {'sasf': w_sasf,   'flrgb': w_flrgb,   'icm2o': w_icm2o,   'iom2c': w_iom2c}
    model_labels      = {'sasf': 'SASF', 'flrgb': 'FLRGB', 'icm2o': 'ICM2O', 'iom2c': 'IOM2C'}
    names = ['Real', 'Spoof']
    total_raw = sum(model_raw_weights.values()) or 1.0

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_single_model, *args): key
                   for key, args in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.error("Model %s failed: %s", key, exc)
                results[key] = None

    active_weights = {}
    lines = []
    for key in ('sasf', 'flrgb', 'icm2o', 'iom2c'):
        res  = results.get(key)
        thr  = model_thresholds[key]
        lbl  = model_labels[key]
        norm_pct = model_raw_weights[key] / total_raw
        if res is None:
            lines.append(f"{lbl} ({norm_pct:.0%}):\t ❌ failed")
            continue
        spoof_label, spoof_prob, _ = res
        conf = _confidence(spoof_prob, thr)
        lines.append(
            f"{lbl} ({norm_pct:.0%}):\t P={spoof_prob:.4f}  →  {names[spoof_label]}"
            f"  (conf: {conf:.2%})"
        )
        active_weights[key] = model_raw_weights[key]

    if not active_weights:
        return None, lines, False

    total_w              = sum(active_weights.values()) or 1.0
    norm_w               = {k: v / total_w for k, v in active_weights.items()}
    ensemble_spoof_score = sum(norm_w[k] * results[k][1] for k in active_weights)
    ensemble_threshold   = sum(norm_w[k] * model_thresholds[k] for k in active_weights)
    is_spoof             = ensemble_spoof_score >= ensemble_threshold
    conf                 = _confidence(ensemble_spoof_score, ensemble_threshold)

    lines += [
        "",
        "─" * 42,
        f"Ensemble score : {ensemble_spoof_score:.4f}  (threshold: {ensemble_threshold:.4f})",
        f"Spoof verdict  : {'🔴 SPOOF' if is_spoof else '🟢 REAL'}  (conf: {conf:.2%})",
    ]
    return ensemble_spoof_score, lines, is_spoof


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single image
# ─────────────────────────────────────────────────────────────────────────────

def run_image(input_image, thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c,
              w_sasf, w_flrgb, w_icm2o, w_iom2c):
    if input_image is None or not hasattr(input_image, "shape"):
        return None, "Please upload or capture an image first."
    if not MODELS_OK:
        return None, f"⚠️ Models failed to load.\n{MODEL_LOAD_ERROR or 'unknown'}"

    bboxes, landmarks = ModelD(input_image)
    if len(landmarks) < 1:
        return input_image, "⚠️ No face detected."

    bbox, landmark        = bboxes[0], landmarks[0]
    ensemble_score, lines, is_spoof = _run_ensemble(
        input_image, bbox, landmark,
        thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c,
        w_sasf, w_flrgb, w_icm2o, w_iom2c
    )
    if ensemble_score is None:
        return input_image, "\n".join(lines)

    color     = (220, 50, 50) if is_spoof else (50, 200, 50)
    verdict   = "🔴 SPOOF" if is_spoof else "🟢 REAL"
    annotated = _draw_on_image(input_image, bbox, verdict, color)
    return annotated, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Live webcam streaming
# ─────────────────────────────────────────────────────────────────────────────

def process_live_frame(
    frame,
    checker: TemporalLivenessChecker,
    frozen_frame,           # last annotated result frame (or None)
    verdict_done: bool,     # True = verdict already shown, stop processing
    thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c,
    w_sasf, w_flrgb, w_icm2o, w_iom2c,
    temporal_weight,
):
    """Called on every streaming webcam frame via gr.Image.stream()."""

    # ── If verdict already reached: freeze — return stored result unchanged ───
    if verdict_done and frozen_frame is not None:
        return frozen_frame, gr.update(), verdict_done, frozen_frame

    if frame is None or not hasattr(frame, "shape"):
        return None, "No frame received.", False, None

    if not MODELS_OK:
        return frame, f"⚠️ Models failed to load.\n{MODEL_LOAD_ERROR}", False, None

    # ── Face detection ────────────────────────────────────────────────────────
    bboxes, landmarks = ModelD(frame)
    if len(landmarks) < 1:
        return frame, "⚠️ No face detected — centre your face in the frame.", False, None

    bbox, landmark = bboxes[0], landmarks[0]

    # ── Accumulate landmark into temporal checker ─────────────────────────────
    checker.add_frame(landmark)

    # ── While collecting: show live progress overlay ──────────────────────────
    if not checker.ready:
        n, mn = checker.frame_count, checker.min_frames
        progress_frame = _draw_on_image(
            frame, bbox, f"Collecting… {n}/{mn}", color=(180, 180, 0)
        )
        return (
            progress_frame,
            f"⏳ Collecting frames: {n}/{mn}\nHold still and look at the camera.",
            False,   # verdict_done
            None,    # frozen_frame
        )

    # ── Enough frames — run spoof ensemble ───────────────────────────────────
    ensemble_score, lines, is_spoof = _run_ensemble(
        frame, bbox, landmark,
        thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c,
        w_sasf, w_flrgb, w_icm2o, w_iom2c
    )
    if ensemble_score is None:
        return frame, "\n".join(lines), False, None

    # ── Fuse ensemble score with micro-motion score ───────────────────────────
    temporal_result = checker.evaluate()
    spoof_w         = 1.0 - float(temporal_weight)
    is_live, fused_score, fused_reason = fuse_with_spoof_score(
        spoof_score     = ensemble_score,
        temporal_result = temporal_result,
        spoof_weight    = spoof_w,
        temporal_weight = float(temporal_weight),
    )

    # ── Annotate result frame ─────────────────────────────────────────────────
    color     = (220, 50, 50) if not is_live else (50, 200, 50)
    verdict   = "🟢 LIVE / REAL" if is_live else "🔴 SPOOF"
    annotated = _draw_on_image(frame, bbox, verdict, color)

    status = "\n".join(lines) + "\n\n── Micro-motion ──\n" + fused_reason
    status += "\n\n✅ Verdict locked. Press 🔄 Reset to scan again."

    checker.reset()

    # verdict_done=True freezes all future frames
    return annotated, status, True, annotated


def reset_checker(checker: TemporalLivenessChecker):
    checker.reset()
    return checker, False, None, "🔄 Session reset — collecting new frames…"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    with gr.Blocks(title="Face Anti-Spoofing Detector", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
# 🛡️ Face Anti-Spoofing Detector
**4-model ensemble** (SASF · FLRGB · ICM2O · IOM2C) + **micro-motion liveness**
""")

        # ── Shared thresholds ─────────────────────────────────────────────────
        with gr.Accordion("⚙️ Model Thresholds", open=False):
            with gr.Row():
                thr_sasf  = gr.Slider(0.0, 1.0, value=0.0094, step=0.0001, label="SASF threshold")
                thr_flrgb = gr.Slider(0.0, 1.0, value=0.2808, step=0.0001, label="FLRGB threshold")
                thr_icm2o = gr.Slider(0.0, 1.0, value=0.9980, step=0.0001, label="ICM2O threshold")
                thr_iom2c = gr.Slider(0.0, 1.0, value=0.9944, step=0.0001, label="IOM2C threshold")

        thresholds = [thr_sasf, thr_flrgb, thr_icm2o, thr_iom2c]

        # ── Shared model weights ───────────────────────────────────────────────
        with gr.Accordion("⚖️ Model Weights  (auto-normalised to sum = 1.0)", open=True):
            gr.Markdown("Adjust how much each model contributes to the final ensemble verdict.")
            with gr.Row():
                w_sasf  = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="SASF weight")
                w_flrgb = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="FLRGB weight")
                w_icm2o = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="ICM2O weight")
                w_iom2c = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="IOM2C weight")
            weight_display = gr.Markdown(
                "_SASF 15% · FLRGB 15% · ICM2O 35% · IOM2C 35%  —  sum = 1.00_"
            )

        weights = [w_sasf, w_flrgb, w_icm2o, w_iom2c]

        def _update_weight_display(ws, wf, wi1, wi2):
            total = ws + wf + wi1 + wi2
            if total == 0:
                return "_All weights are zero — please set at least one above 0._"
            ns, nf, ni1, ni2 = ws/total, wf/total, wi1/total, wi2/total
            return (
                f"_SASF **{ns:.0%}** · FLRGB **{nf:.0%}** · "
                f"ICM2O **{ni1:.0%}** · IOM2C **{ni2:.0%}**  —  sum = 1.00_"
            )

        for w in weights:
            w.change(
                fn=_update_weight_display,
                inputs=weights,
                outputs=weight_display
            )

        with gr.Tabs():

            # ── Tab 1: Image ──────────────────────────────────────────────────
            with gr.TabItem("📷 Image"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(
                            type='numpy',
                            sources=['upload', 'webcam'],
                            label='Upload or capture a face photo'
                        )
                        with gr.Row():
                            img_run_btn   = gr.Button("▶ Run", variant="primary")
                            img_clear_btn = gr.Button("🗑 Clear")
                    with gr.Column():
                        img_output = gr.Image(type='numpy', label='Result')
                        img_text   = gr.TextArea(
                            label='Per-model scores + verdict', lines=10
                        )

                img_run_btn.click(
                    fn=run_image,
                    inputs=[img_input] + thresholds + weights,
                    outputs=[img_output, img_text]
                )
                img_clear_btn.click(
                    fn=lambda: [None, None, ''],
                    inputs=None,
                    outputs=[img_input, img_output, img_text]
                )

            # ── Tab 2: Live webcam ────────────────────────────────────────────
            with gr.TabItem("🎥 Live"):
                gr.Markdown("""
Streams webcam frames, accumulates ~15 frames of micro-motion,
then fuses motion score with the 4-model ensemble for a combined verdict.
**Result is locked on screen — press 🔄 Reset to scan again.**
""")
                checker_state      = gr.State(TemporalLivenessChecker())
                verdict_done_state = gr.State(False)   # True  → freeze output
                frozen_frame_state = gr.State(None)    # holds last result frame

                with gr.Row():
                    with gr.Column():
                        live_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Webcam feed"
                        )
                        temporal_w = gr.Slider(
                            0.0, 1.0, value=0.35, step=0.05,
                            label="Motion weight  (remainder → spoof model)"
                        )
                        reset_btn = gr.Button("🔄 Reset session", variant="primary")

                    with gr.Column():
                        live_output = gr.Image(type='numpy', label='Result (locked after verdict)')
                        live_text   = gr.TextArea(
                            label='Live scores + fused verdict',
                            lines=14,
                            value="Waiting for webcam stream…"
                        )

                live_input.stream(
                    fn=process_live_frame,
                    inputs=[
                        live_input, checker_state, frozen_frame_state,
                        verdict_done_state,
                    ] + thresholds + weights + [temporal_w],
                    outputs=[live_output, live_text, verdict_done_state, frozen_frame_state],
                )
                reset_btn.click(
                    fn=reset_checker,
                    inputs=[checker_state],
                    outputs=[checker_state, verdict_done_state, frozen_frame_state, live_text]
                )

        # ── References ────────────────────────────────────────────────────────
        with gr.Accordion("📚 Model references", open=False):
            gr.Markdown("""
- **SASF** — Silent-Face-Anti-Spoofing · [github](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **FLRGB** — Face Liveness Detection RGB · [ModelScope](https://modelscope.cn/models/iic/cv_manual_face-liveness_flrgb)
- **ICM2O / IOM2C** — Instance-Aware Domain Generalisation CVPR 2023 · [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf)
- **Micro-motion** — Nose-tip displacement variance + FFT periodicity (anti-replay)
""")

    app.launch()


if __name__ == '__main__':
    demo()