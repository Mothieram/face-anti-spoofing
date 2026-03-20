"""
liveness_temporal.py
────────────────────
Temporal liveness check using facial micro-motion analysis.
Works with RetinaFace 5-point landmarks — no extra dependencies needed.

RetinaFace landmark order:
    0 → left eye
    1 → right eye
    2 → nose tip
    3 → left mouth corner
    4 → right mouth corner

Decision logic:
    - Track nose tip (x, y) across N frames
    - Compute per-frame displacement from previous frame
    - Compute variance of displacement
    - Real face   → small random jitter   → variance > MOTION_VAR_MIN
    - Photo still → near-zero variance    → flagged as spoof
    - Replay (phone wobble) → high periodic variance → caught by FFT periodicity check

Usage (FastAPI):
    checker = TemporalLivenessChecker()
    for landmarks in all_landmarks:         # landmarks per frame from RetinaFace
        checker.add_frame(landmarks)
    result = checker.evaluate()
    print(result.is_live, result.score, result.reason)

Usage (Gradio webcam):
    checker = TemporalLivenessChecker()
    # call checker.add_frame(landmarks) each time a new webcam frame arrives
    # call checker.evaluate() once enough frames are collected
    # call checker.reset() to start a new session
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ─── Tuneable constants ────────────────────────────────────────────────────────

MIN_FRAMES        = 15       # minimum frames before giving a verdict (~0.5s at 30fps)
MAX_FRAMES        = 60       # buffer cap (2s at 30fps) — oldest frames dropped

# Displacement variance thresholds (normalised by inter-eye distance)
MOTION_VAR_MIN    = 0.00008  # below → too still → spoof
MOTION_VAR_MAX    = 0.015    # above → excessive shaking → reject

# Periodicity: dominant FFT frequency energy / total energy
# If > threshold → mechanical/periodic motion → likely replay attack
PERIODICITY_RATIO = 0.60

# Sub-score weights
W_VARIANCE        = 0.50
W_NATURALNESS     = 0.30
W_ANTI_PERIODIC   = 0.20


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TemporalResult:
    is_live           : bool
    score             : float        # 0.0 (spoof) → 1.0 (live)
    reason            : str
    frame_count       : int
    variance          : float
    mean_displacement : float
    periodicity_ratio : float
    sub_scores        : dict = field(default_factory=dict)


# ─── Main class ───────────────────────────────────────────────────────────────

class TemporalLivenessChecker:
    """
    Accumulates landmark data across frames and evaluates micro-motion liveness.

    Parameters
    ----------
    min_frames : int
        Minimum frames required before evaluate() gives a real verdict.
    max_frames : int
        Maximum buffer size — oldest frames are dropped when exceeded.
    """

    def __init__(
        self,
        min_frames: int = MIN_FRAMES,
        max_frames: int = MAX_FRAMES,
    ):
        self.min_frames      = min_frames
        self.max_frames      = max_frames
        self._nose_positions : List[Tuple[float, float]] = []
        self._face_widths    : List[float] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self):
        """Clear all accumulated frames. Call before a new session."""
        self._nose_positions = []
        self._face_widths    = []

    @property
    def frame_count(self) -> int:
        return len(self._nose_positions)

    @property
    def ready(self) -> bool:
        """True once enough frames have been collected."""
        return self.frame_count >= self.min_frames

    def add_frame(self, landmarks: np.ndarray) -> None:
        """
        Add one frame's landmark data to the buffer.

        Parameters
        ----------
        landmarks : array-like, shape (5, 2) or (10,)
            RetinaFace 5-point landmarks in pixel coordinates.
            Order: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        lm = np.asarray(landmarks, dtype=float).reshape(5, 2)

        nose        = lm[2]                                             # (x, y)
        inter_eye_d = float(np.linalg.norm(lm[1] - lm[0])) + 1e-6    # face scale

        self._nose_positions.append((float(nose[0]), float(nose[1])))
        self._face_widths.append(inter_eye_d)

        # Drop oldest frame if buffer is full
        if len(self._nose_positions) > self.max_frames:
            self._nose_positions.pop(0)
            self._face_widths.pop(0)

    def evaluate(self) -> TemporalResult:
        """
        Evaluate liveness from the accumulated frames.

        Returns TemporalResult even with insufficient frames —
        is_live=False and reason explains it.
        """
        n = self.frame_count

        if n < self.min_frames:
            return TemporalResult(
                is_live           = False,
                score             = 0.0,
                reason            = (
                    f"⏳ Not enough frames yet ({n}/{self.min_frames}). "
                    "Hold still and look at the camera."
                ),
                frame_count       = n,
                variance          = 0.0,
                mean_displacement = 0.0,
                periodicity_ratio = 0.0,
            )

        positions   = np.array(self._nose_positions)   # (N, 2)
        face_widths = np.array(self._face_widths)       # (N,)

        # Normalise nose positions by inter-eye distance → scale invariant
        norm_pos = positions / face_widths[:, None]

        # Per-frame displacement from previous frame
        displacements = np.linalg.norm(
            np.diff(norm_pos, axis=0), axis=1
        )  # shape (N-1,)

        variance          = float(np.var(displacements))
        mean_displacement = float(np.mean(displacements))

        # ── Sub-score 1: Variance ───────────────────────────────────────────
        if variance < MOTION_VAR_MIN:
            var_score = 0.0      # too still → spoof
        elif variance > MOTION_VAR_MAX:
            var_score = 0.2      # excessive shaking → penalise
        else:
            var_score = float(np.clip(
                (variance - MOTION_VAR_MIN) / (MOTION_VAR_MAX - MOTION_VAR_MIN),
                0.0, 1.0
            ))

        # ── Sub-score 2: Naturalness (fraction of frames with any movement) ─
        nonzero_ratio = float(np.mean(displacements > 1e-5))
        natural_score = nonzero_ratio

        # ── Sub-score 3: Anti-periodicity (catch replay attacks via FFT) ────
        periodicity_ratio = _compute_periodicity(displacements)
        anti_periodic_score = (
            float(1.0 - periodicity_ratio)
            if periodicity_ratio > PERIODICITY_RATIO
            else 1.0
        )

        # ── Weighted final score ─────────────────────────────────────────────
        score = float(np.clip(
            W_VARIANCE      * var_score          +
            W_NATURALNESS   * natural_score      +
            W_ANTI_PERIODIC * anti_periodic_score,
            0.0, 1.0
        ))

        is_live = score >= 0.5

        reason = _build_reason(
            is_live, score, variance, mean_displacement,
            periodicity_ratio, var_score, natural_score, anti_periodic_score,
        )

        return TemporalResult(
            is_live           = is_live,
            score             = score,
            reason            = reason,
            frame_count       = n,
            variance          = variance,
            mean_displacement = mean_displacement,
            periodicity_ratio = periodicity_ratio,
            sub_scores        = {
                "variance_score"      : round(var_score, 4),
                "naturalness_score"   : round(natural_score, 4),
                "anti_periodic_score" : round(anti_periodic_score, 4),
            },
        )


# ─── Private helpers ──────────────────────────────────────────────────────────

def _compute_periodicity(displacements: np.ndarray) -> float:
    """
    Ratio of dominant FFT frequency energy to total energy.
    High value → periodic / mechanical motion → replay attack indicator.
    """
    if len(displacements) < 8:
        return 0.0
    centred      = displacements - displacements.mean()
    fft_mag      = np.abs(np.fft.rfft(centred))
    total_energy = float(np.sum(fft_mag ** 2)) + 1e-9
    dom_energy   = float(np.max(fft_mag) ** 2)
    return float(np.clip(dom_energy / total_energy, 0.0, 1.0))


def _build_reason(
    is_live, score, variance, mean_disp,
    periodicity, var_score, natural_score, anti_periodic_score,
) -> str:
    verdict = "✅ LIVE" if is_live else "❌ SPOOF"
    lines = [
        f"{verdict}  (motion score: {score:.2%})",
        f"  displacement variance : {variance:.6f}  →  score: {var_score:.2%}",
        f"  mean displacement     : {mean_disp:.6f}",
        f"  naturalness           : {natural_score:.2%}",
        f"  periodicity ratio     : {periodicity:.2%}  →  score: {anti_periodic_score:.2%}",
    ]
    if var_score == 0.0:
        lines.append("  ⚠ Motion too low — possible static photo attack")
    if periodicity > PERIODICITY_RATIO:
        lines.append("  ⚠ Periodic motion detected — possible replay attack")
    return "\n".join(lines)


# ─── Fusion helper ────────────────────────────────────────────────────────────

def fuse_with_spoof_score(
    spoof_score     : float,
    temporal_result : TemporalResult,
    spoof_weight    : float = 0.65,
    temporal_weight : float = 0.35,
) -> Tuple[bool, float, str]:
    """
    Fuse your existing 4-model ensemble spoof score with the temporal motion score.

    Parameters
    ----------
    spoof_score      : float — ensemble output (0.0 = real, 1.0 = spoof)
    temporal_result  : TemporalResult from TemporalLivenessChecker.evaluate()
    spoof_weight     : weight for the spoof model (default 65%)
    temporal_weight  : weight for temporal motion  (default 35%)

    Returns
    -------
    (is_live, fused_score, reason_str)
        fused_score : 0.0 = definitely spoof, 1.0 = definitely live
    """
    spoof_liveness    = 1.0 - float(spoof_score)
    temporal_liveness = temporal_result.score

    fused   = float(np.clip(
        spoof_weight * spoof_liveness + temporal_weight * temporal_liveness,
        0.0, 1.0
    ))
    is_live = fused >= 0.5

    reason = (
        f"{'✅ LIVE' if is_live else '❌ SPOOF'}  (fused score: {fused:.2%})\n"
        f"  spoof model liveness : {spoof_liveness:.2%}  (weight: {spoof_weight:.0%})\n"
        f"  motion liveness      : {temporal_liveness:.2%}  (weight: {temporal_weight:.0%})\n"
        f"  ── motion detail ──\n"
        f"  {temporal_result.reason}"
    )
    return is_live, fused, reason


# ─── Gradio integration example ───────────────────────────────────────────────
"""
In your app.py (Gradio), use gr.State to hold the checker per user session:

    import gradio as gr
    from liveness_temporal import TemporalLivenessChecker, fuse_with_spoof_score

    def process_frame(frame, checker: TemporalLivenessChecker, spoof_score: float):
        bboxes, landmarks = ModelD(frame)
        if not landmarks:
            return checker, "No face detected"

        checker.add_frame(landmarks[0])

        if checker.ready:
            temporal = checker.evaluate()
            is_live, fused, reason = fuse_with_spoof_score(spoof_score, temporal)
            return checker, reason

        return checker, f"Collecting frames… {checker.frame_count}/{checker.min_frames}"

    with gr.Blocks() as demo:
        checker_state = gr.State(TemporalLivenessChecker())
        webcam        = gr.Image(sources=["webcam"], streaming=True)
        result_box    = gr.Textbox()

        webcam.stream(
            fn=process_frame,
            inputs=[webcam, checker_state, spoof_score_state],
            outputs=[checker_state, result_box],
        )
"""