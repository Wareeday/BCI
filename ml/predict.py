"""
ml/predict.py
=============
Real-time inference orchestrator.

Manages a primary CNN model with LDA/SVM fallback.
Implements the confidence-gated command decision logic:
  - Confidence >= 0.85: issue command
  - Confidence 0.75–0.85: request confirmation
  - Confidence < 0.75: hold, do not actuate

Safety: a false positive (unintended wheelchair movement) is classified
as a CRITICAL risk in the risk matrix. The confidence gate is the primary
mitigation (target: <4.2% false positive rate per ISO 14155 bench test).
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger

from ml.cnn_model import BCICNNModel
from ml.sklearn_baseline import SKLearnBaseline


class CommandDecision(Enum):
    ISSUE = "issue"             # confidence >= PRIMARY_THRESHOLD
    CONFIRM = "confirm"         # confidence in [FALLBACK_THRESHOLD, PRIMARY_THRESHOLD)
    HOLD = "hold"               # confidence < FALLBACK_THRESHOLD


@dataclass
class PredictionResult:
    """Output of one inference cycle."""
    timestamp: float
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    decision: CommandDecision
    model_used: str             # 'cnn', 'lda', 'svm'
    inference_ms: float


class RealTimePredictor:
    """
    Real-time inference with primary CNN + fallback baseline.

    Decision pipeline:
      1. Try CNN (primary)
      2. If CNN unavailable/crashed → fallback to LDA
      3. Apply confidence gate
      4. Return CommandDecision
    """

    PRIMARY_THRESHOLD = 0.85
    FALLBACK_THRESHOLD = 0.75
    CLASS_NAMES_MI = ["left", "right", "feet", "rest"]
    CLASS_NAMES_P300 = ["non-target", "target"]

    def __init__(
        self,
        cnn_model: Optional[BCICNNModel] = None,
        fallback_model: Optional[SKLearnBaseline] = None,
        paradigm: str = "motor_imagery",
        primary_threshold: float = 0.85,
        fallback_threshold: float = 0.75,
    ):
        self.cnn = cnn_model
        self.fallback = fallback_model
        self.paradigm = paradigm
        self.PRIMARY_THRESHOLD = primary_threshold
        self.FALLBACK_THRESHOLD = fallback_threshold

        self._class_names = (
            self.CLASS_NAMES_MI if paradigm == "motor_imagery" else self.CLASS_NAMES_P300
        )
        self._prediction_history: list[PredictionResult] = []
        self._cnn_failures = 0

        logger.info(
            f"RealTimePredictor: paradigm={paradigm}, "
            f"primary_threshold={primary_threshold}, "
            f"fallback_threshold={fallback_threshold}"
        )

    def predict(self, epoch: np.ndarray, feature_vector: Optional[np.ndarray] = None) -> PredictionResult:
        """
        Run inference on one epoch.

        Args:
            epoch:          (n_channels, n_samples) — for CNN
            feature_vector: (n_features,) — for sklearn fallback

        Returns:
            PredictionResult with CommandDecision
        """
        t_start = time.perf_counter()
        model_used = "none"
        pred_class = 0
        probs = np.ones(len(self._class_names)) / len(self._class_names)
        confidence = 1.0 / len(self._class_names)

        # ── Primary: CNN ───────────────────────────────────────────
        cnn_ok = (
            self.cnn is not None
            and self._cnn_failures < 3     # disable after 3 consecutive failures
        )

        if cnn_ok:
            try:
                pred_class, probs, confidence = self.cnn.predict(epoch)
                model_used = "cnn"
                self._cnn_failures = 0
            except Exception as exc:
                logger.error(f"CNN inference failed: {exc}")
                self._cnn_failures += 1
                cnn_ok = False

        # ── Fallback: sklearn ──────────────────────────────────────
        if not cnn_ok and self.fallback is not None and feature_vector is not None:
            try:
                pred_class, probs, confidence = self.fallback.predict(feature_vector)
                model_used = self.fallback.method
                logger.warning(f"Using fallback {model_used.upper()} classifier")
            except Exception as exc:
                logger.error(f"Fallback inference also failed: {exc}")

        # ── Confidence gate ────────────────────────────────────────
        if confidence >= self.PRIMARY_THRESHOLD:
            decision = CommandDecision.ISSUE
        elif confidence >= self.FALLBACK_THRESHOLD:
            decision = CommandDecision.CONFIRM
        else:
            decision = CommandDecision.HOLD

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        class_name = self._class_names[pred_class] if pred_class < len(self._class_names) else "unknown"

        result = PredictionResult(
            timestamp=time.time(),
            predicted_class=pred_class,
            class_name=class_name,
            confidence=confidence,
            probabilities=probs,
            decision=decision,
            model_used=model_used,
            inference_ms=inference_ms,
        )

        self._prediction_history.append(result)
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-500:]

        if decision == CommandDecision.ISSUE:
            logger.debug(
                f"COMMAND: {class_name} (conf={confidence:.2f}, model={model_used}, {inference_ms:.1f}ms)"
            )
        elif decision == CommandDecision.HOLD:
            logger.debug(f"HOLD: confidence too low ({confidence:.2f})")

        return result

    def get_recent_accuracy(self, n: int = 30) -> Optional[float]:
        """
        Compute accuracy over last n predictions (requires ground truth labels).
        Used by adaptive calibration monitoring.
        """
        # In real usage, ground truth comes from user confirmation signals
        # Here we return None (no ground truth available in pure inference mode)
        return None

    def get_stats(self) -> dict:
        recent = self._prediction_history[-100:]
        if not recent:
            return {}
        confs = [r.confidence for r in recent]
        issues = sum(1 for r in recent if r.decision == CommandDecision.ISSUE)
        holds = sum(1 for r in recent if r.decision == CommandDecision.HOLD)
        return {
            "total_predictions": len(self._prediction_history),
            "mean_confidence": float(np.mean(confs)),
            "issue_rate": issues / len(recent),
            "hold_rate": holds / len(recent),
            "cnn_failure_count": self._cnn_failures,
            "active_model": "cnn" if self._cnn_failures < 3 else "fallback",
        }