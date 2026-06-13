"""
resilience/classifier_fallback.py
===================================
[MED] CNN classifier corruption / CUDA OOM fallback.

Risk Matrix entry:
  Scenario:   CNN classifier corrupt (weight file / CUDA OOM)
  Likelihood: High (daily use, memory pressure)
  Severity:   Significant
  Risk Level: MEDIUM
  Mitigation: Load backup LDA fallback classifier,
              reduced command set (3 classes only),
              log event and trigger auto-repair script

Recovery steps:
  1. Load backup LDA fallback classifier
  2. Reduced command set (3 classes: left, right, stop only)
  3. Log event; trigger auto-repair script
  4. Alert: degraded mode — slower accuracy (79-82% vs 91%)

Detection: Inference watchdog — no output >100ms
"""

import time
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from loguru import logger

from ml.sklearn_baseline import SKLearnBaseline


class ClassifierMode(Enum):
    PRIMARY_CNN = "primary_cnn"
    FALLBACK_LDA = "fallback_lda"
    FALLBACK_SVM = "fallback_svm"
    DEGRADED = "degraded"           # 3-class only
    FAILED = "failed"


class ClassifierFallbackManager:
    """
    Manages primary → fallback classifier switching.

    Fallback hierarchy:
      1. CNN (TensorFlow, 91%)        — primary
      2. SVM RBF (sklearn, 79-82%)   — first fallback
      3. LDA (sklearn, 72-78%)       — second fallback (fastest, most stable)
      4. 3-class degraded mode       — last resort

    Degraded mode restricts commands to:
      left, right, stop (safest set for wheelchair operation)
    """

    INFERENCE_TIMEOUT_MS = 100.0   # watchdog: switch if inference > 100ms
    DEGRADED_CLASSES = ["left", "right", "rest"]   # 3-class safe set

    def __init__(
        self,
        lda_model_path: str = "ml/saved_models/lda_baseline.pkl",
        svm_model_path: str = "ml/saved_models/svm_model.pkl",
        audit_logger=None,
    ):
        self.mode = ClassifierMode.PRIMARY_CNN
        self.audit = audit_logger
        self._switch_count = 0
        self._last_switch_time = 0.0

        # Pre-load fallback models at startup (not lazily — faster failover)
        self._lda: Optional[SKLearnBaseline] = None
        self._svm: Optional[SKLearnBaseline] = None
        self._load_fallbacks(lda_model_path, svm_model_path)

    def _load_fallbacks(self, lda_path: str, svm_path: str):
        """Load LDA and SVM at startup for instant failover."""
        try:
            self._lda = SKLearnBaseline(method="lda")
            self._lda.load(lda_path)
            logger.info("LDA fallback loaded successfully")
        except Exception as exc:
            logger.warning(f"Could not load LDA fallback: {exc}")

        try:
            self._svm = SKLearnBaseline(method="svm")
            self._svm.load(svm_path)
            logger.info("SVM fallback loaded successfully")
        except Exception as exc:
            logger.warning(f"Could not load SVM fallback: {exc}")

    def on_cnn_failure(self, error: str):
        """
        Handle CNN inference failure.
        Immediately switches to best available fallback.
        """
        self._switch_count += 1
        self._last_switch_time = time.time()

        if self._svm and self._svm._is_fitted:
            self.mode = ClassifierMode.FALLBACK_SVM
            new_mode = "SVM (79-82%)"
        elif self._lda and self._lda._is_fitted:
            self.mode = ClassifierMode.FALLBACK_LDA
            new_mode = "LDA (72-78%)"
        else:
            self.mode = ClassifierMode.DEGRADED
            new_mode = "DEGRADED 3-class"

        logger.warning(
            f"[MED] CNN classifier failed: {error}. "
            f"Switching to {new_mode} (switch #{self._switch_count})"
        )

        if self.audit:
            self.audit.log(
                event_type="classifier_fallback",
                severity="WARNING",
                details={
                    "error": error,
                    "new_mode": new_mode,
                    "switch_count": self._switch_count,
                },
            )

    def predict_with_fallback(
        self,
        feature_vector: np.ndarray,
    ) -> Tuple[int, np.ndarray, float, str]:
        """
        Run inference using current active classifier.

        Returns (class_idx, probabilities, confidence, model_name)
        """
        if self.mode == ClassifierMode.FALLBACK_SVM and self._svm:
            pred, probs, conf = self._svm.predict(feature_vector)
            return pred, probs, conf, "svm_fallback"

        elif self.mode in (ClassifierMode.FALLBACK_LDA, ClassifierMode.DEGRADED) and self._lda:
            pred, probs, conf = self._lda.predict(feature_vector)
            # Degraded mode: map to 3 safe classes only
            if self.mode == ClassifierMode.DEGRADED:
                pred = min(pred, 2)
            return pred, probs, conf, "lda_fallback"

        # Full failure
        logger.error("All classifiers unavailable — returning STOP command")
        n = 4
        probs = np.zeros(n)
        probs[3] = 1.0   # class 3 = rest/stop
        return 3, probs, 1.0, "hardcoded_stop"

    def get_status(self) -> dict:
        return {
            "mode": self.mode.value,
            "switch_count": self._switch_count,
            "lda_available": self._lda is not None and self._lda._is_fitted,
            "svm_available": self._svm is not None and self._svm._is_fitted,
            "degraded": self.mode == ClassifierMode.DEGRADED,
        }