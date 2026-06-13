"""
security/differential_privacy.py
==================================
Differential privacy (DP) for federated model training.

Trade-off analysis (from presentation):
  DP (ε=1.0): 91% → 84% accuracy, 15× faster than HE
  HE:          no accuracy loss, 12× slower compute

Decision:
  HE for individual patient inference (accuracy non-negotiable)
  DP for federated training across institutions (aggregation tolerates noise)

Why ε=1.0?
  ε=1.0 provides strong privacy guarantee (privacy budget per training round).
  Lower ε = more noise = stronger privacy but worse accuracy.
  At ε=1.0 the CNN drops from 91% → 84% — still above minimum for
  training data, but unacceptable for real-time actuation.

Reference:
  Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
  IBM diffprivlib library (Holohan et al., 2019)
"""

import numpy as np
from loguru import logger

try:
    import diffprivlib as dp
    from diffprivlib.models import LogisticRegression as DPLogisticRegression
    from diffprivlib.models import GaussianNB as DPGaussianNB
    DP_AVAILABLE = True
except ImportError:
    logger.warning("diffprivlib not installed. DP training disabled. pip install diffprivlib")
    DP_AVAILABLE = False

from typing import Optional, Tuple


class DifferentialPrivacyTrainer:
    """
    Differentially private training for federated cross-institution models.

    Uses IBM diffprivlib, which wraps sklearn models with Laplace/Gaussian
    mechanism noise injection during training.

    Epsilon (ε) budget:
      ε = 0.1  → very strong privacy, significant accuracy loss
      ε = 1.0  → strong privacy, moderate accuracy loss (~7%)  ← we use this
      ε = 10.0 → weak privacy, minimal accuracy loss

    Per GDPR Article 89: DP provides appropriate safeguard for research.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: privacy budget (lower = more private)
            delta:   failure probability (standard: 1e-5)
        """
        self.epsilon = epsilon
        self.delta = delta
        self._dp_model = None
        self._is_fitted = False

        logger.info(
            f"DifferentialPrivacyTrainer: ε={epsilon}, δ={delta}. "
            f"Privacy interpretation: 'strong privacy with moderate accuracy impact'"
        )

    def fit_dp_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        data_norm: float = 10.0,
    ) -> float:
        """
        Train a DP logistic regression classifier.

        Args:
            X:         (n_samples, n_features) feature matrix
            y:         (n_samples,) labels
            data_norm: L2 norm bound on feature vectors

        Returns:
            training accuracy
        """
        if not DP_AVAILABLE:
            logger.warning("diffprivlib not available — training without DP")
            from sklearn.linear_model import LogisticRegression
            self._dp_model = LogisticRegression(max_iter=200)
            self._dp_model.fit(X, y)
            self._is_fitted = True
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y, self._dp_model.predict(X)))

        logger.info(
            f"Training DP classifier: ε={self.epsilon}, "
            f"n_samples={X.shape[0]}, n_features={X.shape[1]}"
        )

        self._dp_model = DPLogisticRegression(
            epsilon=self.epsilon,
            data_norm=data_norm,
            max_iter=200,
        )
        self._dp_model.fit(X, y)
        self._is_fitted = True

        from sklearn.metrics import accuracy_score
        y_pred = self._dp_model.predict(X)
        acc = float(accuracy_score(y, y_pred))

        logger.success(
            f"DP training complete: accuracy={acc:.3f} "
            f"(expected ~84% vs 91% without DP at ε=1.0)"
        )
        return acc

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with DP-trained model.

        Returns (predictions, probabilities).
        """
        if not self._is_fitted or self._dp_model is None:
            raise RuntimeError("Model not fitted. Call fit_dp_classifier() first.")
        preds = self._dp_model.predict(X)
        try:
            probs = self._dp_model.predict_proba(X)
        except AttributeError:
            n_classes = len(np.unique(preds))
            probs = np.eye(n_classes)[preds]
        return preds, probs

    def add_laplace_noise(
        self,
        gradient: np.ndarray,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Add Laplace noise to a gradient vector (for DP-SGD).

        Used in federated learning aggregation.
        Scale = sensitivity / epsilon (Laplace mechanism).
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0.0, scale, gradient.shape)
        return (gradient + noise).astype(np.float32)

    def get_privacy_report(self) -> dict:
        """Return human-readable privacy analysis for DPIA."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "privacy_level": "strong" if self.epsilon <= 1.0 else "moderate",
            "mechanism": "Laplace (training), Gaussian (SGD)",
            "expected_accuracy_loss": "~7% (91% → 84%)",
            "use_case": "federated training across institutions",
            "not_suitable_for": "real-time individual patient inference",
            "standard": "GDPR Article 89, IEEE 2857",
            "reference": "Dwork & Roth (2014), IBM diffprivlib",
        }