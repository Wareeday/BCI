"""
ml/sklearn_baseline.py
======================
scikit-learn baseline classifiers: LDA, SVM (RBF), RandomForest.

Performance comparison (BCI Competition IV Dataset 2a):
  LDA (baseline): 72–78%  0.5ms CPU   Too inaccurate for safety
  SVM (RBF):      79–82%  1.2ms CPU   No spatial learning
  CNN:            91%     8ms  GPU    ✓ Chosen

These baselines serve as:
1. Fast fallback when CNN is unavailable (CUDA OOM / model corrupt)
2. Grid-search comparison in evaluate.py
3. Feature importance via RandomForest (which features matter most)
"""

import os
import joblib
import numpy as np
from typing import Optional, Tuple
from loguru import logger

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class SKLearnBaseline:
    """
    Wrapper around scikit-learn classifiers for BCI neural decoding.

    Usage:
        clf = SKLearnBaseline(method="lda")
        clf.fit(X_train, y_train)
        label, probs, conf = clf.predict(epoch)
    """

    METHODS = {
        "lda": {
            "estimator": LinearDiscriminantAnalysis(solver="svd"),
            "param_grid": {"lda__solver": ["svd", "lsqr"]},
        },
        "svm": {
            "estimator": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"),
            "param_grid": {
                "svm__C": [0.1, 1.0, 10.0, 100.0],
                "svm__gamma": ["scale", "auto"],
            },
        },
        "rf": {
            "estimator": RandomForestClassifier(n_estimators=100, random_state=42),
            "param_grid": {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [None, 10, 20],
            },
        },
    }

    def __init__(self, method: str = "lda", n_classes: int = 4):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS)}")
        self.method = method
        self.n_classes = n_classes
        self.class_names = ["left", "right", "feet", "rest"][:n_classes]
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted = False
        self._build_pipeline()

    def _build_pipeline(self):
        """Build scaler + classifier pipeline."""
        cfg = self.METHODS[self.method]
        estimator = cfg["estimator"]
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (self.method, estimator),
        ])
        logger.debug(f"Built {self.method.upper()} pipeline")

    def fit(self, X: np.ndarray, y: np.ndarray, grid_search: bool = False):
        """
        Train the classifier.

        Args:
            X: (n_trials, n_features) — flattened feature vectors
            y: (n_trials,) integer labels
            grid_search: run GridSearchCV if True
        """
        logger.info(f"Training {self.method.upper()}: {X.shape[0]} trials, {X.shape[1]} features")

        if grid_search:
            param_grid = self.METHODS[self.method]["param_grid"]
            gs = GridSearchCV(
                self._pipeline, param_grid,
                cv=5, scoring="accuracy", n_jobs=-1, verbose=1,
            )
            gs.fit(X, y)
            self._pipeline = gs.best_estimator_
            logger.success(
                f"{self.method.upper()} GridSearch best params: {gs.best_params_}, "
                f"CV accuracy: {gs.best_score_:.3f}"
            )
        else:
            self._pipeline.fit(X, y)

        # Report training accuracy
        y_pred = self._pipeline.predict(X)
        train_acc = accuracy_score(y, y_pred)
        logger.success(f"{self.method.upper()} train accuracy: {train_acc:.3f}")
        self._is_fitted = True

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate on test set and return metrics dict."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        y_pred = self._pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(
            y, y_pred,
            target_names=self.class_names,
            output_dict=True,
        )
        logger.info(
            f"{self.method.upper()} test accuracy: {acc:.3f}\n"
            + classification_report(y, y_pred, target_names=self.class_names)
        )
        return {"accuracy": acc, "report": report}

    def predict(self, X_single: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Real-time inference on one feature vector.

        Args:
            X_single: (n_features,) or (1, n_features)
        Returns:
            (predicted_class, probabilities, confidence)
        """
        if not self._is_fitted:
            probs = np.ones(self.n_classes) / self.n_classes
            return 0, probs, 1.0 / self.n_classes

        x = X_single.reshape(1, -1)
        probs = self._pipeline.predict_proba(x)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return pred_class, probs, confidence

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importances (RandomForest only)."""
        if self.method != "rf":
            return None
        rf = self._pipeline.named_steps["rf"]
        return rf.feature_importances_

    def save(self, path: str):
        """Serialize pipeline with joblib."""
        joblib.dump(self._pipeline, path)
        logger.info(f"{self.method.upper()} model saved: {path}")

    def load(self, path: str):
        """Load serialized pipeline."""
        if os.path.exists(path):
            self._pipeline = joblib.load(path)
            self._is_fitted = True
            logger.success(f"{self.method.upper()} model loaded: {path}")
        else:
            logger.warning(f"Model file not found: {path}")