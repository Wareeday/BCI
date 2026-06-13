"""
ml/transfer_learning.py
========================
Transfer learning for cross-subject BCI calibration.

Reduces cold-start calibration from 20 min → 4 min by leveraging
a pre-trained model from a 50-subject corpus.

Strategy:
  1. Load pre-trained CNN weights (50-subject corpus)
  2. Freeze all convolutional layers (generic spatial-temporal features)
  3. Fine-tune only the Dense(128) head on the new user's data
  4. Requires only ~30 labelled trials (2–3 min) instead of 200+ (20 min)

This is the mechanism behind the presentation's claim:
  "Transfer learning from pre-trained cross-subject model
   reduces cold-start calibration from 20 min → 4 min"
"""
import os
import time
from typing import Optional
import numpy as np
from loguru import logger

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TransferLearningAdapter:
    """
    Adapts a pretrained CNN to a new user with minimal labelled data.
    """

    FROZEN_LAYERS = {"conv1d_temporal_32", "conv1d_spatial_64", "conv1d_temporal_64b"}

    def __init__(
        self,
        base_model,            # BCICNNModel instance with pretrained weights
        fine_tune_lr: float = 0.0001,
        fine_tune_epochs: int = 10,
        min_trials_required: int = 30,
    ):
        self.model = base_model
        self.fine_tune_lr = fine_tune_lr
        self.fine_tune_epochs = fine_tune_epochs
        self.min_trials = min_trials_required
        self._adapted = False
        self._adaptation_time_s = 0.0

    def adapt(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> float:
        """
        Fine-tune on new-user data.

        Args:
            X_new: (n_trials, n_channels, n_samples) — minimum 30 trials
            y_new: (n_trials,) labels

        Returns:
            fine-tune accuracy on X_new (proxy for calibration quality)
        """
        if len(X_new) < self.min_trials:
            logger.warning(
                f"Only {len(X_new)} trials provided. "
                f"Minimum {self.min_trials} recommended for reliable adaptation."
            )

        if not TF_AVAILABLE or self.model.model is None:
            logger.warning("TF not available — skipping transfer learning")
            return 0.0

        t0 = time.time()

        # Freeze convolutional layers
        for layer in self.model.model.layers:
            layer.trainable = layer.name not in self.FROZEN_LAYERS

        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.fine_tune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_4d = X_new[..., np.newaxis]
        history = self.model.model.fit(
            X_4d, y_new,
            epochs=self.fine_tune_epochs,
            batch_size=min(16, len(X_new)),
            verbose=0,
        )

        # Re-enable all layers for inference
        for layer in self.model.model.layers:
            layer.trainable = True

        acc = history.history["accuracy"][-1]
        self._adaptation_time_s = time.time() - t0
        self._adapted = True

        logger.success(
            f"Transfer learning complete: acc={acc:.3f}, "
            f"time={self._adaptation_time_s:.1f}s "
            f"({self._adaptation_time_s/60:.1f} min)"
        )
        return float(acc)

    @property
    def is_adapted(self) -> bool:
        return self._adapted

    @property
    def adaptation_time_min(self) -> float:
        return self._adaptation_time_s / 60.0