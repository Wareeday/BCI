"""
dsp/epoch_extraction.py
=======================
Epoch extraction for P300 and motor imagery paradigms.

P300 paradigm:
  - Epoch: 0–800 ms post-stimulus
  - Baseline correction: -200–0 ms pre-stimulus
  - 10 intensifications per character (oddball paradigm)

Motor imagery paradigm:
  - Epoch: 0–4 s post-cue
  - 4 classes: left hand, right hand, feet, rest
  - Based on BCI Competition IV Dataset 2a protocol
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class Epoch:
    """A single EEG epoch ready for feature extraction."""
    data: np.ndarray            # (n_channels, n_samples)
    timestamp: float
    epoch_type: str             # 'p300' or 'motor_imagery'
    label: Optional[int] = None   # ground truth (for training)
    trial_id: Optional[int] = None


class EpochExtractor:
    """
    Extracts fixed-length epochs from continuous EEG.

    P300:  8 channels × 200 samples (800 ms @ 250 Hz)
    Motor: 8 channels × 1000 samples (4 s @ 250 Hz)
    """

    EPOCH_PARAMS = {
        "p300": {
            "duration_s": 0.8,
            "baseline_s": 0.2,
            "description": "ERP epoch 0–800ms, N200+P300 components",
        },
        "motor_imagery": {
            "duration_s": 4.0,
            "baseline_s": 0.5,
            "description": "Motor imagery cue 0–4s, mu/beta ERD",
        },
    }

    def __init__(
        self,
        sample_rate: float = 250.0,
        epoch_type: str = "p300",
    ):
        self.sample_rate = sample_rate
        self.epoch_type = epoch_type
        params = self.EPOCH_PARAMS[epoch_type]
        self.epoch_samples = int(params["duration_s"] * sample_rate)
        self.baseline_samples = int(params["baseline_s"] * sample_rate)
        logger.debug(
            f"EpochExtractor: type={epoch_type}, "
            f"samples={self.epoch_samples} ({params['duration_s']}s)"
        )

    def extract(
        self,
        data: np.ndarray,
        timestamp: float,
        label: Optional[int] = None,
    ) -> Epoch:
        """
        Extract and baseline-correct one epoch.

        Args:
            data:      (n_channels, n_samples) — may be longer than epoch
            timestamp: onset time
            label:     class label for training (None for inference)
        """
        n_channels, n_samples = data.shape
        # Take exactly epoch_samples (already correct from pipeline)
        epoch_data = data[:, :self.epoch_samples].copy()

        # Baseline correction: subtract mean of baseline window
        if self.baseline_samples > 0 and n_samples >= self.baseline_samples:
            baseline_mean = np.mean(
                epoch_data[:, :self.baseline_samples], axis=1, keepdims=True
            )
            epoch_data -= baseline_mean

        return Epoch(
            data=epoch_data.astype(np.float32),
            timestamp=timestamp,
            epoch_type=self.epoch_type,
            label=label,
        )