"""
dsp/asr_cleaning.py
===================
Artifact Subspace Reconstruction (ASR) — MNE-Python implementation.

ASR identifies and removes high-amplitude artifact bursts by comparing
EEG data against a clean reference window using PCA.

Used after ICA to catch residual artifacts that ICA misses (brief
spike-like transients from electrode pops or movement).

Reference: Mullen et al. (2015) Real-time neuroimaging and cognitive
  monitoring using wearable dry EEG. IEEE TNSRE.
"""
import numpy as np
from loguru import logger


class ASRCleaning:
    """
    Simplified ASR implementation.

    Full ASR (as in EEGLab/MNE) requires a clean reference segment
    to fit the covariance model. We use a simulated threshold approach
    compatible with the BCI real-time budget.
    """

    def __init__(
        self,
        sample_rate: float = 250.0,
        cutoff: float = 5.0,           # rejection threshold (SD multiplier)
        reference_window_s: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.reference_window_s = reference_window_s
        self._reference_cov: np.ndarray | None = None
        self._artifacts_cleaned = 0

    def fit(self, clean_data: np.ndarray):
        """Fit on a clean reference segment (first 5 s of calibration)."""
        self._reference_cov = np.cov(clean_data)
        logger.info("ASR: reference covariance fitted")

    def clean(self, data: np.ndarray) -> np.ndarray:
        """
        Remove artifact bursts exceeding cutoff × RMS.

        Simple version: per-channel amplitude clipping at cutoff * std.
        Production version: full rank-deficient ASR reconstruction.
        """
        if data.size == 0:
            return data

        clean = data.copy()
        for ch in range(data.shape[0]):
            ch_data = clean[ch]
            threshold = self.cutoff * np.std(ch_data)
            mask = np.abs(ch_data) > threshold
            if np.any(mask):
                # Replace artifact samples with interpolated values
                clean[ch, mask] = np.interp(
                    np.where(mask)[0],
                    np.where(~mask)[0],
                    ch_data[~mask],
                ) if np.sum(~mask) > 1 else 0.0
                self._artifacts_cleaned += int(np.sum(mask))

        return clean.astype(np.float32)

    @property
    def artifacts_cleaned(self) -> int:
        return self._artifacts_cleaned