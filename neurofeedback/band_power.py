"""
neurofeedback/band_power.py
============================
Real-time EEG band power estimation using Welch PSD.

Frequency bands:
  Delta: 1–4 Hz   (deep sleep, not relevant for BCI but tracked)
  Theta: 4–8 Hz   (drowsiness, mental fatigue indicator)
  Alpha: 8–13 Hz  (relaxation, mu rhythm suppression in motor imagery)
  Beta:  13–30 Hz (active concentration, motor imagery activation)

Alpha/beta ratio is the primary neurofeedback metric for motor BCI.
Higher beta = stronger motor imagery intention.
"""

import numpy as np
from scipy.signal import welch
from collections import deque
from typing import Optional


class BandPowerCalculator:
    """Compute instantaneous band power from EEG epoch using Welch PSD."""

    BANDS = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
    }

    def __init__(self, sample_rate: float = 250.0, window_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        # Normalisation baseline (updated from first 5 epochs)
        self._baselines: dict[str, deque] = {b: deque(maxlen=50) for b in self.BANDS}

    def compute(self, data: np.ndarray) -> dict[str, float]:
        """
        Compute mean band power across all channels.

        Args:
            data: (n_channels, n_samples)

        Returns:
            {band_name: power_in_uV2}
        """
        n_per_seg = min(self.window_samples, data.shape[1])
        powers = {b: 0.0 for b in self.BANDS}

        for ch in range(data.shape[0]):
            freqs, psd = welch(
                data[ch],
                fs=self.sample_rate,
                nperseg=n_per_seg,
                window="hann",
            )
            for band, (low, high) in self.BANDS.items():
                idx = (freqs >= low) & (freqs < high)
                powers[band] += float(np.mean(psd[idx])) if np.any(idx) else 0.0

        # Average across channels
        n_ch = data.shape[0]
        for band in powers:
            powers[band] /= n_ch
            self._baselines[band].append(powers[band])

        return powers

    def normalise(self, power: float, band: str) -> float:
        """
        Normalise band power to [0, 1] relative to session baseline.

        Returns 0.5 until enough baseline samples collected.
        """
        baseline = list(self._baselines.get(band, []))
        if len(baseline) < 5:
            return 0.5
        mean_base = float(np.mean(baseline))
        std_base = float(np.std(baseline)) + 1e-8
        z = (power - mean_base) / std_base
        # Map z-score to [0, 1] using sigmoid
        return float(1.0 / (1.0 + np.exp(-z)))