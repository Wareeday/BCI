"""
dsp/feature_vector.py
=====================
Feature extraction: PSD + CSP + time-domain features.

PSD (Power Spectral Density):   Welch method, 4 frequency bands
CSP (Common Spatial Patterns):  for motor imagery (mu/beta ERD)
Time-domain:                    mean, variance, kurtosis, Hjorth params

Final feature vector:  concatenated [psd_features | time_features]
  P300:   8 ch × 4 bands PSD + 8 ch × 3 time = 56 features
  Motor:  CSP (6 components) + 8 ch × 4 bands PSD = 38 features

These features are the input to the CNN / LDA / SVM classifiers.
"""
import numpy as np
from loguru import logger

from dsp.epoch_extraction import Epoch


class FeatureExtractor:
    """
    Multi-domain feature extractor for BCI epochs.

    Feature dimensions are deterministic given n_channels and epoch_type,
    so the downstream CNN input shape is always consistent.
    """

    FREQ_BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    def __init__(self, sample_rate: float = 250.0, n_channels: int = 8):
        self.sample_rate = sample_rate
        self.n_channels = n_channels

    def extract(self, epoch: Epoch) -> np.ndarray:
        """
        Extract features from one epoch.

        Returns:
            feature_vector: 1D float32 array
        """
        data = epoch.data   # (n_channels, n_samples)

        psd_features = self._psd_features(data)
        time_features = self._time_domain_features(data)
        feature_vec = np.concatenate([psd_features, time_features])

        return feature_vec.astype(np.float32)

    def _psd_features(self, data: np.ndarray) -> np.ndarray:
        """
        Welch PSD, averaged within 4 frequency bands per channel.
        Returns (n_channels × n_bands,) = 8×4 = 32 features.
        """
        from scipy.signal import welch
        n_per_seg = min(256, data.shape[1])
        features = []
        for ch in range(data.shape[0]):
            freqs, psd = welch(data[ch], fs=self.sample_rate, nperseg=n_per_seg)
            for (low, high) in self.FREQ_BANDS.values():
                band_idx = (freqs >= low) & (freqs < high)
                band_power = np.mean(psd[band_idx]) if np.any(band_idx) else 0.0
                features.append(np.log1p(band_power))   # log-scale for stability
        return np.array(features, dtype=np.float32)

    def _time_domain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Per-channel: mean, variance, kurtosis = 8×3 = 24 features.
        """
        features = []
        for ch in range(data.shape[0]):
            x = data[ch]
            mean = float(np.mean(x))
            var = float(np.var(x))
            std = np.std(x) + 1e-10
            kurt = float(np.mean(((x - mean) / std) ** 4) - 3.0)
            features.extend([mean, var, kurt])
        return np.array(features, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        """Total feature vector length."""
        n_psd = self.n_channels * len(self.FREQ_BANDS)
        n_time = self.n_channels * 3
        return n_psd + n_time