"""
dsp/notch_filter.py
===================
50 Hz (EU) / 60 Hz (US) IIR notch filter.

Removes powerline interference. GNU Radio equivalent: iir_filter block
with IIR design: notch_filter(Q=30, fs=250, f0=50).

Design choice: notch over lowpass at 40 Hz because some motor imagery
features (gamma, 30–80 Hz) may be required in future extensions.
The notch surgically removes only the powerline frequency.
"""
import numpy as np
from scipy.signal import iirnotch, sosfilt, tf2sos
from loguru import logger


class NotchFilter:
    """IIR notch filter at powerline frequency."""

    def __init__(
        self,
        notch_hz: float = 50.0,
        quality_factor: float = 30.0,
        sample_rate: float = 250.0,
    ):
        self.notch_hz = notch_hz
        self.quality_factor = quality_factor
        self.sample_rate = sample_rate

        b, a = iirnotch(notch_hz, quality_factor, sample_rate)
        self.sos = tf2sos(b, a)
        logger.debug(f"NotchFilter: {notch_hz} Hz, Q={quality_factor}, Fs={sample_rate}")

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter.

        Args:
            data: (n_channels, n_samples)
        Returns:
            filtered: same shape
        """
        from scipy.signal import sosfiltfilt
        return sosfiltfilt(self.sos, data, axis=1).astype(np.float32)