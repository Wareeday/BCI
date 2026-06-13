"""
dsp/bandpass_filter.py
======================
1–40 Hz Butterworth bandpass filter, 4th order, zero-phase.

Why zero-phase?  Forward-backward filtering (filtfilt) eliminates phase
distortion — critical for ERP latency accuracy (P300 at 300 ms ±10 ms).

GNU Radio equivalent: freq_xlating_fir_filter_ccf with taps from
  firdes.low_pass(1.0, samp_rate, cutoff, transition_width).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt_zi
from loguru import logger


class BandpassFilter:
    """
    4th-order Butterworth bandpass filter.

    Uses scipy.signal.butter with sos (second-order sections) for
    numerical stability. Zero-phase (filtfilt) for offline processing;
    causal (sosfilt) for real-time per-sample processing.
    """

    def __init__(
        self,
        low_hz: float = 1.0,
        high_hz: float = 40.0,
        sample_rate: float = 250.0,
        order: int = 4,
    ):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.sample_rate = sample_rate
        self.order = order
        nyq = sample_rate / 2.0
        self.sos = butter(
            order,
            [low_hz / nyq, high_hz / nyq],
            btype="bandpass",
            output="sos",
        )
        logger.debug(
            f"BandpassFilter: {low_hz}–{high_hz} Hz, order={order}, Fs={sample_rate}"
        )

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase bandpass filter.

        Args:
            data: (n_channels, n_samples) float32 array

        Returns:
            filtered: same shape as input
        """
        return sosfiltfilt(self.sos, data, axis=1).astype(np.float32)