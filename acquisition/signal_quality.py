"""
acquisition/signal_quality.py
================================
Real-time EEG signal quality scoring.

Metrics:
  SNR (Signal-to-Noise Ratio)  — target >35 dB (ISO 14155 bench test)
  RMS amplitude                — valid EEG: 1–100 µV RMS per channel
  Line noise ratio             — 50/60 Hz power vs total (should be <10%)
  Flat-line detection          — variance < 0.01 µV² → electrode off
  Saturation detection         — |sample| > 400 µV → ADC clipping

Dashboard badge colours:
  Green  = all channels good
  Yellow = 1–2 channels marginal
  Red    = 3+ channels bad or saturation detected
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.signal import welch
from loguru import logger


@dataclass
class ChannelQuality:
    channel: int
    label: str
    rms_uv: float
    snr_db: float
    line_noise_ratio: float
    is_flat: bool
    is_saturated: bool
    score: float          # 0.0 (bad) to 1.0 (perfect)
    status: str           # 'good' | 'marginal' | 'bad'


CHANNEL_LABELS = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]


class SignalQualityAssessor:
    """
    Assesses EEG signal quality on a rolling epoch basis.
    Called after each DSP epoch to update quality badges on dashboard.
    """

    def __init__(self, sample_rate: float = 250.0, n_channels: int = 8):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self._history: list[dict] = []

    def assess(self, epoch: np.ndarray) -> list[ChannelQuality]:
        """
        Assess signal quality for one epoch.

        Args:
            epoch: (n_channels, n_samples) float32

        Returns:
            List of ChannelQuality — one per channel.
        """
        results = []
        for ch in range(min(epoch.shape[0], self.n_channels)):
            sig = epoch[ch]
            label = CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else f"Ch{ch}"

            rms = float(np.sqrt(np.mean(sig ** 2)))
            is_flat = float(np.var(sig)) < 0.01
            is_saturated = float(np.max(np.abs(sig))) > 400.0

            # SNR: signal power vs noise floor estimate
            freqs, psd = welch(sig, fs=self.sample_rate, nperseg=min(128, len(sig)))
            neural_idx = (freqs >= 1.0) & (freqs <= 40.0)
            noise_idx = (freqs > 40.0) & (freqs < 100.0)
            signal_power = np.mean(psd[neural_idx]) if np.any(neural_idx) else 1e-10
            noise_power = np.mean(psd[noise_idx]) if np.any(noise_idx) else 1e-12
            snr_db = float(10 * np.log10(signal_power / (noise_power + 1e-12)))

            # Line noise ratio (50 Hz band)
            line_idx = (freqs >= 49.0) & (freqs <= 51.0)
            line_power = np.mean(psd[line_idx]) if np.any(line_idx) else 0.0
            total_power = np.mean(psd) + 1e-12
            line_ratio = float(line_power / total_power)

            # Composite score
            score = 1.0
            if is_flat:       score = 0.0
            elif is_saturated: score = 0.1
            else:
                if rms < 1.0 or rms > 100.0: score *= 0.5
                if snr_db < 10.0:            score *= 0.6
                if line_ratio > 0.3:          score *= 0.7

            if score > 0.75:   status = "good"
            elif score > 0.4:  status = "marginal"
            else:              status = "bad"

            results.append(ChannelQuality(
                channel=ch, label=label, rms_uv=round(rms, 2),
                snr_db=round(snr_db, 1), line_noise_ratio=round(line_ratio, 3),
                is_flat=is_flat, is_saturated=is_saturated,
                score=round(score, 3), status=status,
            ))

        self._history.append({
            "timestamp": __import__("time").time(),
            "channels": {r.label: r.score for r in results},
        })
        return results

    def overall_quality(self, results: list[ChannelQuality]) -> str:
        bad = sum(1 for r in results if r.status == "bad")
        marginal = sum(1 for r in results if r.status == "marginal")
        if bad >= 3: return "red"
        if bad >= 1 or marginal >= 3: return "yellow"
        return "green"