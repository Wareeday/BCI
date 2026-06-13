"""
dsp/pipeline.py
===============
Real-time DSP pipeline orchestrator.

Implements the 6-stage processing pipeline from the presentation:
  1. Bandpass filter   (1–40 Hz Butterworth, 4th order, zero-phase)
  2. Notch filter      (50 Hz EU / 60 Hz US, GNU Radio IIR block)
  3. ICA artifact removal (FastICA — EOG, EMG, cardiac)
  4. ASR cleaning      (Artifact Subspace Reconstruction, MNE)
  5. Epoch extraction  (P300: 0–800 ms, Motor imagery: 0–4 s)
  6. Feature vector    (PSD, CSP, time-domain)

Total DSP latency budget:
  Acquisition 4ms + GNU Radio 3ms + Feature extraction 2ms = 9ms ✓ (<10ms)

Why GNU Radio over pure Python?
  C++ backend processes at 250 Hz with <1ms block latency using SIMD.
  Pure Python adds 8–15ms per buffer on Raspberry Pi 4.
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from loguru import logger

from dsp.bandpass_filter import BandpassFilter
from dsp.notch_filter import NotchFilter
from dsp.ica_artifact_removal import ICARemoval
from dsp.asr_cleaning import ASRCleaning
from dsp.epoch_extraction import EpochExtractor
from dsp.feature_vector import FeatureExtractor
from acquisition.openbci_board import EEGSample


@dataclass
class ProcessedEEG:
    """Output of the full DSP pipeline — ready for ML classifier."""
    timestamp: float
    raw_channels: np.ndarray            # (n_channels, n_samples)
    filtered_channels: np.ndarray       # after bandpass + notch
    clean_channels: np.ndarray          # after ICA + ASR
    feature_vector: np.ndarray          # flattened feature array
    epoch_type: str                     # 'p300' or 'motor_imagery'
    latency_ms: float                   # end-to-end DSP latency
    quality_score: float                # 0–1 signal quality


class DSPPipeline:
    """
    End-to-end DSP pipeline: raw EEG → feature vector.

    Design decision: each stage is independently configurable so
    the pipeline can be paused, reconfigured, and restarted without
    restarting the entire platform (GNU Radio live-tuning equivalent).
    """

    SAMPLE_RATE = 250       # Hz
    EPOCH_DURATION_P300 = 0.8      # seconds (0–800 ms)
    EPOCH_DURATION_MI = 4.0        # seconds (motor imagery)
    BUFFER_SECONDS = 5.0           # rolling input buffer

    def __init__(
        self,
        n_channels: int = 8,
        notch_hz: float = 50.0,        # 50 Hz EU, use 60.0 for US
        bandpass_low: float = 1.0,
        bandpass_high: float = 40.0,
        epoch_type: str = "p300",      # or "motor_imagery"
        on_features: Optional[Callable] = None,
    ):
        self.n_channels = n_channels
        self.epoch_type = epoch_type
        self.on_features = on_features

        # Buffer: rolling window of raw samples (n_channels × buffer_samples)
        self._buffer_size = int(self.SAMPLE_RATE * self.BUFFER_SECONDS)
        self._buffer = np.zeros((n_channels, self._buffer_size), dtype=np.float32)
        self._buffer_idx = 0
        self._samples_received = 0

        # Epoch size based on type
        epoch_dur = self.EPOCH_DURATION_P300 if epoch_type == "p300" else self.EPOCH_DURATION_MI
        self._epoch_samples = int(self.SAMPLE_RATE * epoch_dur)

        # Initialise all processing stages
        logger.info("Initialising DSP pipeline stages...")
        self.bandpass = BandpassFilter(
            low_hz=bandpass_low,
            high_hz=bandpass_high,
            sample_rate=self.SAMPLE_RATE,
            order=4,
        )
        self.notch = NotchFilter(
            notch_hz=notch_hz,
            sample_rate=self.SAMPLE_RATE,
        )
        self.ica = ICARemoval(n_components=n_channels)
        self.asr = ASRCleaning(sample_rate=self.SAMPLE_RATE)
        self.epoch_extractor = EpochExtractor(
            sample_rate=self.SAMPLE_RATE,
            epoch_type=epoch_type,
        )
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.SAMPLE_RATE,
            n_channels=n_channels,
        )

        self._latencies: list[float] = []
        logger.success("DSP pipeline ready")

    def process_sample(self, sample: EEGSample) -> Optional[ProcessedEEG]:
        """
        Add one sample to rolling buffer; process epoch when full.

        Returns ProcessedEEG when a complete epoch is ready, else None.
        Called at 250 Hz — must be fast.
        """
        t_start = time.perf_counter()

        # Update rolling buffer (circular)
        self._buffer[:, self._buffer_idx % self._buffer_size] = sample.channels
        self._buffer_idx += 1
        self._samples_received += 1

        # Only process when we have a full epoch
        if self._samples_received < self._epoch_samples:
            return None
        # Slide epoch: take last epoch_samples from buffer
        if self._buffer_idx >= self._buffer_size:
            epoch_raw = np.roll(self._buffer, -self._buffer_idx, axis=1)[:, -self._epoch_samples:]
        else:
            epoch_raw = self._buffer[:, :self._buffer_idx]
            if epoch_raw.shape[1] < self._epoch_samples:
                return None
            epoch_raw = epoch_raw[:, -self._epoch_samples:]

        # ── Stage 1: Bandpass filter ───────────────────────────────
        filtered = self.bandpass.filter(epoch_raw)

        # ── Stage 2: Notch filter ──────────────────────────────────
        notched = self.notch.filter(filtered)

        # ── Stage 3: ICA artifact removal ─────────────────────────
        ica_clean = self.ica.remove_artifacts(notched)

        # ── Stage 4: ASR cleaning ──────────────────────────────────
        asr_clean = self.asr.clean(ica_clean)

        # ── Stage 5 + 6: Epoch extraction + feature vector ────────
        epoch = self.epoch_extractor.extract(asr_clean, sample.timestamp)
        features = self.feature_extractor.extract(epoch)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        self._latencies.append(latency_ms)

        if latency_ms > 10.0:
            logger.warning(f"DSP latency exceeded 10ms budget: {latency_ms:.1f}ms")

        quality = self._compute_quality(asr_clean)

        result = ProcessedEEG(
            timestamp=sample.timestamp,
            raw_channels=epoch_raw,
            filtered_channels=notched,
            clean_channels=asr_clean,
            feature_vector=features,
            epoch_type=self.epoch_type,
            latency_ms=latency_ms,
            quality_score=quality,
        )

        if self.on_features:
            self.on_features(result)

        return result

    def get_latency_stats(self) -> dict:
        """Return DSP latency statistics for monitoring."""
        if not self._latencies:
            return {}
        arr = np.array(self._latencies[-1000:])    # last 1000 epochs
        return {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
            "budget_violations": int(np.sum(arr > 10.0)),
        }

    @staticmethod
    def _compute_quality(clean: np.ndarray) -> float:
        """
        Simple signal quality score [0, 1] based on amplitude range.
        Valid EEG: 1–100 µV peak-to-peak per channel.
        """
        pp = np.ptp(clean, axis=1)    # peak-to-peak per channel
        good = np.sum((pp > 1.0) & (pp < 150.0))
        return good / clean.shape[0]