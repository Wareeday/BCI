"""
tests/test_dsp.py
==================
Unit tests for DSP pipeline components.

Tests:
  - Bandpass filter frequency response
  - Notch filter 50Hz attenuation
  - ICA artifact removal
  - ASR cleaning
  - Epoch extraction
  - Feature vector shape and range
  - DSP latency budget (<10ms)
"""

import time
import pytest
import numpy as np
from scipy.signal import welch


class TestBandpassFilter:
    def test_filter_output_shape(self, synthetic_epoch):
        from dsp.bandpass_filter import BandpassFilter
        f = BandpassFilter(low_hz=1.0, high_hz=40.0, sample_rate=250.0)
        out = f.filter(synthetic_epoch)
        assert out.shape == synthetic_epoch.shape

    def test_filter_attenuates_high_frequency(self):
        from dsp.bandpass_filter import BandpassFilter
        f = BandpassFilter(low_hz=1.0, high_hz=40.0, sample_rate=250.0)
        t = np.linspace(0, 0.8, 200)
        # 60 Hz signal should be attenuated
        data = np.tile(np.sin(2 * np.pi * 60 * t), (8, 1)).astype(np.float32)
        out = f.filter(data)
        assert np.std(out) < np.std(data) * 0.1   # >90% attenuation

    def test_filter_passes_alpha(self):
        from dsp.bandpass_filter import BandpassFilter
        f = BandpassFilter(low_hz=1.0, high_hz=40.0, sample_rate=250.0)
        t = np.linspace(0, 0.8, 200)
        # 10 Hz alpha should pass through
        data = np.tile(np.sin(2 * np.pi * 10 * t), (8, 1)).astype(np.float32) * 15.0
        out = f.filter(data)
        assert np.std(out) > 5.0   # signal preserved

    def test_output_dtype(self, synthetic_epoch):
        from dsp.bandpass_filter import BandpassFilter
        f = BandpassFilter()
        out = f.filter(synthetic_epoch)
        assert out.dtype == np.float32


class TestNotchFilter:
    def test_notch_attenuates_50hz(self):
        from dsp.notch_filter import NotchFilter
        nf = NotchFilter(notch_hz=50.0, sample_rate=250.0)
        t = np.linspace(0, 0.8, 200)
        data = np.tile(np.sin(2 * np.pi * 50 * t), (8, 1)).astype(np.float32) * 10.0
        out = nf.filter(data)
        assert np.std(out) < np.std(data) * 0.3

    def test_notch_preserves_alpha(self):
        from dsp.notch_filter import NotchFilter
        nf = NotchFilter(notch_hz=50.0, sample_rate=250.0)
        t = np.linspace(0, 0.8, 200)
        data = np.tile(np.sin(2 * np.pi * 10 * t), (8, 1)).astype(np.float32) * 10.0
        out = nf.filter(data)
        assert np.std(out) > 3.0

    def test_output_shape(self, synthetic_epoch):
        from dsp.notch_filter import NotchFilter
        nf = NotchFilter()
        out = nf.filter(synthetic_epoch)
        assert out.shape == synthetic_epoch.shape


class TestASRCleaning:
    def test_removes_high_amplitude_artifact(self):
        from dsp.asr_cleaning import ASRCleaning
        asr = ASRCleaning(cutoff=5.0)
        data = np.random.randn(8, 200).astype(np.float32) * 3.0
        data[3, 50:70] += 500.0   # inject artifact
        cleaned = asr.clean(data)
        assert np.max(np.abs(cleaned[3, 50:70])) < 100.0

    def test_output_shape(self, synthetic_epoch):
        from dsp.asr_cleaning import ASRCleaning
        asr = ASRCleaning()
        out = asr.clean(synthetic_epoch)
        assert out.shape == synthetic_epoch.shape

    def test_clean_data_unchanged(self):
        from dsp.asr_cleaning import ASRCleaning
        asr = ASRCleaning(cutoff=10.0)
        data = np.random.randn(8, 200).astype(np.float32) * 2.0
        cleaned = asr.clean(data)
        assert np.allclose(data, cleaned, atol=1e-5)


class TestEpochExtraction:
    def test_p300_epoch_shape(self, synthetic_epoch):
        from dsp.epoch_extraction import EpochExtractor
        extractor = EpochExtractor(sample_rate=250.0, epoch_type="p300")
        epoch = extractor.extract(synthetic_epoch, timestamp=0.0)
        assert epoch.data.shape == (8, 200)

    def test_baseline_correction_applied(self, synthetic_epoch):
        from dsp.epoch_extraction import EpochExtractor
        extractor = EpochExtractor(sample_rate=250.0, epoch_type="p300")
        epoch = extractor.extract(synthetic_epoch, timestamp=1.0)
        # Baseline period mean should be near zero after correction
        baseline_mean = np.mean(epoch.data[:, :50])
        assert abs(baseline_mean) < 5.0

    def test_label_preserved(self, synthetic_epoch):
        from dsp.epoch_extraction import EpochExtractor
        extractor = EpochExtractor(sample_rate=250.0, epoch_type="p300")
        epoch = extractor.extract(synthetic_epoch, timestamp=0.0, label=2)
        assert epoch.label == 2


class TestFeatureExtractor:
    def test_feature_vector_length(self, synthetic_epoch):
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch
        extractor = FeatureExtractor(sample_rate=250.0, n_channels=8)
        epoch = Epoch(data=synthetic_epoch, timestamp=0.0, epoch_type="p300")
        features = extractor.extract(epoch)
        assert features.shape == (56,)   # 8ch×4bands PSD + 8ch×3 time = 32+24=56

    def test_feature_dtype(self, synthetic_epoch):
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch
        extractor = FeatureExtractor(sample_rate=250.0, n_channels=8)
        epoch = Epoch(data=synthetic_epoch, timestamp=0.0, epoch_type="p300")
        features = extractor.extract(epoch)
        assert features.dtype == np.float32

    def test_features_finite(self, synthetic_epoch):
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch
        extractor = FeatureExtractor(sample_rate=250.0, n_channels=8)
        epoch = Epoch(data=synthetic_epoch, timestamp=0.0, epoch_type="p300")
        features = extractor.extract(epoch)
        assert np.all(np.isfinite(features))


class TestDSPLatencyBudget:
    """Critical: DSP pipeline must complete in <10ms."""

    def test_full_pipeline_under_10ms(self, synthetic_epoch):
        from dsp.bandpass_filter import BandpassFilter
        from dsp.notch_filter import NotchFilter
        from dsp.asr_cleaning import ASRCleaning
        from dsp.feature_vector import FeatureExtractor
        from dsp.epoch_extraction import Epoch

        bp = BandpassFilter()
        nf = NotchFilter()
        asr = ASRCleaning()
        fe = FeatureExtractor(sample_rate=250.0, n_channels=8)

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            filtered = bp.filter(synthetic_epoch)
            notched = nf.filter(filtered)
            cleaned = asr.clean(notched)
            epoch_obj = Epoch(data=cleaned, timestamp=0.0, epoch_type="p300")
            _ = fe.extract(epoch_obj)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        # Allow 6ms for compute (acquisition 4ms is hardware, not measured here)
        assert mean_ms < 6.0, f"DSP mean={mean_ms:.2f}ms exceeds 6ms budget"
        assert p95_ms < 9.0, f"DSP p95={p95_ms:.2f}ms exceeds 9ms budget"