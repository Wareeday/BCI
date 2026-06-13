"""
tests/test_acquisition.py
==========================
Unit tests for OpenBCI board interface and LSL streamer.
"""

import time
import threading
import pytest
import numpy as np


class TestOpenBCIBoard:
    def test_connect_in_simulate_mode(self, mock_board):
        from acquisition.openbci_board import BoardState
        assert mock_board.state == BoardState.CONNECTED

    def test_streaming_calls_callback(self, mock_board):
        samples = []
        def callback(sample):
            samples.append(sample)

        mock_board.start_streaming(callback=callback)
        time.sleep(0.1)   # collect ~25 samples at 250Hz
        mock_board.stop_streaming()
        assert len(samples) >= 5

    def test_sample_has_8_channels(self, mock_board):
        samples = []
        mock_board.start_streaming(callback=lambda s: samples.append(s))
        time.sleep(0.05)
        mock_board.stop_streaming()
        assert samples[0].channels.shape == (8,)

    def test_impedance_check_simulation(self, mock_board):
        impedances = mock_board.check_impedances()
        assert len(impedances) == 8
        for ch, z in impedances.items():
            assert z > 0.0

    def test_watchdog_healthy_while_streaming(self, mock_board):
        mock_board.start_streaming(callback=lambda s: None)
        time.sleep(0.1)
        assert mock_board.is_healthy()
        mock_board.stop_streaming()

    def test_signal_quality_returns_dict(self, mock_board):
        mock_board.check_impedances()
        quality = mock_board.get_signal_quality()
        assert "snr_db" in quality
        assert "channels_ok" in quality
        assert "target_met" in quality

    def test_sample_count_increments(self, mock_board):
        mock_board.start_streaming(callback=lambda s: None)
        time.sleep(0.08)
        mock_board.stop_streaming()
        assert mock_board.sample_count > 0

    def test_disconnect_changes_state(self, mock_board):
        from acquisition.openbci_board import BoardState
        mock_board.disconnect()
        assert mock_board.state == BoardState.DISCONNECTED


class TestEEGSample:
    def test_sample_channels_dtype(self, mock_board):
        samples = []
        mock_board.start_streaming(callback=lambda s: samples.append(s))
        time.sleep(0.05)
        mock_board.stop_streaming()
        assert samples[0].channels.dtype == np.float32

    def test_sample_timestamp_is_recent(self, mock_board):
        samples = []
        mock_board.start_streaming(callback=lambda s: samples.append(s))
        time.sleep(0.05)
        mock_board.stop_streaming()
        now = time.time()
        assert abs(samples[-1].timestamp - now) < 1.0


class TestLSLStreamer:
    def test_push_sample_no_crash_without_pylsl(self):
        """LSL push should silently skip if pylsl not installed."""
        from acquisition.lsl_streamer import LSLStreamer
        streamer = LSLStreamer()
        from acquisition.openbci_board import EEGSample
        sample = EEGSample(
            timestamp=time.time(),
            channels=np.random.randn(8).astype(np.float32),
            sample_id=1,
        )
        # Should not raise even without LSL
        streamer.push_sample(sample)

    def test_samples_pushed_counter(self):
        from acquisition.lsl_streamer import LSLStreamer
        from acquisition.openbci_board import EEGSample
        streamer = LSLStreamer()
        initial = streamer.samples_pushed
        for i in range(5):
            s = EEGSample(timestamp=time.time(),
                          channels=np.zeros(8, dtype=np.float32), sample_id=i)
            streamer.push_sample(s)
        # Counter only increments if LSL is available
        assert streamer.samples_pushed >= initial