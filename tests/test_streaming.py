"""
tests/test_streaming.py
========================
Unit tests for Kafka producer, consumer, and watchdog.
"""

import time
import pytest
import numpy as np


class TestKafkaProducer:
    def test_producer_initialises_without_broker(self, mock_kafka_producer):
        assert not mock_kafka_producer._connected
        assert mock_kafka_producer._messages_sent == 0

    def test_publish_raw_sample_no_crash(self, mock_kafka_producer):
        from acquisition.openbci_board import EEGSample
        sample = EEGSample(
            timestamp=time.time(),
            channels=np.random.randn(8).astype(np.float32),
            sample_id=1,
        )
        mock_kafka_producer.publish_raw_sample(sample)   # should not raise

    def test_publish_features_no_crash(self, mock_kafka_producer):
        features = np.random.randn(56).astype(np.float32)
        mock_kafka_producer.publish_features(
            timestamp=time.time(),
            feature_vector=features,
            epoch_type="motor_imagery",
        )

    def test_publish_command_no_crash(self, mock_kafka_producer):
        mock_kafka_producer.publish_command(
            timestamp=time.time(),
            command="left",
            confidence=0.91,
            model_used="cnn",
        )

    def test_stats_dict_keys(self, mock_kafka_producer):
        stats = mock_kafka_producer.stats
        assert "connected" in stats
        assert "messages_sent" in stats
        assert "errors" in stats


class TestSignalWatchdog:
    def test_watchdog_healthy_after_ping(self):
        from streaming.watchdog import SignalWatchdog
        wd = SignalWatchdog(eeg_timeout_ms=500.0)
        wd.start()
        wd.ping_eeg()
        assert wd.is_healthy
        wd.stop()

    def test_watchdog_status_has_required_keys(self):
        from streaming.watchdog import SignalWatchdog
        wd = SignalWatchdog()
        wd.start()
        status = wd.get_status()
        for key in ("state", "safe_state_count"):
            assert key in status
        wd.stop()

    def test_kafka_ping_updates_time(self):
        from streaming.watchdog import SignalWatchdog
        wd = SignalWatchdog(kafka_timeout_ms=500.0)
        wd.start()
        before = wd._last_kafka_time
        time.sleep(0.05)
        wd.ping_kafka()
        assert wd._last_kafka_time >= before
        wd.stop()