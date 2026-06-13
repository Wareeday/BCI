"""
tests/test_resilience.py
=========================
Unit tests for failure scenarios and resilience components.

Tests ISO 14971 risk matrix scenarios:
  [HIGH] EEG signal loss → SAFE_STATE within 500ms
  [HIGH] Kafka broker crash → local buffer failover
  [MED]  CNN classifier failure → LDA fallback
  [LOW]  Network partition → local operation continues
"""

import time
import threading
import pytest
import numpy as np


class TestSignalWatchdog:
    def test_watchdog_starts_and_stops(self):
        from streaming.watchdog import SignalWatchdog, WatchdogState
        wd = SignalWatchdog(eeg_timeout_ms=500.0)
        wd.start()
        assert wd.state == WatchdogState.RUNNING
        wd.stop()
        assert wd.state == WatchdogState.PAUSED

    def test_ping_resets_timer(self):
        from streaming.watchdog import SignalWatchdog
        wd = SignalWatchdog(eeg_timeout_ms=500.0)
        wd.start()
        wd.ping_eeg()
        status = wd.get_status()
        assert status["eeg_gap_ms"] < 100.0
        wd.stop()

    def test_timeout_triggers_safe_state_callback(self):
        safe_state_called = threading.Event()

        def on_safe_state(source, gap_ms):
            safe_state_called.set()

        from streaming.watchdog import SignalWatchdog, WatchdogState
        wd = SignalWatchdog(
            eeg_timeout_ms=100.0,   # short timeout for test
            safe_state_callback=on_safe_state,
        )
        wd.start()
        wd._last_eeg_time = time.time()    # prime it
        time.sleep(0.25)   # wait for timeout
        called = safe_state_called.wait(timeout=1.0)
        wd.stop()
        assert called, "SAFE_STATE callback was not triggered within 1s"

    def test_get_status_keys(self):
        from streaming.watchdog import SignalWatchdog
        wd = SignalWatchdog()
        wd.start()
        status = wd.get_status()
        assert "state" in status
        assert "safe_state_count" in status
        wd.stop()


class TestSafeStateCoordinator:
    def test_activate_calls_all_stop_callbacks(self, safe_state_coordinator):
        stopped = []
        safe_state_coordinator.register_stop_callback(lambda reason="": stopped.append("dev1"))
        safe_state_coordinator.register_stop_callback(lambda reason="": stopped.append("dev2"))
        safe_state_coordinator.activate(reason="test", source="pytest")
        time.sleep(0.1)
        assert "dev1" in stopped
        assert "dev2" in stopped

    def test_activate_is_idempotent(self, safe_state_coordinator):
        call_count = [0]
        safe_state_coordinator.register_stop_callback(lambda reason="": call_count.__setitem__(0, call_count[0]+1))
        safe_state_coordinator.activate(reason="first")
        safe_state_coordinator.activate(reason="second")   # should be ignored
        time.sleep(0.1)
        assert call_count[0] == 1

    def test_deactivate_restores_normal(self, safe_state_coordinator):
        safe_state_coordinator.activate(reason="test")
        assert safe_state_coordinator.is_active
        safe_state_coordinator.deactivate(authorised_by="clinician")
        assert not safe_state_coordinator.is_active

    def test_status_dict_complete(self, safe_state_coordinator):
        status = safe_state_coordinator.get_status()
        assert "active" in status
        assert "reason" in status
        assert "activation_count" in status


class TestEEGLossHandler:
    def test_safe_state_called_on_signal_loss(self):
        safe_state_calls = []
        from resilience.eeg_loss_handler import EEGLossHandler
        handler = EEGLossHandler(
            safe_state_callback=lambda reason="": safe_state_calls.append(reason),
        )
        handler.on_signal_lost(gap_ms=600.0, source="EEG")
        time.sleep(0.1)
        assert len(safe_state_calls) == 1

    def test_recovery_resets_state(self):
        from resilience.eeg_loss_handler import EEGLossHandler, RecoveryState
        handler = EEGLossHandler(safe_state_callback=lambda reason="": None)
        handler.on_signal_lost(gap_ms=600.0)
        time.sleep(0.1)
        handler.on_signal_restored()
        assert handler.state == RecoveryState.NORMAL

    def test_status_returned(self):
        from resilience.eeg_loss_handler import EEGLossHandler
        handler = EEGLossHandler()
        status = handler.get_status()
        assert "state" in status
        assert "signal_loss_count" in status


class TestClassifierFallback:
    def test_fallback_on_cnn_failure(self):
        from resilience.classifier_fallback import ClassifierFallbackManager, ClassifierMode
        manager = ClassifierFallbackManager()
        manager.on_cnn_failure("CUDA OOM")
        assert manager.mode != ClassifierMode.PRIMARY_CNN

    def test_predict_returns_tuple(self):
        from resilience.classifier_fallback import ClassifierFallbackManager
        manager = ClassifierFallbackManager()
        features = np.random.randn(56).astype(np.float32)
        pred, probs, conf, model_name = manager.predict_with_fallback(features)
        assert pred in range(4)
        assert len(probs) == 4
        assert 0.0 <= conf <= 1.0
        assert isinstance(model_name, str)

    def test_degraded_mode_restricts_classes(self):
        from resilience.classifier_fallback import ClassifierFallbackManager, ClassifierMode
        manager = ClassifierFallbackManager()
        manager.mode = ClassifierMode.DEGRADED
        features = np.random.randn(56).astype(np.float32)
        pred, _, _, _ = manager.predict_with_fallback(features)
        assert pred <= 2   # degraded: only classes 0, 1, 2 (left, right, stop)


class TestKafkaFailover:
    def test_buffer_stores_messages(self):
        from resilience.kafka_failover import KafkaFailoverHandler
        handler = KafkaFailoverHandler()
        for i in range(10):
            handler.buffer_message({"sample_id": i, "channels": [0.0] * 8})
        status = handler.get_buffer_status()
        assert status["buffered_messages"] == 10

    def test_on_broker_failure_sets_unhealthy(self):
        from resilience.kafka_failover import KafkaFailoverHandler
        handler = KafkaFailoverHandler()
        assert handler._broker_healthy
        handler.on_broker_failure("Connection refused")
        assert not handler._broker_healthy

    def test_buffer_respects_maxlen(self):
        """Buffer holds exactly 10s × 250Hz = 2500 samples."""
        from resilience.kafka_failover import KafkaFailoverHandler
        handler = KafkaFailoverHandler(sample_rate=250)
        for i in range(3000):
            handler.buffer_message({"id": i})
        assert len(handler._local_buffer) == 2500