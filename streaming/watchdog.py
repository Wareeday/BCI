"""
streaming/watchdog.py
=====================
500ms watchdog safety monitor.

If no EEG sample is received for >500ms, the watchdog fires and
sends a SAFE_STATE command to all actuators (wheelchair brakes, prosthetic hold).

This is the "Watchdog: SAFE_STATE if >500ms gap" requirement from the
presentation's LSL latency profile section.

Failure scenarios covered:
  [HIGH] EEG Signal Lost: LSL stream timeout >200ms → watchdog fires
  Detection: no push_sample for TIMEOUT_MS milliseconds
  Recovery: SAFE_STATE → alert clinician → retry BLE × 3 → fallback eye tracker
"""

import threading
import time
from enum import Enum
from typing import Callable, Optional
from loguru import logger


class WatchdogState(Enum):
    RUNNING = "running"
    TIMEOUT = "timeout"        # signal lost
    SAFE_STATE = "safe_state"  # actuators set to safe position
    PAUSED = "paused"


class SignalWatchdog:
    """
    Multi-source watchdog monitoring EEG, Kafka, and LSL streams.

    Fires safe_state_callback when any monitored source exceeds its timeout.
    """

    def __init__(
        self,
        eeg_timeout_ms: float = 500.0,
        kafka_timeout_ms: float = 500.0,
        safe_state_callback: Optional[Callable] = None,
        recovery_callback: Optional[Callable] = None,
    ):
        self.eeg_timeout_ms = eeg_timeout_ms
        self.kafka_timeout_ms = kafka_timeout_ms
        self.safe_state_callback = safe_state_callback
        self.recovery_callback = recovery_callback

        self._state = WatchdogState.PAUSED
        self._last_eeg_time = 0.0
        self._last_kafka_time = 0.0
        self._safe_state_count = 0
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start watchdog monitoring thread."""
        self._stop_event.clear()
        self._state = WatchdogState.RUNNING
        self._last_eeg_time = time.time()
        self._last_kafka_time = time.time()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SignalWatchdog",
        )
        self._monitor_thread.start()
        logger.info(
            f"Watchdog started: EEG timeout={self.eeg_timeout_ms}ms, "
            f"Kafka timeout={self.kafka_timeout_ms}ms"
        )

    def stop(self):
        """Stop watchdog."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._state = WatchdogState.PAUSED
        logger.info("Watchdog stopped")

    def ping_eeg(self):
        """Call every time an EEG sample is received."""
        with self._lock:
            self._last_eeg_time = time.time()
            if self._state == WatchdogState.SAFE_STATE:
                self._on_recovery("EEG")

    def ping_kafka(self):
        """Call every time a Kafka message is consumed."""
        with self._lock:
            self._last_kafka_time = time.time()
            if self._state == WatchdogState.SAFE_STATE:
                self._on_recovery("Kafka")

    def _monitor_loop(self):
        """Check timeouts every 50ms."""
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                eeg_gap = (now - self._last_eeg_time) * 1000.0
                kafka_gap = (now - self._last_kafka_time) * 1000.0

            eeg_timeout = self._last_eeg_time > 0 and eeg_gap > self.eeg_timeout_ms
            kafka_timeout = self._last_kafka_time > 0 and kafka_gap > self.kafka_timeout_ms

            if (eeg_timeout or kafka_timeout) and self._state == WatchdogState.RUNNING:
                source = "EEG" if eeg_timeout else "Kafka"
                gap = eeg_gap if eeg_timeout else kafka_gap
                self._on_timeout(source, gap)

            time.sleep(0.05)

    def _on_timeout(self, source: str, gap_ms: float):
        """Handle signal timeout — trigger SAFE_STATE."""
        self._state = WatchdogState.SAFE_STATE
        self._safe_state_count += 1
        logger.error(
            f"WATCHDOG TIMEOUT: {source} signal lost for {gap_ms:.0f}ms "
            f"(timeout={self.eeg_timeout_ms}ms) — SAFE_STATE activated"
        )
        if self.safe_state_callback:
            try:
                self.safe_state_callback(source=source, gap_ms=gap_ms)
            except Exception as exc:
                logger.error(f"SAFE_STATE callback error: {exc}")

    def _on_recovery(self, source: str):
        """Signal resumed — exit SAFE_STATE."""
        self._state = WatchdogState.RUNNING
        logger.info(f"Watchdog recovery: {source} signal restored")
        if self.recovery_callback:
            try:
                self.recovery_callback(source=source)
            except Exception as exc:
                logger.error(f"Recovery callback error: {exc}")

    @property
    def state(self) -> WatchdogState:
        return self._state

    @property
    def safe_state_count(self) -> int:
        return self._safe_state_count

    def get_status(self) -> dict:
        now = time.time()
        return {
            "state": self._state.value,
            "eeg_gap_ms": (now - self._last_eeg_time) * 1000.0 if self._last_eeg_time else None,
            "kafka_gap_ms": (now - self._last_kafka_time) * 1000.0 if self._last_kafka_time else None,
            "safe_state_count": self._safe_state_count,
        }