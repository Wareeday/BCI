"""
resilience/safe_state.py
=========================
Global SAFE_STATE coordinator.

SAFE_STATE definition:
  All actuators (wheelchair, prosthetic) brought to a safe resting
  position within 10ms of activation. No new commands accepted.

Triggers:
  - EEG signal lost >500ms
  - Kafka stream silent >500ms
  - CNN inference timeout >100ms × 3
  - Manual clinician override
  - System shutdown

This is the central safety coordinator that all other components call.
"""

import threading
import time
from typing import Optional, Callable, List
from loguru import logger


class SafeStateCoordinator:
    """
    Single-point coordinator for SAFE_STATE activation.

    All safety-critical paths call this coordinator rather than
    directly calling device APIs, ensuring consistent behaviour.
    """

    def __init__(self, audit_logger=None):
        self.audit = audit_logger
        self._active = False
        self._reason = ""
        self._activation_time = 0.0
        self._activation_count = 0
        self._lock = threading.Lock()

        # Registered actuator stop callbacks
        self._stop_callbacks: List[Callable] = []
        # Registered alert callbacks (clinician dashboard)
        self._alert_callbacks: List[Callable] = []

    def register_stop_callback(self, callback: Callable):
        """Register an actuator that should stop on SAFE_STATE."""
        self._stop_callbacks.append(callback)

    def register_alert_callback(self, callback: Callable):
        """Register a clinician alert that fires on SAFE_STATE."""
        self._alert_callbacks.append(callback)

    def activate(self, reason: str, source: str = "unknown"):
        """
        Activate SAFE_STATE — halt all actuators immediately.
        Thread-safe, idempotent.
        """
        with self._lock:
            if self._active:
                return
            self._active = True
            self._reason = reason
            self._activation_time = time.time()
            self._activation_count += 1

        logger.critical(
            f"SAFE_STATE ACTIVATED #{self._activation_count}: "
            f"reason='{reason}', source={source}"
        )

        # Fire all stop callbacks simultaneously
        threads = []
        for cb in self._stop_callbacks:
            t = threading.Thread(target=self._call_safe, args=(cb, reason))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=0.05)   # 50ms max per callback

        # Alert clinician
        for cb in self._alert_callbacks:
            try:
                cb(reason=reason, source=source, timestamp=self._activation_time)
            except Exception as exc:
                logger.error(f"Alert callback error: {exc}")

        if self.audit:
            self.audit.log(
                event_type="safe_state",
                severity="CRITICAL",
                details={
                    "reason": reason,
                    "source": source,
                    "activation_count": self._activation_count,
                },
            )

    def deactivate(self, authorised_by: str = "system"):
        """Deactivate SAFE_STATE — resume normal operation."""
        with self._lock:
            if not self._active:
                return
            duration = time.time() - self._activation_time
            self._active = False
            self._reason = ""

        logger.info(
            f"SAFE_STATE deactivated by {authorised_by} "
            f"(was active for {duration:.2f}s)"
        )

        if self.audit:
            self.audit.log(
                event_type="safe_state_deactivated",
                details={
                    "authorised_by": authorised_by,
                    "duration_s": duration,
                },
            )

    def _call_safe(self, callback: Callable, reason: str):
        try:
            callback(reason=reason)
        except Exception as exc:
            logger.error(f"Safe stop callback error: {exc}")

    @property
    def is_active(self) -> bool:
        return self._active

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "reason": self._reason,
            "activation_count": self._activation_count,
            "duration_s": (
                time.time() - self._activation_time if self._active else 0.0
            ),
        }