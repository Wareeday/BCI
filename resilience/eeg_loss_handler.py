"""
resilience/eeg_loss_handler.py
================================
[HIGH] EEG Signal Loss recovery handler.

Risk Matrix entry (ISO 14971 aligned):
  Scenario:   EEG signal lost (electrode detach / BLE drop)
  Likelihood: Medium (5–8% error)
  Severity:   CRITICAL
  Risk Level: HIGH
  Mitigation: Confidence threshold 85% + dual-channel confirmation
              + deadman watchdog 500ms

Recovery steps (from presentation failure scenario):
  1. SAFE_STATE to actuator (immediate, <10ms)
  2. Alert clinician dashboard
  3. Retry BLE reconnect ×3
  4. If fail: fallback to eye tracker

Detection: LSL stream timeout >200ms → watchdog fires
"""

import threading
import time
from enum import Enum
from typing import Callable, Optional
from loguru import logger


class RecoveryState(Enum):
    NORMAL = "normal"
    SIGNAL_LOST = "signal_lost"
    RETRYING = "retrying"
    FALLBACK = "fallback"       # eye tracker mode
    FAILED = "failed"


class EEGLossHandler:
    """
    Handles EEG signal loss with automatic recovery.

    Three-stage recovery:
      Stage 1: immediate SAFE_STATE (< 10ms)
      Stage 2: retry BLE reconnect × 3 (up to 15 s)
      Stage 3: fallback to eye tracker or manual mode
    """

    MAX_RETRIES = 3
    RETRY_DELAY_S = 5.0
    BLE_RECONNECT_TIMEOUT_S = 10.0

    def __init__(
        self,
        safe_state_callback: Optional[Callable] = None,
        alert_clinician_callback: Optional[Callable] = None,
        reconnect_callback: Optional[Callable] = None,
        fallback_callback: Optional[Callable] = None,
        audit_logger=None,
    ):
        self.safe_state_cb = safe_state_callback
        self.alert_cb = alert_clinician_callback
        self.reconnect_cb = reconnect_callback
        self.fallback_cb = fallback_callback
        self.audit = audit_logger

        self.state = RecoveryState.NORMAL
        self._retry_count = 0
        self._recovery_thread: Optional[threading.Thread] = None
        self._signal_loss_count = 0
        self._last_loss_time = 0.0

    def on_signal_lost(self, gap_ms: float, source: str = "EEG"):
        """
        Called when watchdog detects signal loss.
        Executes recovery sequence in background thread.
        """
        if self.state != RecoveryState.NORMAL:
            return   # recovery already in progress

        self._signal_loss_count += 1
        self._last_loss_time = time.time()
        self.state = RecoveryState.SIGNAL_LOST

        logger.error(
            f"[HIGH] EEG signal lost: gap={gap_ms:.0f}ms, source={source}. "
            f"Starting recovery sequence #{self._signal_loss_count}."
        )

        if self.audit:
            self.audit.log(
                event_type="signal_lost",
                severity="ERROR",
                details={
                    "gap_ms": gap_ms,
                    "source": source,
                    "recovery_attempt": self._signal_loss_count,
                },
            )

        # Step 1: Immediate SAFE_STATE
        self._activate_safe_state(source)

        # Steps 2–4 in background thread
        self._recovery_thread = threading.Thread(
            target=self._recovery_sequence,
            daemon=True,
            name="EEGLossRecovery",
        )
        self._recovery_thread.start()

    def on_signal_restored(self):
        """Called when EEG signal resumes."""
        self._retry_count = 0
        self.state = RecoveryState.NORMAL
        logger.info("EEG signal restored — recovery complete")
        if self.audit:
            self.audit.log(
                event_type="signal_restored",
                details={"downtime_s": time.time() - self._last_loss_time},
            )

    def _activate_safe_state(self, source: str):
        """Step 1: immediately halt all actuators."""
        logger.warning("Recovery Step 1: SAFE_STATE activated — halting all actuators")
        if self.safe_state_cb:
            try:
                self.safe_state_cb(reason=f"EEG signal lost ({source})")
            except Exception as exc:
                logger.error(f"SAFE_STATE callback error: {exc}")

    def _recovery_sequence(self):
        """Step 2: alert clinician. Step 3: retry BLE × 3. Step 4: fallback."""

        # Step 2: Alert clinician dashboard
        logger.warning("Recovery Step 2: Alerting clinician dashboard")
        if self.alert_cb:
            try:
                self.alert_cb(
                    message="EEG signal lost — attempting reconnection",
                    severity="HIGH",
                )
            except Exception as exc:
                logger.error(f"Alert callback error: {exc}")

        # Step 3: Retry BLE reconnect ×3
        self.state = RecoveryState.RETRYING
        for attempt in range(1, self.MAX_RETRIES + 1):
            logger.info(
                f"Recovery Step 3: BLE reconnect attempt {attempt}/{self.MAX_RETRIES}"
            )
            time.sleep(self.RETRY_DELAY_S)

            if self.reconnect_cb:
                try:
                    success = self.reconnect_cb()
                    if success:
                        logger.success(f"BLE reconnect succeeded on attempt {attempt}")
                        self.on_signal_restored()
                        return
                except Exception as exc:
                    logger.error(f"Reconnect attempt {attempt} failed: {exc}")

            self._retry_count = attempt

        # Step 4: Fallback to eye tracker
        logger.warning(
            "Recovery Step 4: All BLE reconnects failed — "
            "activating eye tracker fallback mode"
        )
        self.state = RecoveryState.FALLBACK
        if self.fallback_cb:
            try:
                self.fallback_cb()
            except Exception as exc:
                logger.error(f"Fallback activation failed: {exc}")
                self.state = RecoveryState.FAILED

        if self.audit:
            self.audit.log(
                event_type="recovery_failed",
                severity="CRITICAL",
                details={
                    "retries": self.MAX_RETRIES,
                    "fallback_activated": self.state == RecoveryState.FALLBACK,
                },
            )

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "signal_loss_count": self._signal_loss_count,
            "retry_count": self._retry_count,
            "last_loss_time": self._last_loss_time,
        }