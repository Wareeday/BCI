"""
devices/safety_watchdog.py
============================
Device-level safety watchdog — monitors all actuators.

Distinct from streaming/watchdog.py (which monitors signal streams).
This watchdog monitors the actuator devices themselves:
  - Wheelchair motor encoder feedback
  - Prosthetic servo position feedback
  - TTS engine health

If any device stops responding → SAFE_STATE for that device only
(not full system SAFE_STATE unless EEG is also lost).

ISO 14155 §14: Any unintended movement → SAE. This watchdog is
the last line of defence against runaway actuators.
"""
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional
from loguru import logger


class DeviceHealth(Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"    # responding but with errors
    TIMEOUT  = "timeout"     # no heartbeat within threshold
    STOPPED  = "stopped"     # safe-stated


@dataclass
class DeviceStatus:
    name: str
    health: DeviceHealth = DeviceHealth.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    timeout_threshold_s: float = 2.0
    error_count: int = 0
    safe_state_count: int = 0


class DeviceSafetyWatchdog:
    """
    Monitors heartbeats from all connected actuator devices.

    Each device must call watchdog.heartbeat(device_name) at least
    once per timeout_threshold_s. If it misses, the watchdog
    activates SAFE_STATE for that device and logs an adverse event.
    """

    CHECK_INTERVAL_S = 0.5

    def __init__(
        self,
        safe_state_callback: Optional[Callable] = None,
        audit_logger=None,
    ):
        self._devices: Dict[str, DeviceStatus] = {}
        self.safe_state_cb = safe_state_callback
        self.audit = audit_logger
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register_device(
        self,
        name: str,
        timeout_threshold_s: float = 2.0,
    ):
        """Register a device for monitoring."""
        with self._lock:
            self._devices[name] = DeviceStatus(
                name=name,
                timeout_threshold_s=timeout_threshold_s,
                last_heartbeat=time.time(),
            )
        logger.info(
            f"DeviceSafetyWatchdog: registered '{name}' "
            f"(timeout={timeout_threshold_s}s)"
        )

    def heartbeat(self, device_name: str):
        """
        Called by each device to signal it is healthy.
        Must be called at least once per timeout_threshold_s.
        """
        with self._lock:
            if device_name in self._devices:
                dev = self._devices[device_name]
                dev.last_heartbeat = time.time()
                if dev.health in (DeviceHealth.TIMEOUT, DeviceHealth.DEGRADED):
                    dev.health = DeviceHealth.HEALTHY
                    logger.info(f"Device '{device_name}' recovered")

    def start(self):
        """Start background monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="DeviceSafetyWatchdog",
        )
        self._thread.start()
        logger.info("DeviceSafetyWatchdog started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("DeviceSafetyWatchdog stopped")

    def _monitor_loop(self):
        """Check all device heartbeats every 500ms."""
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                devices_snapshot = list(self._devices.values())

            for dev in devices_snapshot:
                elapsed = now - dev.last_heartbeat
                if elapsed > dev.timeout_threshold_s and dev.health == DeviceHealth.HEALTHY:
                    self._on_device_timeout(dev, elapsed)

            time.sleep(self.CHECK_INTERVAL_S)

    def _on_device_timeout(self, dev: DeviceStatus, elapsed_s: float):
        """Handle device timeout — activate per-device SAFE_STATE."""
        with self._lock:
            dev.health = DeviceHealth.TIMEOUT
            dev.safe_state_count += 1

        logger.error(
            f"DEVICE TIMEOUT: '{dev.name}' — no heartbeat for {elapsed_s:.1f}s "
            f"(threshold={dev.timeout_threshold_s}s). "
            f"Activating SAFE_STATE for {dev.name}."
        )

        if self.audit:
            self.audit.log(
                event_type="device_timeout",
                severity="ERROR",
                details={
                    "device": dev.name,
                    "elapsed_s": elapsed_s,
                    "safe_state_count": dev.safe_state_count,
                },
            )

        if self.safe_state_cb:
            try:
                self.safe_state_cb(device=dev.name, reason="watchdog_timeout")
            except Exception as exc:
                logger.error(f"SAFE_STATE callback error for {dev.name}: {exc}")

    def get_all_status(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "health": dev.health.value,
                    "last_heartbeat_ago_s": round(time.time() - dev.last_heartbeat, 2),
                    "error_count": dev.error_count,
                    "safe_state_count": dev.safe_state_count,
                }
                for name, dev in self._devices.items()
            }

    def all_healthy(self) -> bool:
        with self._lock:
            return all(
                d.health == DeviceHealth.HEALTHY
                for d in self._devices.values()
            )