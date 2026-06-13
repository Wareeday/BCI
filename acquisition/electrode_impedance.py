"""
acquisition/electrode_impedance.py
====================================
Electrode impedance monitoring and reporting.

ISO 14155 Phase 1 Bench Test criterion: all channels < 5 kΩ.
Runs automatically on session start and every 30 minutes during use.
Poor impedance → signal quality badge turns red on dashboard.
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger


@dataclass
class ImpedanceResult:
    channel: int
    label: str
    impedance_kohm: float
    timestamp: float
    passed: bool          # < 5 kΩ
    recommendation: str


CHANNEL_LABELS = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]
TARGET_KOHM = 5.0


class ElectrodeImpedanceMonitor:
    """
    Monitors electrode impedance for all 8 channels.

    In real hardware: sends 'z' command to OpenBCI Cyton and parses
    the impedance values returned over serial.
    In simulation: generates realistic values with occasional drift.
    """

    def __init__(self, n_channels: int = 8, simulate: bool = True):
        self.n_channels = n_channels
        self.simulate = simulate
        self._history: list[list[ImpedanceResult]] = []
        self._last_check: float = 0.0
        self._check_interval_s = 1800.0   # 30 minutes

    def measure(self, board=None) -> list[ImpedanceResult]:
        """
        Measure impedance for all channels.

        Returns list of ImpedanceResult — one per channel.
        ISO 14155: target < 5 kΩ, log result with timestamp.
        """
        if self.simulate or board is None:
            return self._simulate_measurement()

        raw = board.check_impedances()
        results = []
        for ch, kohm in raw.items():
            label = CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else f"Ch{ch}"
            passed = kohm < TARGET_KOHM
            results.append(ImpedanceResult(
                channel=ch,
                label=label,
                impedance_kohm=round(kohm, 2),
                timestamp=time.time(),
                passed=passed,
                recommendation="" if passed else f"Re-gel electrode {label} (current: {kohm:.1f}kΩ)",
            ))
        self._history.append(results)
        self._last_check = time.time()
        self._log_summary(results)
        return results

    def _simulate_measurement(self) -> list[ImpedanceResult]:
        """Simulate realistic impedance values with some drift."""
        np.random.seed(int(time.time()) % 1000)
        results = []
        for ch in range(self.n_channels):
            # Most channels good; occasional marginal channel
            base = 1.5 + np.random.exponential(1.2)
            kohm = round(min(base, 9.9), 2)
            passed = kohm < TARGET_KOHM
            label = CHANNEL_LABELS[ch]
            results.append(ImpedanceResult(
                channel=ch,
                label=label,
                impedance_kohm=kohm,
                timestamp=time.time(),
                passed=passed,
                recommendation="" if passed else f"Re-gel {label} ({kohm:.1f}kΩ > {TARGET_KOHM}kΩ)",
            ))
        self._history.append(results)
        self._last_check = time.time()
        self._log_summary(results)
        return results

    def _log_summary(self, results: list[ImpedanceResult]):
        passed = sum(1 for r in results if r.passed)
        failed = [r for r in results if not r.passed]
        if failed:
            logger.warning(
                f"Impedance check: {passed}/{self.n_channels} passed. "
                f"Poor channels: {[r.label for r in failed]}"
            )
        else:
            logger.success(f"Impedance check: all {self.n_channels} channels < {TARGET_KOHM}kΩ ✓")

    def needs_recheck(self) -> bool:
        """Return True if 30 minutes have passed since last check."""
        return (time.time() - self._last_check) > self._check_interval_s

    def get_quality_report(self, results: list[ImpedanceResult]) -> dict:
        """Return structured quality report for dashboard and ISO 14155 logging."""
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        return {
            "timestamp": time.time(),
            "total_channels": self.n_channels,
            "channels_passed": len(passed),
            "channels_failed": len(failed),
            "max_impedance_kohm": max(r.impedance_kohm for r in results),
            "target_kohm": TARGET_KOHM,
            "iso14155_phase1_passed": len(failed) == 0,
            "recommendations": [r.recommendation for r in failed],
            "channel_values": {r.label: r.impedance_kohm for r in results},
        }