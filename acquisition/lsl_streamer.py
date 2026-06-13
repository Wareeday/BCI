"""
acquisition/lsl_streamer.py
============================
LabStreamingLayer (LSL) outlet.

Publishes EEG samples to a local network LSL stream.
Any LSL-aware application (BrainViz, OpenViBE, EEGLAB) can subscribe.

Latency contribution: ~0 ms (push_sample is synchronous, <0.1 ms overhead)
Total DSP budget contribution: this is step 0 — acquisition side.
"""

import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except ImportError:
    logger.warning("pylsl not installed — LSL streaming disabled. pip install pylsl")
    LSL_AVAILABLE = False

from acquisition.openbci_board import EEGSample


class LSLStreamer:
    """
    Publishes EEG data as an LSL outlet.

    Downstream consumers:
    - streaming/lsl_bridge.py (bridges to Kafka)
    - dashboard/eeg_plot.py (real-time plot)
    - Any external LSL app (BrainViz, OpenViBE)
    """

    CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]
    CHANNEL_UNITS = "microvolts"
    CHANNEL_TYPE = "EEG"

    def __init__(
        self,
        stream_name: str = "OpenBCI_EEG",
        stream_type: str = "EEG",
        channel_count: int = 8,
        sample_rate: float = 250.0,
        source_id: str = "openbci-cyton-001",
    ):
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.channel_count = channel_count
        self.sample_rate = sample_rate
        self.source_id = source_id

        self._outlet: Optional[object] = None
        self._sample_count = 0
        self._last_push_time = 0.0
        self._watchdog_timeout = 0.5   # 500 ms → SAFE_STATE

    def start(self):
        """Create LSL outlet (network-discoverable on local subnet)."""
        if not LSL_AVAILABLE:
            logger.warning("LSL not available — skipping outlet creation")
            return

        info = StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=self.channel_count,
            nominal_srate=self.sample_rate,
            channel_format="float32",
            source_id=self.source_id,
        )

        # Add channel metadata (EEGLAB / MNE-Python compatible)
        channels = info.desc().append_child("channels")
        for ch_name in self.CHANNEL_NAMES[: self.channel_count]:
            ch = channels.append_child("channel")
            ch.append_child_value("label", ch_name)
            ch.append_child_value("unit", self.CHANNEL_UNITS)
            ch.append_child_value("type", self.CHANNEL_TYPE)

        self._outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)
        logger.success(
            f"LSL outlet created: '{self.stream_name}' "
            f"({self.channel_count} ch, {self.sample_rate} Hz)"
        )

    def push_sample(self, sample: EEGSample):
        """
        Push one EEG sample to LSL outlet.
        Called from OpenBCI callback thread — must be <1 ms.
        """
        if not LSL_AVAILABLE or self._outlet is None:
            return

        try:
            self._outlet.push_sample(
                sample.channels.tolist(),
                timestamp=sample.timestamp,
            )
            self._sample_count += 1
            self._last_push_time = time.time()
        except Exception as exc:
            logger.error(f"LSL push_sample failed: {exc}")

    def stop(self):
        """Destroy LSL outlet."""
        self._outlet = None
        logger.info("LSL outlet closed")

    def is_healthy(self) -> bool:
        """Return False if no sample pushed within watchdog timeout."""
        if self._last_push_time == 0.0:
            return True
        return (time.time() - self._last_push_time) < self._watchdog_timeout

    @property
    def samples_pushed(self) -> int:
        return self._sample_count