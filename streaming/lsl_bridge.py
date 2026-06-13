"""
streaming/lsl_bridge.py
========================
LSL → Kafka bridge.

Reads from an LSL inlet (published by acquisition/lsl_streamer.py)
and forwards each sample to Kafka for downstream processing.

This decouples the acquisition hardware from the streaming layer:
  OpenBCI → LSL outlet → [network] → LSL inlet → Kafka producer → CNN
"""
import threading
import time
from typing import Optional
from loguru import logger

try:
    from pylsl import StreamInlet, resolve_stream, local_clock
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logger.warning("pylsl not installed. LSL bridge disabled.")

import numpy as np
from streaming.kafka_producer import EEGKafkaProducer
from acquisition.openbci_board import EEGSample


class LSLKafkaBridge:
    """
    Subscribes to an LSL EEG stream and forwards samples to Kafka.

    The bridge runs as a background thread. On startup it resolves
    the LSL stream by name (timeout 10s) then polls at 250 Hz.
    """

    def __init__(
        self,
        stream_name: str = "OpenBCI_EEG",
        stream_type: str = "EEG",
        kafka_producer: Optional[EEGKafkaProducer] = None,
        resolve_timeout_s: float = 10.0,
    ):
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.kafka = kafka_producer
        self.resolve_timeout = resolve_timeout_s

        self._inlet = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples_bridged = 0
        self._connected = False

    def start(self):
        """Resolve LSL stream and start forwarding to Kafka."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._bridge_loop,
            daemon=True,
            name="LSLKafkaBridge",
        )
        self._thread.start()
        logger.info(f"LSL→Kafka bridge starting (stream='{self.stream_name}')")

    def stop(self):
        """Stop the bridge thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(f"LSL→Kafka bridge stopped. Samples bridged: {self._samples_bridged}")

    def _bridge_loop(self):
        """Resolve LSL stream, then pull and forward samples."""
        if not LSL_AVAILABLE:
            logger.warning("pylsl unavailable — LSL bridge running in mock mode")
            self._mock_loop()
            return

        logger.info(f"Resolving LSL stream '{self.stream_name}'...")
        streams = resolve_stream("name", self.stream_name, timeout=self.resolve_timeout)
        if not streams:
            logger.error(f"No LSL stream found for '{self.stream_name}' after {self.resolve_timeout}s")
            return

        self._inlet = StreamInlet(streams[0])
        self._connected = True
        logger.success(f"LSL inlet connected: {streams[0].name()}")

        while not self._stop_event.is_set():
            sample, timestamp = self._inlet.pull_sample(timeout=0.1)
            if sample is not None:
                channels = np.array(sample, dtype=np.float32)
                eeg_sample = EEGSample(
                    timestamp=timestamp,
                    channels=channels,
                    sample_id=self._samples_bridged % 256,
                )
                if self.kafka:
                    self.kafka.publish_raw_sample(eeg_sample)
                self._samples_bridged += 1

    def _mock_loop(self):
        """Generate synthetic samples when LSL is unavailable."""
        while not self._stop_event.is_set():
            channels = np.random.randn(8).astype(np.float32) * 5.0
            eeg_sample = EEGSample(
                timestamp=time.time(),
                channels=channels,
                sample_id=self._samples_bridged % 256,
            )
            if self.kafka:
                self.kafka.publish_raw_sample(eeg_sample)
            self._samples_bridged += 1
            time.sleep(1.0 / 250.0)

    @property
    def samples_bridged(self) -> int:
        return self._samples_bridged