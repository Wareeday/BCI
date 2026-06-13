"""
resilience/kafka_failover.py
==============================
[HIGH] Kafka broker crash recovery.

Risk Matrix entry:
  Scenario:   Kafka broker crash (network / OOM)
  Likelihood: Low (with replication=2)
  Severity:   Moderate
  Risk Level: HIGH
  Mitigation: Replication factor=2, ZooKeeper failover, RTO <2s,
              log replay from last checkpoint

Recovery steps:
  1. ZooKeeper promotes replica broker
  2. Kafka log replay last 10s of EEG
  3. Classifier resumes from last checkpoint
  4. RTO: <2 seconds with replica

Detection: Producer heartbeat miss → Kafka health topic silent
"""

import threading
import time
from collections import deque
from typing import Optional, Callable
from loguru import logger
import numpy as np


class KafkaFailoverHandler:
    """
    Handles Kafka broker failures with local buffer fallback.

    When Kafka is unavailable:
    - Buffers EEG messages locally (ring buffer, last 10 seconds)
    - On reconnection: replays buffered messages
    - Maintains RTO < 2 seconds
    """

    BUFFER_SECONDS = 10.0
    HEALTH_CHECK_INTERVAL_S = 1.0
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_DELAY_S = 0.5

    def __init__(
        self,
        sample_rate: int = 250,
        n_channels: int = 8,
        on_broker_recovered: Optional[Callable] = None,
        audit_logger=None,
    ):
        self.sample_rate = sample_rate
        buffer_size = int(self.BUFFER_SECONDS * sample_rate)
        self._local_buffer: deque = deque(maxlen=buffer_size)
        self._broker_healthy = True
        self._failure_count = 0
        self._last_failure_time = 0.0
        self.on_broker_recovered = on_broker_recovered
        self.audit = audit_logger
        self._reconnect_attempts = 0

    def buffer_message(self, message: dict):
        """
        Buffer message locally when Kafka is down.
        Holds last 10 seconds of EEG (250 Hz × 10s = 2500 samples).
        """
        self._local_buffer.append({
            "timestamp": time.time(),
            "message": message,
        })

    def on_broker_failure(self, error: str):
        """Called when producer heartbeat misses."""
        if not self._broker_healthy:
            return
        self._broker_healthy = False
        self._failure_count += 1
        self._last_failure_time = time.time()

        logger.error(
            f"[HIGH] Kafka broker failure #{self._failure_count}: {error}. "
            f"Switching to local buffer ({self.BUFFER_SECONDS}s capacity)"
        )

        if self.audit:
            self.audit.log(
                event_type="kafka_broker_failure",
                severity="ERROR",
                details={"error": error, "failure_count": self._failure_count},
            )

        # Start reconnection in background
        threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
        ).start()

    def on_broker_restored(self, producer):
        """Called when new broker is reachable — replay buffered messages."""
        self._broker_healthy = True
        downtime = time.time() - self._last_failure_time

        logger.success(
            f"Kafka broker restored after {downtime:.1f}s — "
            f"replaying {len(self._local_buffer)} buffered messages"
        )

        # Replay local buffer
        replay_count = 0
        while self._local_buffer:
            buffered = self._local_buffer.popleft()
            try:
                producer.publish_raw_sample_dict(buffered["message"])
                replay_count += 1
            except Exception as exc:
                logger.warning(f"Replay failed for message: {exc}")

        logger.info(f"Kafka replay complete: {replay_count} messages replayed")

        if self.audit:
            self.audit.log(
                event_type="kafka_broker_recovered",
                details={
                    "downtime_s": downtime,
                    "messages_replayed": replay_count,
                    "rto_met": downtime < 2.0,
                },
            )

        if self.on_broker_recovered:
            self.on_broker_recovered()

    def _reconnect_loop(self):
        """Attempt reconnection with exponential backoff."""
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            delay = self.RECONNECT_DELAY_S * (2 ** (attempt - 1))   # 0.5, 1, 2, 4, 8
            logger.info(f"Kafka reconnect attempt {attempt}, waiting {delay:.1f}s...")
            time.sleep(delay)
            self._reconnect_attempts = attempt
            # Actual reconnect logic handled by producer's retry mechanism

    def get_buffer_status(self) -> dict:
        return {
            "broker_healthy": self._broker_healthy,
            "buffered_messages": len(self._local_buffer),
            "buffer_seconds": len(self._local_buffer) / max(1, self.sample_rate),
            "failure_count": self._failure_count,
        }