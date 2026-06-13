"""
streaming/kafka_consumer.py
============================
Kafka consumer — reads feature vectors from Kafka, runs CNN inference,
publishes commands back.

This is the "CNN Consumer" box in the pipeline flow diagram:
  OpenBCI → GNU Radio DSP → LSL → Kafka Broker (TLS) → CNN Consumer → ROS Actuator

Runs in a background thread, consuming neural-eeg-features topic,
performing real-time inference, and publishing to bci-commands topic.
"""

import json
import threading
import time
from typing import Callable, Optional

import numpy as np
from loguru import logger

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


class EEGKafkaConsumer:
    """
    Consumes EEG feature vectors from Kafka and routes to ML inference.

    Consumer group: bci-cnn-consumer
    This allows multiple consumer instances to share load (one per 8 partitions).
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_features: str = "neural-eeg-features",
        topic_commands: str = "bci-commands",
        group_id: str = "bci-cnn-consumer",
        on_features: Optional[Callable] = None,
        security_protocol: str = "PLAINTEXT",
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
    ):
        self.topic_features = topic_features
        self.topic_commands = topic_commands
        self.on_features = on_features

        self._consumer: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._messages_consumed = 0
        self._watchdog_last_msg = 0.0
        self.WATCHDOG_TIMEOUT_S = 0.5

        if not KAFKA_AVAILABLE:
            logger.warning("kafka-python not available. Consumer disabled.")
            return

        config = {
            "bootstrap_servers": bootstrap_servers,
            "group_id": group_id,
            "auto_offset_reset": "latest",        # only process new messages
            "enable_auto_commit": True,
            "auto_commit_interval_ms": 1000,
            "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
            "consumer_timeout_ms": 1000,
            "session_timeout_ms": 10000,
            "max_poll_records": 10,
        }

        if security_protocol == "SSL" and ssl_cafile:
            config.update({
                "security_protocol": "SSL",
                "ssl_cafile": ssl_cafile,
                "ssl_certfile": ssl_certfile,
                "ssl_keyfile": ssl_keyfile,
            })

        try:
            self._consumer = KafkaConsumer(topic_features, **config)
            logger.success(f"Kafka consumer connected, topic: {topic_features}")
        except Exception as exc:
            logger.warning(f"Kafka consumer init failed: {exc}")

    def start(self):
        """Start background consumption thread."""
        if self._consumer is None:
            logger.warning("Kafka consumer not available — starting mock mode")
            self._thread = threading.Thread(target=self._mock_consume, daemon=True)
        else:
            self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._stop_event.clear()
        self._thread.start()
        logger.info("Kafka consumer thread started")

    def stop(self):
        """Stop consumer thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._consumer:
            self._consumer.close()
        logger.info("Kafka consumer stopped")

    def _consume_loop(self):
        """Main consumption loop."""
        while not self._stop_event.is_set():
            try:
                records = self._consumer.poll(timeout_ms=100)
                for partition_records in records.values():
                    for record in partition_records:
                        self._process_message(record.value)
                        self._watchdog_last_msg = time.time()
                        self._messages_consumed += 1
            except Exception as exc:
                logger.error(f"Kafka consumer error: {exc}")
                time.sleep(0.5)

    def _mock_consume(self):
        """Generate synthetic feature messages when Kafka is unavailable."""
        logger.info("Mock consumer: generating synthetic feature messages")
        while not self._stop_event.is_set():
            mock_msg = {
                "timestamp": time.time(),
                "epoch_type": "motor_imagery",
                "features": np.random.randn(56).tolist(),
            }
            self._process_message(mock_msg)
            self._watchdog_last_msg = time.time()
            self._messages_consumed += 1
            time.sleep(0.8)   # ~1 epoch per second in simulation

    def _process_message(self, msg: dict):
        """Route decoded message to feature callback."""
        if self.on_features:
            try:
                features = np.array(msg.get("features", []), dtype=np.float32)
                self.on_features(
                    timestamp=msg.get("timestamp", time.time()),
                    features=features,
                    epoch_type=msg.get("epoch_type", "motor_imagery"),
                )
            except Exception as exc:
                logger.error(f"Feature callback error: {exc}")

    def is_healthy(self) -> bool:
        """Return False if no message within watchdog timeout."""
        if self._watchdog_last_msg == 0.0:
            return True
        return (time.time() - self._watchdog_last_msg) < self.WATCHDOG_TIMEOUT_S

    @property
    def messages_consumed(self) -> int:
        return self._messages_consumed