"""
streaming/kafka_producer.py
============================
Apache Kafka producer — publishes raw and processed EEG to topics.

Kafka config (from presentation):
  Topic: neural-eeg-raw  |  8 partitions (1 per channel)
  Replication factor: 2   |  RTO <2 seconds
  Retention: 1 hour       |  (GDPR data minimisation)
  Compression: lz4        |  TLS 1.3 + mTLS

Why Kafka over ZeroMQ/RabbitMQ:
  - Log-based replay: if classifier crashes, we can replay last 10s of EEG
  - 100+ concurrent users per broker
  - 8 KB/s per user (250 Hz × 8 ch × 4 bytes)
  - Fault-tolerant: replication factor=2, ZooKeeper failover RTO <2s

Bandwidth calculation:
  250 samples/s × 8 channels × 4 bytes/float32 = 8,000 bytes/s = ~8 KB/s
"""

import json
import time
from typing import Optional
import numpy as np
from loguru import logger

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("kafka-python not installed. Producer disabled.")
    KAFKA_AVAILABLE = False

from acquisition.openbci_board import EEGSample


class EEGKafkaProducer:
    """
    Publishes EEG data to Kafka topics.

    Topics:
      neural-eeg-raw      — raw 8-channel samples (250 Hz)
      neural-eeg-clean    — DSP-filtered samples
      neural-eeg-features — extracted feature vectors
      bci-commands        — decoded commands (left/right/etc.)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_raw: str = "neural-eeg-raw",
        topic_clean: str = "neural-eeg-clean",
        topic_features: str = "neural-eeg-features",
        topic_commands: str = "bci-commands",
        security_protocol: str = "PLAINTEXT",
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        compression: str = "lz4",
    ):
        self.topic_raw = topic_raw
        self.topic_clean = topic_clean
        self.topic_features = topic_features
        self.topic_commands = topic_commands
        self._producer: Optional[object] = None
        self._connected = False
        self._messages_sent = 0
        self._errors = 0

        if not KAFKA_AVAILABLE:
            return

        config = {
            "bootstrap_servers": bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "acks": "all",                  # wait for all replicas
            "retries": 3,
            "retry_backoff_ms": 100,
            "compression_type": compression,
            "linger_ms": 0,                 # no batching — real-time
            "request_timeout_ms": 500,
        }

        if security_protocol == "SSL" and ssl_cafile:
            config.update({
                "security_protocol": "SSL",
                "ssl_cafile": ssl_cafile,
                "ssl_certfile": ssl_certfile,
                "ssl_keyfile": ssl_keyfile,
                "ssl_check_hostname": True,
            })

        try:
            self._producer = KafkaProducer(**config)
            self._connected = True
            logger.success(f"Kafka producer connected: {bootstrap_servers}")
        except NoBrokersAvailable:
            logger.warning(
                f"Kafka broker not available at {bootstrap_servers}. "
                "Messages will be dropped. Start Kafka with: docker-compose up kafka"
            )
        except Exception as exc:
            logger.error(f"Kafka producer init failed: {exc}")

    def publish_raw_sample(self, sample: EEGSample):
        """Publish one raw EEG sample to neural-eeg-raw topic."""
        if not self._connected or self._producer is None:
            return

        msg = {
            "timestamp": sample.timestamp,
            "sample_id": sample.sample_id,
            "channels": sample.channels.tolist(),
        }
        # Partition key = channel 0 value bucket (distributes across 8 partitions)
        partition_key = str(sample.sample_id % 8)
        self._send(self.topic_raw, msg, partition_key)

    def publish_clean_eeg(self, timestamp: float, clean_channels: np.ndarray):
        """Publish DSP-cleaned EEG epoch."""
        if not self._connected or self._producer is None:
            return
        msg = {
            "timestamp": timestamp,
            "channels": clean_channels.tolist(),
        }
        self._send(self.topic_clean, msg, "clean")

    def publish_features(self, timestamp: float, feature_vector: np.ndarray, epoch_type: str):
        """Publish extracted feature vector."""
        if not self._connected or self._producer is None:
            return
        msg = {
            "timestamp": timestamp,
            "epoch_type": epoch_type,
            "features": feature_vector.tolist(),
        }
        self._send(self.topic_features, msg, epoch_type)

    def publish_command(
        self,
        timestamp: float,
        command: str,
        confidence: float,
        model_used: str,
    ):
        """Publish decoded BCI command."""
        if not self._connected or self._producer is None:
            return
        msg = {
            "timestamp": timestamp,
            "command": command,
            "confidence": confidence,
            "model": model_used,
        }
        self._send(self.topic_commands, msg, command)
        logger.info(f"BCI command published: {command} (conf={confidence:.2f})")

    def _send(self, topic: str, value: dict, key: str):
        """Send with error callback."""
        try:
            self._producer.send(
                topic, value=value, key=key
            ).add_errback(self._on_error)
            self._messages_sent += 1
        except KafkaError as exc:
            logger.error(f"Kafka send failed: {exc}")
            self._errors += 1

    def _on_error(self, exc):
        logger.error(f"Kafka async send error: {exc}")
        self._errors += 1

    def flush(self):
        """Flush buffered messages."""
        if self._producer:
            self._producer.flush(timeout=1.0)

    def close(self):
        """Gracefully close producer."""
        if self._producer:
            self._producer.flush(timeout=2.0)
            self._producer.close()
            logger.info("Kafka producer closed")

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "messages_sent": self._messages_sent,
            "errors": self._errors,
        }