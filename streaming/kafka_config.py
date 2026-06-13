"""
streaming/kafka_config.py
==========================
Centralised Kafka configuration constants and helpers.

Single source of truth for all topic names, partition counts,
retention policies, and security settings.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaTopics:
    RAW      = "neural-eeg-raw"       # 8 partitions, 1h retention
    CLEAN    = "neural-eeg-clean"     # 4 partitions, 1h retention
    FEATURES = "neural-eeg-features"  # 4 partitions, 1h retention
    COMMANDS = "bci-commands"         # 1 partition,  24h retention


@dataclass(frozen=True)
class KafkaDefaults:
    BOOTSTRAP       = "localhost:9092"
    REPLICATION     = 1          # set to 2 in production
    PARTITIONS_RAW  = 8          # 1 per EEG channel
    PARTITIONS_PROC = 4
    RETENTION_MS_EEG     = 3_600_000    # 1 hour (GDPR data minimisation)
    RETENTION_MS_COMMANDS = 86_400_000  # 24 hours (audit trail)
    COMPRESSION     = "lz4"
    LINGER_MS       = 0          # no batching — real-time


def get_producer_config(
    bootstrap: str = KafkaDefaults.BOOTSTRAP,
    security: str = "PLAINTEXT",
    ssl_cafile: str = None,
    ssl_certfile: str = None,
    ssl_keyfile: str = None,
) -> dict:
    """Build kafka-python producer config dict."""
    cfg = {
        "bootstrap_servers": bootstrap,
        "acks": "all",
        "retries": 3,
        "retry_backoff_ms": 100,
        "compression_type": KafkaDefaults.COMPRESSION,
        "linger_ms": KafkaDefaults.LINGER_MS,
        "request_timeout_ms": 500,
    }
    if security == "SSL" and ssl_cafile:
        cfg.update({
            "security_protocol": "SSL",
            "ssl_cafile": ssl_cafile,
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
            "ssl_check_hostname": True,
        })
    return cfg


def get_consumer_config(
    bootstrap: str = KafkaDefaults.BOOTSTRAP,
    group_id: str = "bci-cnn-consumer",
    security: str = "PLAINTEXT",
) -> dict:
    """Build kafka-python consumer config dict."""
    return {
        "bootstrap_servers": bootstrap,
        "group_id": group_id,
        "auto_offset_reset": "latest",
        "enable_auto_commit": True,
        "auto_commit_interval_ms": 1000,
        "session_timeout_ms": 10000,
        "max_poll_records": 10,
    }


# Load from environment at import time
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", KafkaDefaults.BOOTSTRAP)
TOPIC_RAW         = os.getenv("KAFKA_TOPIC_RAW",      KafkaTopics.RAW)
TOPIC_CLEAN       = os.getenv("KAFKA_TOPIC_CLEAN",    KafkaTopics.CLEAN)
TOPIC_FEATURES    = os.getenv("KAFKA_TOPIC_FEATURES", KafkaTopics.FEATURES)
TOPIC_COMMANDS    = os.getenv("KAFKA_TOPIC_COMMANDS", KafkaTopics.COMMANDS)