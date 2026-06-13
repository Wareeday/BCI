"""
streaming/schema_registry.py
==============================
Avro schema registry for Kafka message validation.

Ensures all producers and consumers use the same message schema.
Prevents schema drift which would corrupt the ML feature pipeline.

Schemas:
  EEGSampleSchema    — raw 8-channel sample (neural-eeg-raw topic)
  EEGFeaturesSchema  — 56-dim feature vector (neural-eeg-features topic)
  BCICommandSchema   — decoded motor command (bci-commands topic)
"""
import json
import time
from typing import Any, Optional
from loguru import logger


# ── Schema definitions ────────────────────────────────────────────

EEG_SAMPLE_SCHEMA = {
    "type": "record",
    "name": "EEGSample",
    "namespace": "bci.platform",
    "fields": [
        {"name": "timestamp",  "type": "double"},
        {"name": "sample_id",  "type": "int"},
        {"name": "channels",   "type": {"type": "array", "items": "float"}},
    ],
}

EEG_FEATURES_SCHEMA = {
    "type": "record",
    "name": "EEGFeatures",
    "namespace": "bci.platform",
    "fields": [
        {"name": "timestamp",   "type": "double"},
        {"name": "epoch_type",  "type": "string"},
        {"name": "features",    "type": {"type": "array", "items": "float"}},
        {"name": "quality_score", "type": ["null", "float"], "default": None},
    ],
}

BCI_COMMAND_SCHEMA = {
    "type": "record",
    "name": "BCICommand",
    "namespace": "bci.platform",
    "fields": [
        {"name": "timestamp",  "type": "double"},
        {"name": "command",    "type": "string"},
        {"name": "confidence", "type": "float"},
        {"name": "model",      "type": "string"},
    ],
}

_REGISTRY: dict[str, dict] = {
    "neural-eeg-raw":      EEG_SAMPLE_SCHEMA,
    "neural-eeg-features": EEG_FEATURES_SCHEMA,
    "bci-commands":        BCI_COMMAND_SCHEMA,
}


class SchemaRegistry:
    """
    In-memory schema registry.
    In production: use Confluent Schema Registry with HTTP API.
    """

    def __init__(self):
        self._schemas = dict(_REGISTRY)
        self._versions: dict[str, int] = {k: 1 for k in _REGISTRY}

    def get_schema(self, topic: str) -> Optional[dict]:
        return self._schemas.get(topic)

    def validate(self, topic: str, message: dict) -> bool:
        """
        Validate a message dict against the registered schema for a topic.
        Returns True if valid, False otherwise.
        """
        schema = self._schemas.get(topic)
        if schema is None:
            return True   # no schema registered → allow
        required = {f["name"] for f in schema.get("fields", [])}
        missing = required - set(message.keys())
        if missing:
            logger.warning(f"Schema validation failed for {topic}: missing {missing}")
            return False
        return True

    def register(self, topic: str, schema: dict, version: int = 1):
        """Register or update a schema for a topic."""
        self._schemas[topic] = schema
        self._versions[topic] = version
        logger.info(f"Schema registered: {topic} v{version}")

    def get_version(self, topic: str) -> int:
        return self._versions.get(topic, 0)


# Singleton instance
registry = SchemaRegistry()