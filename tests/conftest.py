"""
tests/conftest.py
==================
Shared pytest fixtures for all test modules.

Fixtures:
  synthetic_epoch       — (8, 200) float32 EEG epoch
  synthetic_mi_epoch    — (8, 1000) motor imagery epoch
  synthetic_feature_vec — (56,) feature vector
  mock_audit_logger     — AuditLogger writing to temp file
  mock_kafka_producer   — disconnected producer (no broker needed)
  mock_board            — OpenBCI in simulation mode
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── EEG Data fixtures ─────────────────────────────────────────────

@pytest.fixture
def synthetic_epoch():
    """800ms P300 epoch: (8 channels, 200 samples)."""
    np.random.seed(42)
    epoch = np.random.randn(8, 200).astype(np.float32) * 5.0
    # Add realistic alpha component
    t = np.linspace(0, 0.8, 200)
    epoch[7] += 15.0 * np.sin(2 * np.pi * 10.0 * t)
    return epoch


@pytest.fixture
def synthetic_mi_epoch():
    """4s motor imagery epoch: (8 channels, 1000 samples)."""
    np.random.seed(42)
    epoch = np.random.randn(8, 1000).astype(np.float32) * 5.0
    t = np.linspace(0, 4.0, 1000)
    epoch[2] += 12.0 * np.sin(2 * np.pi * 12.0 * t)   # C3 beta
    epoch[4] += 8.0 * np.sin(2 * np.pi * 12.0 * t)    # C4 beta
    return epoch


@pytest.fixture
def synthetic_feature_vec():
    """56-dimensional feature vector."""
    np.random.seed(42)
    return np.random.randn(56).astype(np.float32)


@pytest.fixture
def batch_epochs():
    """Batch of 20 motor imagery epochs for ML tests."""
    np.random.seed(42)
    X = np.random.randn(20, 8, 1000).astype(np.float32) * 5.0
    y = np.array([i % 4 for i in range(20)], dtype=np.int64)
    return X, y


# ── Infrastructure fixtures ───────────────────────────────────────

@pytest.fixture
def mock_audit_logger(tmp_path):
    """AuditLogger writing to a temporary file."""
    from security.audit_logger import AuditLogger
    log_file = str(tmp_path / "test_audit.log")
    return AuditLogger(log_file=log_file, echo_to_console=False)


@pytest.fixture
def mock_kafka_producer():
    """EEGKafkaProducer that won't crash even if Kafka is unavailable."""
    from streaming.kafka_producer import EEGKafkaProducer
    # Force disconnected mode
    producer = EEGKafkaProducer.__new__(EEGKafkaProducer)
    producer.topic_raw = "neural-eeg-raw"
    producer.topic_clean = "neural-eeg-clean"
    producer.topic_features = "neural-eeg-features"
    producer.topic_commands = "bci-commands"
    producer._producer = None
    producer._connected = False
    producer._messages_sent = 0
    producer._errors = 0
    return producer


@pytest.fixture
def mock_board():
    """OpenBCI board in simulation mode."""
    from acquisition.openbci_board import OpenBCIBoard
    board = OpenBCIBoard(simulate=True)
    board.connect()
    return board


@pytest.fixture
def dsp_pipeline():
    """DSP pipeline in P300 mode."""
    from dsp.pipeline import DSPPipeline
    return DSPPipeline(n_channels=8, epoch_type="p300")


@pytest.fixture
def lda_classifier():
    """Fitted LDA classifier on synthetic data."""
    from ml.sklearn_baseline import SKLearnBaseline
    np.random.seed(42)
    clf = SKLearnBaseline(method="lda", n_classes=4)
    X = np.random.randn(200, 56).astype(np.float32)
    y = np.array([i % 4 for i in range(200)], dtype=np.int64)
    clf.fit(X, y)
    return clf


@pytest.fixture
def safe_state_coordinator(mock_audit_logger):
    """SafeStateCoordinator with audit logger."""
    from resilience.safe_state import SafeStateCoordinator
    return SafeStateCoordinator(audit_logger=mock_audit_logger)