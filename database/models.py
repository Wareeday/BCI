"""
database/models.py
==================
SQLAlchemy ORM models for the BCI platform.

Tables:
  users             — patient/clinician accounts
  sessions          — BCI sessions (start, stop, paradigm)
  eeg_epochs        — stored (anonymised) EEG feature vectors
  bci_commands      — issued commands and outcomes
  consent_records   — GDPR consent audit trail
  audit_log         — immutable event log (mirrors file audit log)
  adverse_events    — ISO 14155 SAE records
  model_versions    — ML model registry with accuracy metadata
"""

import time
import uuid
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    Text, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def new_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=new_uuid)
    pseudonym_id = Column(String(16), unique=True, nullable=False)  # GDPR pseudonym
    role = Column(String(20), default="patient")   # patient | clinician | researcher
    created_at = Column(Float, default=time.time)
    is_active = Column(Boolean, default=True)

    sessions = relationship("BCISession", back_populates="user")
    consents = relationship("ConsentRecord", back_populates="user")


class BCISession(Base):
    __tablename__ = "bci_sessions"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    paradigm = Column(String(30), default="motor_imagery")
    started_at = Column(Float, default=time.time)
    ended_at = Column(Float, nullable=True)
    total_epochs = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)
    calibration_time_min = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="sessions")
    commands = relationship("BCICommand", back_populates="session")
    adverse_events = relationship("AdverseEvent", back_populates="session")

    __table_args__ = (Index("ix_session_user", "user_id"),)


class EEGEpoch(Base):
    """
    Anonymised EEG feature vector — NOT raw waveform.
    Per IEEE 2857 §5.1: raw waveforms deleted after DSP.
    """
    __tablename__ = "eeg_epochs"

    id = Column(String(36), primary_key=True, default=new_uuid)
    session_id = Column(String(36), ForeignKey("bci_sessions.id"), nullable=False)
    pseudonym_id = Column(String(16), nullable=False)   # no direct user_id
    timestamp = Column(Float, nullable=False)
    epoch_type = Column(String(20), default="motor_imagery")
    features_encrypted = Column(Text, nullable=True)    # AES-256-GCM ciphertext
    label = Column(Integer, nullable=True)
    dsp_latency_ms = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_epoch_session", "session_id"),
        Index("ix_epoch_timestamp", "timestamp"),
    )


class BCICommand(Base):
    """Record of every BCI command issued or rejected."""
    __tablename__ = "bci_commands"

    id = Column(String(36), primary_key=True, default=new_uuid)
    session_id = Column(String(36), ForeignKey("bci_sessions.id"), nullable=False)
    timestamp = Column(Float, nullable=False, default=time.time)
    command = Column(String(20), nullable=False)   # left | right | feet | rest
    device = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String(20), nullable=False)
    executed = Column(Boolean, default=False)
    rejection_reason = Column(String(100), nullable=True)

    session = relationship("BCISession", back_populates="commands")

    __table_args__ = (Index("ix_command_session", "session_id"),)


class ConsentRecord(Base):
    """GDPR consent audit trail — Article 7 compliance."""
    __tablename__ = "consent_records"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    purpose = Column(String(50), nullable=False)
    granted = Column(Boolean, nullable=False)
    timestamp_granted = Column(Float, nullable=True)
    timestamp_revoked = Column(Float, nullable=True)
    ip_address = Column(String(45), nullable=True)   # IPv6 max length
    version = Column(String(10), default="1.0")

    user = relationship("User", back_populates="consents")


class AuditLogEntry(Base):
    """
    Database mirror of the file-based audit log.
    Immutable: no UPDATE or DELETE operations permitted.
    FDA 21 CFR Part 11 compliant.
    """
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=new_uuid)
    timestamp = Column(Float, nullable=False, default=time.time)
    event_type = Column(String(50), nullable=False)
    user_id = Column(String(36), nullable=True)
    session_id = Column(String(36), nullable=True)
    severity = Column(String(10), default="INFO")
    details = Column(JSON, default=dict)

    __table_args__ = (
        Index("ix_audit_event_type", "event_type"),
        Index("ix_audit_timestamp", "timestamp"),
    )


class AdverseEvent(Base):
    """
    Serious Adverse Event (SAE) record — ISO 14155 §14.
    Any unintended wheelchair movement or patient injury risk.
    Must be reported to IRB within 24 hours.
    """
    __tablename__ = "adverse_events"

    id = Column(String(36), primary_key=True, default=new_uuid)
    session_id = Column(String(36), ForeignKey("bci_sessions.id"), nullable=True)
    timestamp = Column(Float, nullable=False, default=time.time)
    description = Column(Text, nullable=False)
    device = Column(String(30), nullable=True)
    severity = Column(String(20), default="SAE")
    reported_to_irb = Column(Boolean, default=False)
    irb_report_timestamp = Column(Float, nullable=True)
    standard = Column(String(30), default="ISO 14155 §14")

    session = relationship("BCISession", back_populates="adverse_events")


class MLModelVersion(Base):
    """ML model registry — version, accuracy, deployment status."""
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=new_uuid)
    model_name = Column(String(50), nullable=False)   # cnn_p300 | eegnet | lda
    version = Column(String(20), nullable=False)
    file_path = Column(String(200), nullable=False)
    accuracy = Column(Float, nullable=True)
    inference_ms = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    dataset = Column(String(100), nullable=True)       # BCI Competition IV 2a
    deployed = Column(Boolean, default=False)
    created_at = Column(Float, default=time.time)
    checksum_sha256 = Column(String(64), nullable=True)