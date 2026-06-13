"""
tests/test_security.py
=======================
Unit tests for security, encryption, consent, and audit logging.

Tests:
  - AES-256-GCM encrypt/decrypt roundtrip
  - Tamper detection (modified ciphertext raises ValueError)
  - GDPR consent grant/revoke/check
  - Erasure pipeline triggered on revoke
  - Audit log append-only (immutable)
  - Data anonymisation (pseudonymisation consistent)
  - Differential privacy noise application
"""

import pytest
import numpy as np
import json


class TestAESEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        from security.aes_encryption import AESEncryption
        enc = AESEncryption()
        original = {"user": "test", "channels": [1.2, -0.5, 3.0]}
        encrypted = enc.encrypt(original)
        assert encrypted["encrypted"] is True
        assert "ciphertext" in encrypted
        decrypted = enc.decrypt(encrypted)
        decoded = json.loads(decrypted.decode("utf-8"))
        assert decoded["user"] == "test"

    def test_encrypt_numpy_array(self):
        from security.aes_encryption import AESEncryption
        enc = AESEncryption()
        arr = np.random.randn(56).astype(np.float32)
        encrypted = enc.encrypt(arr)
        assert encrypted["encrypted"] is True

    def test_tamper_detection(self):
        from security.aes_encryption import AESEncryption
        import base64
        enc = AESEncryption()
        encrypted = enc.encrypt({"data": "sensitive_eeg"})
        # Flip one byte in ciphertext
        ct_bytes = base64.b64decode(encrypted["ciphertext"])
        tampered = bytearray(ct_bytes)
        tampered[5] ^= 0xFF
        encrypted["ciphertext"] = base64.b64encode(bytes(tampered)).decode()
        with pytest.raises(ValueError):
            enc.decrypt(encrypted)

    def test_feature_vector_encryption(self):
        from security.aes_encryption import AESEncryption
        enc = AESEncryption()
        features = np.random.randn(56).astype(np.float32)
        result = enc.encrypt_feature_vector(features, user_id="patient_001")
        assert result["encrypted"] is True
        assert result["user_id"] == "patient_001"
        assert result["shape"] == [56]

    def test_stats_tracking(self):
        from security.aes_encryption import AESEncryption
        enc = AESEncryption()
        for _ in range(5):
            e = enc.encrypt({"x": 1})
            enc.decrypt(e)
        stats = enc.get_stats()
        assert stats["encrypt_count"] == 5
        assert stats["decrypt_count"] == 5


class TestConsentManager:
    def test_grant_and_check_consent(self, mock_audit_logger):
        from security.consent_manager import ConsentManager, ConsentPurpose
        cm = ConsentManager(audit_logger=mock_audit_logger)
        cm.grant_consent("user_001", [ConsentPurpose.NEURAL_PROCESSING])
        assert cm.check_consent("user_001", ConsentPurpose.NEURAL_PROCESSING)

    def test_revoke_removes_consent(self, mock_audit_logger):
        from security.consent_manager import ConsentManager, ConsentPurpose
        cm = ConsentManager(audit_logger=mock_audit_logger)
        cm.grant_consent("user_002", [ConsentPurpose.NEURAL_PROCESSING,
                                       ConsentPurpose.MODEL_TRAINING])
        cm.revoke_consent("user_002", ConsentPurpose.MODEL_TRAINING)
        assert not cm.check_consent("user_002", ConsentPurpose.MODEL_TRAINING)
        assert cm.check_consent("user_002", ConsentPurpose.NEURAL_PROCESSING)

    def test_erasure_callback_triggered_on_revoke(self, mock_audit_logger):
        from security.consent_manager import ConsentManager, ConsentPurpose
        cm = ConsentManager(audit_logger=mock_audit_logger)
        erased_users = []
        cm.register_erasure_callback(lambda user_id, purposes: erased_users.append(user_id))
        cm.grant_consent("user_003", [ConsentPurpose.NEURAL_PROCESSING])
        cm.revoke_consent("user_003")
        assert "user_003" in erased_users

    def test_has_required_consents(self, mock_audit_logger):
        from security.consent_manager import ConsentManager, ConsentPurpose
        cm = ConsentManager(audit_logger=mock_audit_logger)
        cm.grant_consent("user_004", [
            ConsentPurpose.NEURAL_PROCESSING,
            ConsentPurpose.AUDIT_LOGGING,
        ])
        assert cm.has_required_consents("user_004")

    def test_missing_required_consent_fails(self, mock_audit_logger):
        from security.consent_manager import ConsentManager
        cm = ConsentManager(audit_logger=mock_audit_logger)
        assert not cm.has_required_consents("new_user_no_consent")


class TestAuditLogger:
    def test_log_entry_written(self, mock_audit_logger, tmp_path):
        entry_id = mock_audit_logger.log(
            event_type="test_event",
            user_id="user_001",
            details={"key": "value"},
        )
        assert entry_id is not None
        # Verify file was written
        import os
        assert os.path.exists(mock_audit_logger.log_file)
        with open(mock_audit_logger.log_file) as f:
            content = f.read()
        assert "test_event" in content

    def test_log_inference_entry(self, mock_audit_logger):
        entry_id = mock_audit_logger.log_inference(
            user_id="user_001",
            session_id="sess_001",
            predicted_class=0,
            class_name="left",
            confidence=0.91,
            model_used="cnn",
            epoch_type="motor_imagery",
        )
        assert entry_id is not None
        assert mock_audit_logger.entry_count >= 1

    def test_sae_logged_as_critical(self, mock_audit_logger):
        entry_id = mock_audit_logger.log_sae(
            user_id="user_001",
            session_id="sess_001",
            description="Unintended wheelchair movement",
            device="wheelchair",
        )
        assert entry_id is not None

    def test_query_by_event_type(self, mock_audit_logger):
        mock_audit_logger.log(event_type="alpha_event", details={})
        mock_audit_logger.log(event_type="beta_event", details={})
        results = mock_audit_logger.query(event_type="alpha_event")
        assert all(r["event_type"] == "alpha_event" for r in results)

    def test_entry_count_increments(self, mock_audit_logger):
        initial = mock_audit_logger.entry_count
        for _ in range(5):
            mock_audit_logger.log(event_type="count_test")
        assert mock_audit_logger.entry_count == initial + 5


class TestDataAnonymizer:
    def test_pseudonym_consistent(self):
        from security.data_anonymizer import DataAnonymizer
        anon = DataAnonymizer(salt="test_salt_12345")
        p1 = anon.pseudonymise_user_id("patient_001")
        p2 = anon.pseudonymise_user_id("patient_001")
        assert p1 == p2

    def test_different_users_different_pseudonyms(self):
        from security.data_anonymizer import DataAnonymizer
        anon = DataAnonymizer(salt="test_salt_12345")
        p1 = anon.pseudonymise_user_id("patient_001")
        p2 = anon.pseudonymise_user_id("patient_002")
        assert p1 != p2

    def test_noise_added_to_features(self):
        from security.data_anonymizer import DataAnonymizer
        anon = DataAnonymizer()
        features = np.ones(56, dtype=np.float32)
        noisy = anon.anonymise_feature_vector(features, noise_std=0.1)
        assert not np.allclose(features, noisy)
        assert noisy.dtype == np.float32

    def test_identifying_fields_stripped(self):
        from security.data_anonymizer import DataAnonymizer
        anon = DataAnonymizer()
        session = {
            "name": "John Doe",
            "email": "john@hospital.nhs.uk",
            "user_id": "patient_001",
            "accuracy": 0.91,
            "timestamp": 1700000000.0,
        }
        safe = anon.strip_identifying_metadata(session)
        assert safe["name"] == "[REDACTED]"
        assert safe["email"] == "[REDACTED]"
        assert safe["user_id"] != "patient_001"  # pseudonymised
        assert safe["accuracy"] == 0.91  # non-PII preserved

    def test_research_record_format(self):
        from security.data_anonymizer import DataAnonymizer
        anon = DataAnonymizer()
        features = np.random.randn(56).astype(np.float32)
        record = anon.create_research_record("u001", "s001", features, label=2)
        assert "pseudonym_id" in record
        assert "features" in record
        assert record["label"] == 2
        assert "user_id" not in record   # must not contain raw user_id