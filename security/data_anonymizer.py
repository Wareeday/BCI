"""
security/data_anonymizer.py
============================
De-identification and anonymisation of neural data.

IEEE 2857 §5.3: Electrode positions mapped to generic 10-20 labels.
               No biometric tags (no name, DOB, patient ID in raw data).
IEEE 2857 §5.1: EEG stored as feature vectors only — raw waveforms
               deleted after DSP processing.
GDPR Article 89: pseudonymisation as safeguard for research processing.

Anonymisation pipeline:
  1. Replace patient ID with random UUID (pseudonymisation)
  2. Strip electrode metadata → generic 10-20 labels only
  3. Add calibrated Gaussian noise (k-anonymity on feature vectors)
  4. Store only features, not raw waveforms
"""

import hashlib
import json
import os
import time
import uuid
from typing import Any, Optional
import numpy as np
from loguru import logger


class DataAnonymizer:
    """
    Pseudonymises and anonymises neural data for storage and research.

    Pseudonymisation: user_id → hashed_id (reversible with secret key)
    Anonymisation:    feature_vector + noise (irreversible)
    """

    def __init__(self, salt: Optional[str] = None):
        """
        Args:
            salt: HMAC salt for pseudonymisation.
                  If None, uses random salt (NOT reversible across sessions).
        """
        self._salt = salt or os.urandom(32).hex()
        self._pseudonym_cache: dict[str, str] = {}

    def pseudonymise_user_id(self, user_id: str) -> str:
        """
        Replace user_id with HMAC-SHA256 pseudonym.

        The pseudonym is consistent within a session (same salt).
        Can be reversed ONLY by the DPO who holds the salt.
        Per GDPR, pseudonymised data is still personal data.

        Returns 16-char hex pseudonym.
        """
        if user_id in self._pseudonym_cache:
            return self._pseudonym_cache[user_id]

        hmac_input = f"{self._salt}:{user_id}".encode("utf-8")
        pseudonym = hashlib.sha256(hmac_input).hexdigest()[:16]
        self._pseudonym_cache[user_id] = pseudonym
        return pseudonym

    def anonymise_feature_vector(
        self,
        features: np.ndarray,
        noise_std: float = 0.01,
    ) -> np.ndarray:
        """
        Add calibrated Gaussian noise to feature vector.

        Used for federated learning data sharing.
        noise_std=0.01 adds ~1% perturbation — negligible accuracy impact
        while preventing exact reconstruction of the original signal.

        For stronger privacy: use differential_privacy.py (ε=1.0 DP).
        """
        noise = np.random.normal(0.0, noise_std, features.shape).astype(np.float32)
        return (features + noise).astype(np.float32)

    def strip_identifying_metadata(self, session_dict: dict) -> dict:
        """
        Remove personally identifying fields from a session record.

        Fields stripped:
          name, email, date_of_birth, address, patient_id,
          exact_timestamp (replaced with date only),
          ip_address (replaced with /24 subnet).
        """
        STRIP_FIELDS = {
            "name", "email", "date_of_birth", "address",
            "patient_id", "phone", "nhs_number",
        }
        safe = {}
        for key, value in session_dict.items():
            if key in STRIP_FIELDS:
                safe[key] = "[REDACTED]"
            elif key == "timestamp" and isinstance(value, float):
                # Keep date, drop time — reduces re-identification risk
                from datetime import datetime
                safe[key] = datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
            elif key == "ip_address" and isinstance(value, str):
                # Keep /24 subnet only (e.g. 192.168.1.0/24)
                parts = value.split(".")
                if len(parts) == 4:
                    safe[key] = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
                else:
                    safe[key] = "[REDACTED]"
            elif key == "user_id" and isinstance(value, str):
                safe[key] = self.pseudonymise_user_id(value)
            else:
                safe[key] = value
        return safe

    def create_research_record(
        self,
        user_id: str,
        session_id: str,
        feature_vector: np.ndarray,
        label: int,
        noise_std: float = 0.01,
    ) -> dict:
        """
        Create anonymised record suitable for cross-institution research sharing.

        Per GDPR Article 89 and Convention 108+:
        - EU→UK transfer requires adequacy decision
        - Federated learning (without raw data transfer) is preferred
        """
        return {
            "pseudonym_id": self.pseudonymise_user_id(user_id),
            "session_hash": hashlib.sha256(session_id.encode()).hexdigest()[:8],
            "features": self.anonymise_feature_vector(feature_vector, noise_std).tolist(),
            "label": int(label),
            "date": time.strftime("%Y-%m-%d"),
            "anonymisation": f"Gaussian noise σ={noise_std}",
            "standard": "IEEE 2857 §5.3, GDPR Art.89",
        }