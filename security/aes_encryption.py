"""
security/aes_encryption.py
===========================
AES-256-GCM encryption for neural data at rest and in transit.

Regulatory requirements:
  - FDA cybersecurity guidance: AES-256 + TLS 1.3 required
  - GDPR Article 32: appropriate technical measures (encryption)
  - NIST SP 800-66: key rotation every 90 days
  - IEEE 2857 §5.1: raw EEG waveforms deleted after DSP

AES-256-GCM chosen over AES-CBC because:
  - GCM provides authenticated encryption (prevents tampering)
  - Detects any modification to ciphertext before decryption
  - Required for compliance with FDA cybersecurity framework
"""

import base64
import json
import os
import time
from typing import Union
import numpy as np
from loguru import logger

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not installed. Encryption disabled.")
    CRYPTO_AVAILABLE = False


class AESEncryption:
    """
    AES-256-GCM encryption for EEG feature vectors and session data.

    Key rotation: keys should be rotated every 90 days per NIST SP 800-66.
    Key storage:  use AWS KMS or Azure Key Vault in production.
                  Never store keys in code or plaintext files.
    """

    KEY_SIZE_BYTES = 32    # AES-256
    NONCE_SIZE_BYTES = 12  # GCM standard nonce

    def __init__(self, key_hex: Optional[str] = None):
        """
        Args:
            key_hex: 64-character hex string (32 bytes / 256 bits).
                     If None, generates a new random key (dev mode).
        """
        if not CRYPTO_AVAILABLE:
            self._key = None
            return

        if key_hex:
            self._key = bytes.fromhex(key_hex)
            if len(self._key) != self.KEY_SIZE_BYTES:
                raise ValueError(f"AES key must be 32 bytes, got {len(self._key)}")
        else:
            self._key = AESGCM.generate_key(bit_length=256)
            logger.warning(
                "AES key auto-generated (dev mode). "
                "Set AES_KEY_HEX in .env for production."
            )

        self._aesgcm = AESGCM(self._key)
        self._encrypt_count = 0
        self._decrypt_count = 0

    def encrypt(self, data: Union[bytes, dict, np.ndarray]) -> dict:
        """
        Encrypt data with AES-256-GCM.

        Args:
            data: bytes, JSON-serialisable dict, or numpy array

        Returns:
            {
              "ciphertext": base64-encoded ciphertext,
              "nonce": base64-encoded 12-byte nonce,
              "timestamp": UNIX timestamp,
              "algorithm": "AES-256-GCM"
            }
        """
        if not CRYPTO_AVAILABLE or self._key is None:
            return {"plaintext": str(data), "encrypted": False}

        # Serialise input to bytes
        if isinstance(data, np.ndarray):
            raw = data.tobytes()
        elif isinstance(data, dict):
            raw = json.dumps(data, default=str).encode("utf-8")
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = data

        nonce = os.urandom(self.NONCE_SIZE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, raw, None)
        self._encrypt_count += 1

        return {
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "timestamp": time.time(),
            "algorithm": "AES-256-GCM",
            "encrypted": True,
        }

    def decrypt(self, encrypted: dict) -> bytes:
        """
        Decrypt AES-256-GCM ciphertext.

        Raises ValueError if authentication tag verification fails
        (indicates data tampering).
        """
        if not CRYPTO_AVAILABLE or not encrypted.get("encrypted"):
            return str(encrypted.get("plaintext", "")).encode()

        ciphertext = base64.b64decode(encrypted["ciphertext"])
        nonce = base64.b64decode(encrypted["nonce"])

        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
            self._decrypt_count += 1
            return plaintext
        except Exception as exc:
            logger.error(f"AES decryption failed — possible tampering: {exc}")
            raise ValueError(f"Decryption failed: {exc}") from exc

    def encrypt_feature_vector(self, features: np.ndarray, user_id: str) -> dict:
        """
        Encrypt a feature vector with user_id as authenticated additional data.
        The user_id binds the ciphertext to the user — prevents mix-up attacks.
        """
        if not CRYPTO_AVAILABLE or self._key is None:
            return {"features": features.tolist(), "encrypted": False}

        nonce = os.urandom(self.NONCE_SIZE_BYTES)
        plaintext = features.astype(np.float32).tobytes()
        aad = user_id.encode("utf-8")   # authenticated additional data

        ciphertext = self._aesgcm.encrypt(nonce, plaintext, aad)
        self._encrypt_count += 1

        return {
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "user_id": user_id,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "encrypted": True,
            "algorithm": "AES-256-GCM",
        }

    @property
    def key_hex(self) -> str:
        """Export key as hex (for secure storage — never log this)."""
        return self._key.hex() if self._key else ""

    def get_stats(self) -> dict:
        return {
            "encrypt_count": self._encrypt_count,
            "decrypt_count": self._decrypt_count,
            "algorithm": "AES-256-GCM",
            "key_size_bits": self.KEY_SIZE_BYTES * 8,
        }


# Type hint fix for Optional
from typing import Optional