"""
security/tls_config.py
========================
TLS 1.3 + mTLS configuration for Kafka and API endpoints.

Regulatory requirements:
  FDA 510(k) cybersecurity:  TLS 1.3 mandatory (NIST SP 800-66)
  FDA NIST:                  Key rotation every 90 days
  GDPR Article 32:           Encryption in transit (appropriate measures)
  ISO 14155 §8:              Risk mitigation for data interception

Security architecture:
  Patient LAN → mTLS → DMZ (Kafka broker) → mTLS → Processing VPC
  All inter-zone traffic: TLS 1.3 minimum
  Certificate authority: hospital internal CA (or Let's Encrypt for cloud)

mTLS (mutual TLS):
  Both client AND server present certificates.
  Prevents rogue clients from connecting to Kafka broker.
  Required: ssl.client.auth=required in kafka-server.properties
"""

import os
import ssl
from pathlib import Path
from typing import Optional
from loguru import logger

# ── Certificate paths ─────────────────────────────────────────────
CERT_DIR = Path(os.getenv("CERT_DIR", "security/certs"))

CA_CERT       = CERT_DIR / "ca.pem"
CLIENT_CERT   = CERT_DIR / "client.pem"
CLIENT_KEY    = CERT_DIR / "client.key"
SERVER_CERT   = CERT_DIR / "server.pem"
SERVER_KEY    = CERT_DIR / "server.key"

# 90-day rotation per NIST SP 800-66
CERT_VALIDITY_DAYS = 90


def create_kafka_ssl_context(
    ca_file: Optional[str] = None,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    check_hostname: bool = True,
) -> Optional[ssl.SSLContext]:
    """
    Create SSL context for Kafka mTLS connections.

    Returns None if cert files are not found (dev mode — PLAINTEXT).
    Logs warning if falling back to unencrypted connection.
    """
    ca   = Path(ca_file   or CA_CERT)
    cert = Path(cert_file or CLIENT_CERT)
    key  = Path(key_file  or CLIENT_KEY)

    if not all(p.exists() for p in [ca, cert, key]):
        logger.warning(
            "TLS certificates not found. "
            "Kafka will use PLAINTEXT (development mode only). "
            "Run: bash scripts/generate_certs.sh"
        )
        return None

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.load_verify_locations(str(ca))
        ctx.load_cert_chain(str(cert), str(key))
        ctx.check_hostname = check_hostname
        ctx.verify_mode = ssl.CERT_REQUIRED
        logger.success(
            f"TLS 1.3 context created: "
            f"CA={ca.name}, cert={cert.name}"
        )
        return ctx
    except ssl.SSLError as exc:
        logger.error(f"SSL context creation failed: {exc}")
        return None


def create_api_ssl_context() -> Optional[ssl.SSLContext]:
    """
    Create SSL context for FastAPI/uvicorn HTTPS.
    """
    if not SERVER_CERT.exists() or not SERVER_KEY.exists():
        logger.warning(
            "Server TLS certificates not found. "
            "API running over HTTP (dev mode). "
            "In production: provide server.pem and server.key"
        )
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(str(SERVER_CERT), str(SERVER_KEY))
    return ctx


def get_kafka_ssl_config(
    ca_file: Optional[str] = None,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
) -> dict:
    """
    Return kafka-python SSL configuration dict.
    Ready to unpack into KafkaProducer/KafkaConsumer kwargs.
    """
    ca   = str(ca_file   or CA_CERT)
    cert = str(cert_file or CLIENT_CERT)
    key  = str(key_file  or CLIENT_KEY)

    if not all(Path(p).exists() for p in [ca, cert, key]):
        return {"security_protocol": "PLAINTEXT"}

    return {
        "security_protocol":  "SSL",
        "ssl_cafile":         ca,
        "ssl_certfile":       cert,
        "ssl_keyfile":        key,
        "ssl_check_hostname": True,
    }


def check_cert_expiry(cert_path: Optional[str] = None) -> dict:
    """
    Check TLS certificate expiry.
    Alert if certificate expires within 14 days.
    NIST SP 800-66: rotate every 90 days.
    """
    import datetime
    path = Path(cert_path or CLIENT_CERT)
    if not path.exists():
        return {"status": "not_found", "path": str(path)}

    try:
        import subprocess
        result = subprocess.run(
            ["openssl", "x509", "-noout", "-enddate", "-in", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"status": "error", "detail": result.stderr}

        # Parse: notAfter=Aug 25 12:00:00 2025 GMT
        line = result.stdout.strip()
        date_str = line.split("=", 1)[1]
        expiry = datetime.datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")
        days_left = (expiry - datetime.datetime.utcnow()).days
        status = "ok" if days_left > 14 else "expiring_soon"

        if days_left <= 14:
            logger.warning(
                f"TLS certificate expires in {days_left} days: {path}. "
                "Run: bash scripts/generate_certs.sh"
            )

        return {
            "status": status,
            "cert_path": str(path),
            "expiry_date": expiry.isoformat(),
            "days_remaining": days_left,
            "rotation_required": days_left <= 14,
            "nist_sp_800_66": "90-day rotation policy",
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}