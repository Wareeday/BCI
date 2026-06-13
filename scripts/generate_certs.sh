#!/bin/bash
# scripts/generate_certs.sh
# ===========================
# Generate self-signed TLS certificates for Kafka mTLS.
# Production: replace with certificates signed by hospital CA.
# NIST SP 800-66: key rotation every 90 days.

set -e
CERT_DIR="security/certs"
mkdir -p "$CERT_DIR"

echo "Generating CA key and certificate..."
openssl req -new -x509 -keyout "$CERT_DIR/ca.key" -out "$CERT_DIR/ca.pem" -days 90 \
  -passout pass:bci-ca-pass -subj "/CN=BCI-CA/O=Hospital/C=UK"

echo "Generating client key and CSR..."
openssl req -newkey rsa:2048 -nodes -keyout "$CERT_DIR/client.key" \
  -out "$CERT_DIR/client.csr" -subj "/CN=bci-client/O=Hospital/C=UK"

echo "Signing client certificate with CA..."
openssl x509 -req -CA "$CERT_DIR/ca.pem" -CAkey "$CERT_DIR/ca.key" \
  -in "$CERT_DIR/client.csr" -out "$CERT_DIR/client.pem" \
  -days 90 -CAcreateserial -passin pass:bci-ca-pass

echo "Certificate generated in $CERT_DIR/"
echo "Expires in 90 days per NIST SP 800-66"
ls -la "$CERT_DIR/"