#!/bin/bash
# streaming/config/kafka-topics.sh
# ==================================
# Create and configure all Kafka topics for BCI platform.
# Run once after Kafka broker is started.
#
# Usage:
#   chmod +x streaming/config/kafka-topics.sh
#   ./streaming/config/kafka-topics.sh
#
# Topics created:
#   neural-eeg-raw       8 partitions  1h retention  (GDPR: raw purge)
#   neural-eeg-clean     4 partitions  1h retention
#   neural-eeg-features  4 partitions  1h retention
#   bci-commands         1 partition   24h retention  (audit trail)

set -e

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
KAFKA_BIN="${KAFKA_HOME:-/usr/local/kafka}/bin"

# Fallback: use docker exec if kafka-topics.sh not on PATH
if ! command -v kafka-topics.sh &>/dev/null; then
    if docker ps --format '{{.Names}}' | grep -q bci-kafka; then
        alias kafka-topics.sh="docker exec bci-kafka kafka-topics.sh"
    else
        echo "ERROR: kafka-topics.sh not found. Start Kafka first:"
        echo "  docker-compose up -d kafka"
        exit 1
    fi
fi

echo "Creating BCI Kafka topics on $BOOTSTRAP..."
echo ""

# Wait for broker to be ready
MAX_WAIT=30
WAITED=0
until kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list &>/dev/null 2>&1; do
    echo "  Waiting for Kafka broker ($WAITED/$MAX_WAIT s)..."
    sleep 2
    WAITED=$((WAITED+2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Kafka broker not ready after ${MAX_WAIT}s"
        exit 1
    fi
done

echo "  Kafka broker ready."
echo ""

# ── neural-eeg-raw ────────────────────────────────────────────────
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic neural-eeg-raw \
    --partitions 8 \
    --replication-factor 1 \
    --config retention.ms=3600000 \
    --config cleanup.policy=delete \
    --config compression.type=lz4 \
    --config max.message.bytes=65536
echo "  ✓ neural-eeg-raw (8 partitions, 1h retention, lz4)"

# ── neural-eeg-clean ──────────────────────────────────────────────
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic neural-eeg-clean \
    --partitions 4 \
    --replication-factor 1 \
    --config retention.ms=3600000 \
    --config cleanup.policy=delete \
    --config compression.type=lz4
echo "  ✓ neural-eeg-clean (4 partitions, 1h retention)"

# ── neural-eeg-features ───────────────────────────────────────────
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic neural-eeg-features \
    --partitions 4 \
    --replication-factor 1 \
    --config retention.ms=3600000 \
    --config cleanup.policy=delete \
    --config compression.type=lz4
echo "  ✓ neural-eeg-features (4 partitions, 1h retention)"

# ── bci-commands ──────────────────────────────────────────────────
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic bci-commands \
    --partitions 1 \
    --replication-factor 1 \
    --config retention.ms=86400000 \
    --config cleanup.policy=delete
echo "  ✓ bci-commands (1 partition, 24h retention — audit trail)"

echo ""
echo "All topics created. Summary:"
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list

echo ""
echo "Bandwidth estimate per user:"
echo "  250 Hz × 8 channels × 4 bytes = 8,000 bytes/s = ~8 KB/s"
echo "  100 concurrent users = ~800 KB/s total"
echo ""
echo "GDPR compliance:"
echo "  Raw EEG purged after 1 hour (retention.ms=3600000)"
echo "  Commands retained 24h for audit (ISO 14155 §14)"