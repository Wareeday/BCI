#!/bin/bash
# scripts/setup_kafka.sh
# ======================
# Create all Kafka topics for the BCI platform.
#
# Run after Kafka is started:
#   docker-compose up -d kafka
#   chmod +x scripts/setup_kafka.sh
#   ./scripts/setup_kafka.sh
#
# Topics:
#   neural-eeg-raw      8 partitions (1/channel), retention=1h (GDPR)
#   neural-eeg-clean    4 partitions
#   neural-eeg-features 4 partitions
#   bci-commands        1 partition

set -e

BOOTSTRAP="localhost:9092"
echo "Creating BCI Platform Kafka topics on $BOOTSTRAP..."

# Wait for Kafka to be ready
until kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list > /dev/null 2>&1; do
  echo "Waiting for Kafka..."
  sleep 2
done

# neural-eeg-raw: 8 partitions (1 per channel), 1h retention
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic neural-eeg-raw \
  --partitions 8 \
  --replication-factor 1 \
  --config retention.ms=3600000 \
  --config compression.type=lz4

# neural-eeg-clean: DSP-filtered stream
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic neural-eeg-clean \
  --partitions 4 \
  --replication-factor 1 \
  --config retention.ms=3600000

# neural-eeg-features: feature vectors for ML
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic neural-eeg-features \
  --partitions 4 \
  --replication-factor 1 \
  --config retention.ms=3600000

# bci-commands: decoded motor intention commands
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" \
  --create --if-not-exists \
  --topic bci-commands \
  --partitions 1 \
  --replication-factor 1 \
  --config retention.ms=86400000   # 24h for audit

echo ""
echo "Topics created:"
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list
echo ""
echo "Done. Bandwidth per user: 250Hz × 8ch × 4B = 8 KB/s"
echo "GDPR: raw EEG purged after 1 hour (retention.ms=3600000)"