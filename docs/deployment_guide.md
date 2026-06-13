# BCI Platform — Deployment Guide

## Prerequisites

- Python 3.10+
- Docker + Docker Compose (for Kafka + PostgreSQL)
- Git

**For real hardware (optional):**
- OpenBCI Cyton board + USB dongle
- Arduino Mega (prosthetic control)
- Raspberry Pi 4 (TTS + LSL gateway)

---

## Quick Start (Simulation Mode — No Hardware Required)

```bash
# 1. Clone and enter project
git clone https://github.com/wardamasood/bci-platform.git
cd bci-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENBCI_SIMULATE=true for no hardware

# 5. Start infrastructure (Kafka + PostgreSQL)
docker-compose up -d zookeeper kafka postgres
# Wait ~15 seconds for Kafka to be ready

# 6. Create Kafka topics
chmod +x scripts/setup_kafka.sh
./scripts/setup_kafka.sh

# 7. Run the demo simulation (no dependencies beyond numpy/scipy)
python scripts/demo_simulation.py

# 8. Run the ISO 14155 validation script
python scripts/validate_iso14155.py

# 9. Run latency benchmark
python scripts/benchmark_latency.py

# 10. Start full pipeline (simulation mode)
python scripts/run_pipeline.py --simulate --paradigm motor_imagery
```

---

## Start the API Server

```bash
# Terminal 1: API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Open: http://localhost:8000/docs  (Swagger UI)
# Open: http://localhost:8000/health
```

---

## Start the Clinical Dashboard

```bash
# Terminal 2: Plotly Dash dashboard
python dashboard/app.py

# Open: http://localhost:8050
```

---

## Train the Models

```bash
# Compare all classifiers (LDA, SVM, CNN, EEGNet)
python ml/train.py --model compare

# Train CNN only (50 epochs, motor imagery)
python ml/train.py --model cnn --paradigm motor_imagery --epochs 50

# Train EEGNet backup
python ml/train.py --model eegnet --epochs 50
```

---

## Run Tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=. --cov-report=html

# Individual modules
pytest tests/test_dsp.py -v          # DSP latency tests
pytest tests/test_security.py -v     # GDPR/encryption tests
pytest tests/test_resilience.py -v   # failure scenario tests
pytest tests/test_api.py -v          # API endpoint tests
```

---

## Real Hardware Setup (OpenBCI Cyton)

```bash
# 1. Find serial port
ls /dev/ttyUSB*    # Linux
ls /dev/cu.*       # Mac

# 2. Set in .env
OPENBCI_PORT=/dev/ttyUSB0
OPENBCI_SIMULATE=false

# 3. Run with real hardware
python scripts/run_pipeline.py --port /dev/ttyUSB0 --simulate false
```

---

## Production Deployment Checklist

- [ ] Generate TLS certificates: `chmod +x scripts/generate_certs.sh && ./scripts/generate_certs.sh`
- [ ] Set `KAFKA_SECURITY_PROTOCOL=SSL` in `.env`
- [ ] Set strong `AES_KEY_HEX` (32 bytes): `openssl rand -hex 32`
- [ ] Set `API_SECRET_KEY`: `openssl rand -hex 32`
- [ ] Set `DATABASE_URL` to PostgreSQL
- [ ] Set `default.replication.factor=2` in kafka-server.properties
- [ ] Enable mTLS on Kafka (`ssl.client.auth=required`)
- [ ] Run OpenSCAP compliance scan
- [ ] Rotate TLS certificates every 90 days (NIST SP 800-66)
- [ ] Configure Azure Confidential VM for audit log storage
- [ ] Run `scripts/validate_iso14155.py` and attach report to FDA submission

---

## Architecture Summary

```
OpenBCI (250Hz) → LSL → GNU Radio DSP (9ms) → Kafka → CNN (8ms) → ROS → Wheelchair
                                                              ↓
                                                     AuditLogger (IEEE 2857 §7.1)
```

**Total end-to-end latency:** ~87ms ✓ (target <100ms)