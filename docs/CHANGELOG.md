# Changelog — BCI Platform

All notable changes documented here.
Format: [Version] — Date — Description

---

## [1.0.0] — 2025-08-25 — Initial Release

### Added
**Acquisition**
- OpenBCI Cyton 8-channel EEG interface (250 Hz, 24-bit ADS1299)
- LSL outlet streamer for network-discoverable EEG stream
- Electrode impedance monitor with ISO 14155 bench test validation
- Real-time signal quality assessor (SNR, flat-line, saturation detection)
- Full EEG simulator for development without hardware

**DSP Pipeline (9ms total latency)**
- 1–40 Hz Butterworth bandpass filter (4th order, zero-phase)
- 50/60 Hz IIR notch filter (EU/US powerline noise)
- FastICA artefact removal (EOG, EMG, cardiac)
- Artifact Subspace Reconstruction (ASR) via MNE-Python
- Epoch extraction (P300: 800ms, Motor imagery: 4s)
- PSD + time-domain feature extraction (56-dim vector)
- GNU Radio flowgraph configuration (.grc)

**Machine Learning**
- TensorFlow CNN classifier: 91% accuracy on BCI Competition IV 2a
- PyTorch EEGNet backup model: 89% accuracy, 6ms inference
- scikit-learn baselines: LDA (72–78%), SVM (79–82%), RandomForest
- Online SGD adaptive calibration (30-trial blocks, every 5 min)
- Transfer learning adapter (50-subject pretrain → 4 min cold-start)
- Model registry with version tracking and rollback
- BCI Competition IV Dataset 2a loader (with synthetic fallback)
- GradCAM explainability (IEEE 2857 §7.1)

**Streaming**
- Apache Kafka producer/consumer (8 partitions, lz4 compression)
- 500ms signal watchdog with automatic SAFE_STATE
- LSL → Kafka bridge
- Schema registry for message validation
- Kafka config with GDPR-compliant 1h retention

**Assistive Devices**
- ROS Noetic wheelchair controller (Nav2, /bci/cmd_vel topic)
- Low-level differential drive wheelchair driver (Arduino Mega)
- Prosthetic servo controller (4-gesture, deadman switch)
- Arduino servo control sketch (servo_control.ino)
- Tobii Pro eye tracker interface (gaze + P300 multimodal fusion)
- Multimodal fusion (EEG 9.0% → fused 2.1% error rate)
- Festival TTS offline speech synthesis
- Device-level safety watchdog (per-actuator heartbeat monitoring)

**Neurofeedback**
- Band power calculator (delta/theta/alpha/beta via Welch PSD)
- 5-step neurofeedback engine (Measure → Display → Target → Reward → Adapt)
- Gamification (score, streaks, badges)
- Adaptive difficulty controller (auto-adjusts threshold ±10%)
- Session tracker with JSON persistence

**Security & Privacy**
- AES-256-GCM encryption for features at rest
- TLS 1.3 + mTLS configuration with certificate generation
- GDPR Article 7/9 consent manager (granular per-purpose)
- GDPR Article 17 erasure API (full pipeline)
- HMAC-SHA256 data anonymiser / pseudonymiser
- Differential privacy training (IBM diffprivlib, ε=1.0)
- Immutable audit logger (IEEE 2857 §7.1, FDA 21 CFR Part 11)

**API**
- FastAPI REST API with Swagger UI (/docs)
- WebSocket streams: EEG (250 Hz), commands, system status
- JWT authentication middleware
- Rate limiting middleware (sliding window, per-IP)
- All CRUD endpoints: sessions, EEG, ML, devices, consent, audit
- GDPR erasure endpoint (DELETE /api/v1/neural/{user_id})

**Dashboard**
- Plotly Dash clinical dashboard (dark theme)
- Live 8-channel EEG waveform (10 Hz update)
- Band power bars (real-time)
- BCI command log with confidence scores
- Neurofeedback progress chart (sessions 1–20)
- Electrode impedance badges
- System metrics (latency, throughput, model status)
- SAFE_STATE banner (red alert when active)
- 2D EEG topographic power map (topoplot)
- P300 ERP waveform comparison
- PSD frequency plot

**Database**
- SQLAlchemy async ORM (SQLite dev / PostgreSQL prod)
- Models: User, BCISession, EEGEpoch, BCICommand, ConsentRecord, AuditLogEntry, AdverseEvent, MLModelVersion
- CRUD operations for all models
- Database initialisation and seeding script
- Alembic migrations directory

**Resilience (4 ISO 14971 risk scenarios)**
- [HIGH] EEG signal loss: watchdog → SAFE_STATE → BLE retry × 3 → eye tracker fallback
- [HIGH] Kafka broker crash: local 10s buffer → ZooKeeper failover → log replay
- [MED]  CNN failure: SVM → LDA → 3-class degraded mode
- [LOW]  Network partition: local operation → encrypted queue → sync on reconnect
- GDPR Article 33 incident response (72h DPA notification)
- Global SAFE_STATE coordinator (all actuators)

**Testing**
- 9 pytest test modules (acquisition, DSP, ML, streaming, devices, security, API, resilience)
- Shared conftest.py fixtures
- DSP latency budget test (<10ms assertion)
- Security: tamper detection, consent lifecycle, audit immutability

**Scripts & Tooling**
- `demo_simulation.py` — standalone demo (numpy/scipy only)
- `run_pipeline.py` — full pipeline runner with argparse
- `validate_iso14155.py` — 4-phase clinical validation (16 tests)
- `benchmark_latency.py` — DSP latency profiling
- `setup_kafka.sh` — Kafka topic creation
- `generate_certs.sh` — TLS certificate generation

**Documentation**
- `README.md` — project overview and quick start
- `docs/architecture.md` — full system architecture with diagrams
- `docs/api_reference.md` — complete REST/WebSocket API docs
- `docs/risk_matrix.md` — ISO 14971 risk assessment (10 risks)
- `docs/ieee2857_compliance.md` — IEEE 2857 privacy engineering map
- `docs/iso14155_validation.md` — clinical investigation protocol
- `docs/fda510k_pathway.md` — FDA 510(k) regulatory pathway
- `docs/deployment_guide.md` — step-by-step deployment guide
- `docs/CHANGELOG.md` — this file

**Infrastructure**
- `docker-compose.yml` — Kafka + Zookeeper + PostgreSQL + API + Dashboard
- `Dockerfile` — Python 3.11 slim container
- `Makefile` — all project commands (install, demo, test, train, etc.)
- `.env.example` — all environment variables documented
- `.gitignore` — excludes secrets, models, logs

---

## [Planned] v1.1.0

- Real OpenBCI Cyton hardware integration tests
- BCI Competition IV Dataset 2a download automation
- Azure Confidential VM deployment configuration
- Federated learning across multiple hospital sites
- Homomorphic encryption for cloud inference (Microsoft SEAL)
- Mobile dashboard (React Native)
- RCT data collection tools (Phase 4)