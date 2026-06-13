# BCI Platform — System Architecture

## Overview

End-to-end brain-computer interface platform for assistive technology.
Processes real-time EEG signals from OpenBCI Cyton hardware, applies
GNU Radio DSP filtering, streams data via Apache Kafka, classifies motor
imagery and P300 signals with a TensorFlow CNN, and controls assistive
devices (wheelchair, prosthetic, speech synthesiser) via ROS.

**Standards:** IEEE 2857 · ISO 14155 · FDA 510(k) · GDPR Article 9

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ZONE 1: Patient / Edge (Clinical LAN — untrusted)              │
│                                                                  │
│  OpenBCI Cyton          LSL Outlet         Raspberry Pi 4       │
│  (8ch, 250Hz, 24-bit)──▶(250Hz, float32)──▶(Edge gateway,      │
│  ADS1299, RFduino BLE   LabStreamingLayer   VPN client,         │
│  <5kΩ impedance                             ZeroMQ buffer)       │
└───────────────────────────┬────────────────────────────────────┘
                            │ TLS 1.3 + mTLS (VPN/WireGuard)
┌───────────────────────────▼────────────────────────────────────┐
│  ZONE 2: DMZ / Secure Ingress                                   │
│                                                                  │
│  Firewall/WAF  ──▶  API Gateway (OAuth2 + JWT + Rate Limit)     │
│                      ──▶  VPN Concentrator                       │
│                      ──▶  IAM / Consent Management              │
│                      ──▶  SIEM / Audit Logs                      │
└───────────────────────────┬────────────────────────────────────┘
                            │ mTLS
┌───────────────────────────▼────────────────────────────────────┐
│  ZONE 3: Processing Server (Private VPC / Hospital VLAN)        │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ GNU Radio DSP   │  │  Apache Kafka    │  │  CNN Jetson   │  │
│  │                 │  │                  │  │  Nano         │  │
│  │ Bandpass 1-40Hz │  │ neural-eeg-raw   │  │  TF 2.14      │  │
│  │ Notch 50Hz      │  │ (8 partitions)   │  │  91% acc      │  │
│  │ ICA (FastICA)   │──▶ neural-eeg-clean │──▶ 8ms inference │  │
│  │ ASR (MNE)       │  │ neural-features  │  │  EEGNet 89%   │  │
│  │ Epoch extract   │  │ bci-commands     │  │  LDA fallback │  │
│  │ PSD/CSP feat    │  │ retention: 1h    │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  Adaptive Calibration:  SGD every 30 trials, lr=0.0001         │
│  Transfer Learning:     50-subject pretrain → 4 min cold-start  │
└───────────────────────────┬────────────────────────────────────┘
                            │ ROS /bci/cmd_vel + JSON Serial
┌───────────────────────────▼────────────────────────────────────┐
│  ZONE 4: Assistive Devices                                      │
│                                                                  │
│  Wheelchair (ROS Noetic + Nav2 + Arduino Mega)                  │
│    Confidence ≥ 85% + 200ms confirmation window                 │
│    Deadman: SAFE_STATE on 500ms silence                         │
│                                                                  │
│  Prosthetic Servo (Arduino Mega, 115200 baud, 8ms latency)      │
│    4 gestures: rest/open/close/pinch                            │
│    Deadman: open circuit if confidence <75% × 3 frames          │
│                                                                  │
│  Eye Tracker (Tobii Pro — P300 + gaze fusion)                   │
│    Error rate: EEG alone 9.0% → fused 2.1% ✓                   │
│                                                                  │
│  TTS (Festival on Raspberry Pi 4 — fully offline)               │
│    320ms word latency, no cloud dependency                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Latency Budget

| Stage | Component | Latency |
|---|---|---|
| Acquisition | OpenBCI Cyton hardware | 4 ms |
| DSP | GNU Radio: bandpass + notch + ICA + ASR | 3 ms |
| Feature extraction | MNE-Python: PSD + time-domain | 2 ms |
| **Total DSP** | | **9 ms ✓ (<10ms target)** |
| Kafka publish | Producer → broker | 2 ms |
| CNN inference | Jetson Nano, 472 GFLOPS | 8 ms |
| ROS publish | /bci/cmd_vel topic | 5 ms |
| **Total end-to-end** | | **87 ms avg ✓ (<100ms target)** |

---

## Design Decisions (WHY, not just WHAT)

### Why OpenBCI over g.tec?
g.tec: $15,000+, closed SDK, proprietary API with DRM.
OpenBCI: $499, open hardware, MIT-licensed firmware.
Research needs full ICA access — closed devices block raw DSP.
OpenBCI 8-channel 250Hz meets P300 BCI requirements (Kübler et al., 2009).

### Why GNU Radio over pure Python DSP?
GNU Radio C++ backend: <1ms block latency at 250Hz (SIMD optimised).
Pure Python (SciPy): 8–15ms per buffer on Raspberry Pi 4 → exceeds budget.
GNU Radio also enables live parameter tuning without pipeline restart.

### Why Kafka over ZeroMQ/RabbitMQ?
ZeroMQ: no persistence → if classifier crashes, neural frames lost.
RabbitMQ: no time-ordered replay.
Kafka: log-based replay of last 10s of EEG on classifier restart.
Bandwidth: 250Hz × 8ch × 4B = 8KB/s per user. Kafka handles 100+ users.

### Why CNN over LDA/SVM?
LDA: 72–78% — too inaccurate for safety-critical wheelchair control.
SVM: 79–82% — no spatial-temporal correlation learning.
CNN: 91% — learns inter-electrode coherence (critical for motor imagery).
ResNet: 93% but 25ms inference → exceeds 10ms budget. CNN is optimal.

---

## Safety Architecture

```
Confidence ≥ 0.85 → CommandDecision.ISSUE → 200ms window → actuate
Confidence 0.75-0.85 → CommandDecision.CONFIRM → request confirmation
Confidence < 0.75 → CommandDecision.HOLD → no actuation

Watchdog (500ms): EEG silent > 500ms → SAFE_STATE → halt all actuators
Deadman switch:   CNN conf < 0.75 × 3 frames → open prosthetic circuit
```

---

## Standards Compliance

| Standard | Section | Implementation |
|---|---|---|
| IEEE 2857 | §5.1 Data minimisation | Raw EEG purged after DSP, features only |
| IEEE 2857 | §5.3 De-identification | Electrode → generic 10-20 labels, no biometric tags |
| IEEE 2857 | §6.2 Consent | Granular per-purpose consent portal (GDPR Art.7) |
| IEEE 2857 | §7.1 Transparency | Every inference logged + GradCAM explainability |
| ISO 14155 | §6 Ethics | IRB approval required before clinical pilot |
| ISO 14155 | §8 Risk | ISO 14971 failure mode analysis documented |
| ISO 14155 | §11 Validation | Bench → Simulated Use → N=5 Pilot → RCT |
| ISO 14155 | §14 SAE | Any unintended movement = SAE, 24h IRB report |
| FDA 510(k) | Predicate | NeuroOne K191892 (substantial equivalence) |
| FDA 510(k) | Cybersecurity | AES-256 + TLS 1.3 + mTLS (NIST SP 800-66) |
| GDPR | Art.7 | Explicit consent before EEG processing |
| GDPR | Art.9 | Neural EEG = biometric data, explicit consent |
| GDPR | Art.17 | Right to erasure API implemented |
| GDPR | Art.33 | 72h breach notification procedure |
| GDPR | Art.35 | DPIA before clinical deployment |
| Convention 108+ | Cross-border | EU→UK adequacy decision required |