# BCI Platform — API Reference

**Base URL:** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)
**Auth:** Bearer JWT token (dev mode: no auth required)

---

## System

### `GET /health`
System health check. Used by Docker healthcheck and monitoring.

**Response:**
```json
{"status": "healthy", "session_active": false, "audit_entries": 42}
```

### `GET /`
API root with version and standards information.

---

## Sessions — `/api/v1/sessions`

### `POST /api/v1/sessions/start`
Start a new BCI session.

**Request body:**
```json
{"user_id": "patient_001", "paradigm": "motor_imagery", "simulate": true}
```
`paradigm`: `"motor_imagery"` or `"p300"`

**Response:**
```json
{"session_id": "A3F1B2C4", "user_id": "patient_001", "paradigm": "motor_imagery",
 "started_at": 1700000000.0, "status": "active"}
```

### `POST /api/v1/sessions/stop/{session_id}`
Stop an active session.

### `GET /api/v1/sessions/active`
List all active sessions.

---

## EEG — `/api/v1/eeg`

### `GET /api/v1/eeg/latest`
Return the most recent EEG sample (for dashboard polling).

**Response:**
```json
{"timestamp": 1700000000.0, "channels": [1.2, -0.5, 2.1, 0.8, -1.4, 3.2, 0.1, -0.9],
 "sample_id": 1234, "quality_score": 0.95}
```

### `GET /api/v1/eeg/features`
Return the most recent feature vector (56-dim).

### `GET /api/v1/eeg/quality`
Return electrode impedance and SNR metrics.

**Response:**
```json
{
  "snr_db": 42.0,
  "channels_ok": 8,
  "total_channels": 8,
  "target_met": true,
  "impedances_kohm": {"0": 2.1, "1": 1.8, ...},
  "standard": "ISO 14155 Phase 1 Bench Test (target SNR >35dB)"
}
```

---

## Machine Learning — `/api/v1/ml`

### `GET /api/v1/ml/status`
Current model status and performance metrics.

**Response:**
```json
{
  "primary_model": "CNN (TensorFlow 2.14)",
  "accuracy": 0.91,
  "inference_ms": 8.0,
  "fallback_model": "LDA (scikit-learn 1.3)",
  "confidence_threshold": 0.85
}
```

### `GET /api/v1/ml/predict/demo`
Demo inference on synthetic epoch.

**Response:**
```json
{
  "predicted_class": 0,
  "class_name": "left",
  "confidence": 0.9134,
  "probabilities": {"left": 0.9134, "right": 0.0521, "feet": 0.0231, "rest": 0.0114},
  "model": "CNN",
  "inference_ms": 8.2,
  "decision": "issue"
}
```

### `GET /api/v1/ml/gradcam/{user_id}`
GradCAM explanation for last prediction — **IEEE 2857 §7.1**.

**Response:**
```json
{
  "user_id": "patient_001",
  "heatmap": [[...], ...],
  "channel_importance": {"Fp1": 0.12, "C3": 0.78, ...},
  "ieee_2857_section": "§7.1 Model Transparency"
}
```

---

## Devices — `/api/v1/devices`

### `GET /api/v1/devices/status`
Current state of all connected devices.

### `POST /api/v1/devices/override`
Manual clinician override — takes priority over BCI commands.

**Request body:**
```json
{"device": "wheelchair", "command": "stop", "reason": "clinician_override"}
```

### `POST /api/v1/devices/safe-state`
Activate SAFE_STATE on all devices — emergency stop.

**Query param:** `reason=manual`

---

## Consent / GDPR — `/api/v1/consent`

### `POST /api/v1/consent/{user_id}/grant`
Grant GDPR consent — **required before any EEG processing**.

**Request body:**
```json
{"purposes": ["neural_processing", "audit_logging"]}
```
Optional purposes: `"model_training"`, `"anonymized_research"`

### `POST /api/v1/consent/{user_id}/revoke`
Revoke consent — triggers **GDPR Article 17 erasure pipeline**.

**Query param:** `purpose=model_training` (omit to revoke all)

**Response:**
```json
{
  "user_id": "patient_001",
  "erasure_triggered": true,
  "steps_completed": ["db_records_deleted", "kafka_compaction_triggered", "audit_logged"],
  "gdpr_article": "Article 17 — Right to erasure"
}
```

### `GET /api/v1/consent/{user_id}`
Check consent status.

### `DELETE /api/v1/neural/{user_id}`
GDPR Article 17 — immediate erasure of all neural data.

---

## Audit — `/api/v1/audit`

### `GET /api/v1/audit/`
Query immutable audit log — **IEEE 2857 §7.1**.

**Query params:** `event_type`, `user_id`, `limit` (max 500)

### `GET /api/v1/audit/sae`
Return all Serious Adverse Events — **ISO 14155 §14**.

---

## WebSocket Streams

### `WS /ws/eeg`
Stream raw EEG at 250 Hz.
```json
{"timestamp": 1700000000.0, "channels": [1.2, -0.5, ...], "sample_id": 1234, "type": "eeg_sample"}
```

### `WS /ws/commands`
Stream decoded BCI commands.
```json
{"timestamp": 1700000000.0, "command": "left", "confidence": 0.91, "decision": "issue", "model": "cnn"}
```

### `WS /ws/status`
Stream system health metrics every second.
```json
{"eeg_streaming": true, "kafka_healthy": true, "safe_state": false, "dsp_latency_ms": 7.8}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| 401 | Missing or invalid JWT token |
| 404 | Resource not found (session, user, device) |
| 422 | Validation error (check request body) |
| 429 | Rate limit exceeded (see `X-RateLimit-*` headers) |
| 500 | Internal server error (check logs/bci_platform.log) |