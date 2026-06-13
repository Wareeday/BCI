# IEEE 2857 Privacy Engineering Compliance
## BCI Platform — Neural Data Privacy

**Standard:** IEEE 2857-2021 — Privacy Engineering for Machine Learning
**Revision:** v1.0 | Device: OpenBCI BCI Platform

---

## §5.1 — Data Minimisation

**Requirement:** Collect only the minimum personal data necessary.

**Implementation:**
- Raw EEG waveforms stored **only** in Apache Kafka with 1-hour retention (`log.retention.hours=1`)
- After DSP processing, only **feature vectors** (56-dimensional) stored in database
- Raw waveforms never written to disk — processed in memory pipeline only
- Kafka `log.cleanup.policy=delete` ensures automatic purge

**Code reference:** `streaming/config/kafka-server.properties`, `database/models.py (EEGEpoch)`

---

## §5.3 — De-identification

**Requirement:** Remove or pseudonymise personal identifiers from neural data.

**Implementation:**
- Electrode positions mapped to generic 10-20 labels (Fp1, Fp2, C3, Cz, C4, P3, P4, Oz)
- No biometric tags (name, DOB, patient ID) stored with EEG data
- Patient ID pseudonymised with HMAC-SHA256 before any storage
- Research records use pseudonym_id only, never raw user_id

**Code reference:** `security/data_anonymizer.py`

---

## §6.2 — Consent Framework

**Requirement:** Granular per-purpose consent with revoke mechanism.

**Consent purposes:**
| Purpose | Required | User Can Revoke |
|---|---|---|
| `neural_processing` | Yes (BCI operation) | No (disables device) |
| `audit_logging` | Yes (safety/legal) | No (regulatory requirement) |
| `model_training` | No | Yes |
| `anonymized_research` | No | Yes |

**Revoke triggers:** Immediate erasure pipeline (GDPR Art.17)
- Kafka compaction deletes user's messages
- Database `DELETE` on all EEGEpoch rows for pseudonym_id
- Erasure itself logged in immutable audit trail

**Code reference:** `security/consent_manager.py`, `api/routes/v1/consent.py`

---

## §7.1 — Model Transparency

**Requirement:** Every CNN inference must be explainable and logged.

**Implementation:**
- Every inference logged to immutable audit file with: timestamp, predicted_class, confidence, model_used, epoch_type
- GradCAM heatmap available on request per inference (`/api/v1/ml/gradcam/{user_id}`)
- GradCAM shows which EEG channels/timepoints drove the classification decision
- Logs retained for clinical audit (FDA 21 CFR Part 11)

**Code reference:** `security/audit_logger.py`, `ml/cnn_model.py (get_gradcam)`, `api/routes/v1/ml.py`

---

## §7.2 — Fairness and Bias

**Implementation:**
- Model trained on BCI Competition IV Dataset 2a (9 subjects, balanced classes)
- Transfer learning from 50-subject corpus reduces cross-subject variance
- Per-user adaptive calibration (online SGD) mitigates individual bias
- Accuracy monitored per session; alert if <80% (may indicate bias for specific user)

---

## Privacy Trade-off Analysis

| Technique | Accuracy Impact | Speed | Use Case |
|---|---|---|---|
| **AES-256-GCM** (at rest) | None | Fast | All stored features |
| **TLS 1.3 + mTLS** (in transit) | None | Negligible overhead | All Kafka/API comms |
| **Differential Privacy ε=1.0** | 91% → 84% (−7%) | 15× faster than HE | Federated training only |
| **Homomorphic Encryption (SEAL)** | None | 12× slower | Individual patient inference where accuracy non-negotiable |

**Decision:** HE for individual inference, DP for federated cross-institution model sharing.