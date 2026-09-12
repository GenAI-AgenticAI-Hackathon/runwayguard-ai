# ⚠️ RunwayGuard AI — Engineering Disclosures & Limitations

RunwayGuard AI commits to technical honesty. In accordance with professional engineering ethics, this document explicitly details the boundary between operational code and simulated capabilities.

---

## 1. Frame-by-Frame Processing vs. Real-Time Video Tracking

* **Deck Claim:** "Continuous real-time multi-camera runway video tracking."
* **Reality in Code:** The system implements sequential frame ingestion via Groq Vision. Each frame is processed as an individual multimodal completion.
* **Why:** Hosted vision LLMs introduce a 1.0–2.5 second network and inference latency per frame. Real-time 30 FPS video tracking requires local edge deployment (e.g., TensorRT-optimized YOLOv8 or ByteTrack) rather than remote cloud API calls.
* **Current Honest Implementation:**
  * Single-image diagnostic analysis via manual upload, webcam snapshot, or scenario injection.
  * Directory processing simulating sequential frame analysis without claiming true cross-frame persistent object ID tracking.

---

## 2. Agent Action Physical Execution

* **Deck Claim:** "Autonomous runway traffic control and automated vehicle deployment."
* **Reality in Code:** The agent executes real, concrete Python functions (`halt_traffic()`, `alert_ground_crew()`, `dispatch_sweep_team()`, `log_only()`) triggered via native Groq function calling (`tool_choice="required"`).
* **Limitation:** In this environment, these functions format operational dispatch messages and write to stdout / audit logs. They do not interface with live airport ATC radio frequencies or physical telemetry gateways (e.g., FAA SWIM API or Eurocontrol NM B2B).

---

## 3. Session-Based In-Memory Audit Persistence

* **Current Implementation:** The audit trail is stored in an in-memory session log and rendered to the Gradio UI and incident reports.
* **Limitation:** Audit records are not persisted to a PostgreSQL/TimescaleDB time-series database across server restarts. In a certified aviation deployment, every audit entry would be cryptographically hashed and written to a write-once tamper-evident datastore.

---

## 4. Vision Model Confidence Calibration

* **Current Implementation:** The confidence metric ($0.0 \le c \le 1.0$) is returned directly by the vision LLM based on visual fidelity.
* **Limitation:** LLM-generated confidence figures are self-assessed and not statistically calibrated softmax probabilities from a dedicated discriminative classifier. They serve as an operational heuristic rather than a certified risk probability.

---

*Authored by the RunwayGuard AI Engineering Team.*
