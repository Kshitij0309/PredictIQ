# PredictIQ — Next Task: Phase 6 (Drift Monitoring, Prediction Logging & Observability)

## 1. Task Definition
**Task Name:** Phase 6 — Prediction Logging, Data & Concept Drift Monitoring, and Production Observability  
**Objective:** Instrument the PredictIQ production serving system with continuous monitoring capabilities to detect data drift, prediction drift, and concept drift over time; log inference requests safely; and expose Prometheus metrics for operational dashboards.

---

## 2. Scope & Boundaries
- **In Scope:**
  1. **Asynchronous Prediction Logging (`src/predictiq/monitoring/logging.py`)**:
     - Background logging of incoming customer payloads and model prediction scores.
     - Privacy-preserving pseudonymization (hashing customer IDs if present, retaining feature distributions only).
     - Storage options: structured JSONL or lightweight SQLite/PostgreSQL telemetry table.
  2. **Data & Prediction Drift Detection (`src/predictiq/monitoring/drift.py`)**:
     - Baseline distribution capture from the Phase 4 training dataset.
     - Numerical drift metrics: Kolmogorov-Smirnov (KS) test, Wasserstein Distance, Population Stability Index (PSI).
     - Categorical drift metrics: Chi-Square test, Jensen-Shannon Divergence.
     - Scheduled or on-demand drift report generation comparing serving windows to baseline.
  3. **Prometheus Observability Metrics (`src/predictiq/monitoring/metrics.py`)**:
     - Serving metrics: request throughput, p50/p95/p99 latency histograms, HTTP status counters.
     - ML metrics: prediction count by risk tier (Low/Medium/High), rolling mean churn probability.
     - `/metrics` endpoint mounted to FastAPI application.
  4. **Alerting Triggers & Thresholds**:
     - Automated warning thresholds for severe feature drift (e.g., PSI > 0.25).
     - Health alert if predicted churn distribution shifts significantly from baseline (26.5%).
  5. **Automated Monitoring Tests**:
     - Unit tests for drift calculation functions.
     - Integration tests verifying prediction logging does not add latency to the `/predict` critical path.
- **Out of Scope (Deferred to Future Phases):**
  - Automated continuous retraining or automated model redeployment.
  - Kubernetes / Helm deployment manifests.

---

## 3. Prerequisites
- [x] Phase 1 Project Foundation verified.
- [x] Phase 2 Dataset Acquisition, Validation & EDA verified.
- [x] Phase 3 Feature Engineering & MLflow Tracking verified.
- [x] Phase 4 Evaluation Protocol Correction, Tuning, Calibration, Thresholding & SHAP verified.
- [x] Phase 5 FastAPI Production Inference Service, Containerization, and Latency Benchmarking verified (113 passing tests, Docker container validated).

---

## 4. Acceptance Criteria
1. Prediction payloads and scores are logged asynchronously with zero impact on inference latency.
2. Drift detection module calculates PSI and KS test statistics against baseline training distributions.
3. `/metrics` endpoint exposes Prometheus-formatted counters and latency histograms.
4. Comprehensive unit and integration tests added, maintaining 100% test pass rate and clean Ruff linting.

---

## 5. Verification Commands
```powershell
# 1. Run all unit, API, and monitoring tests
& "D:\Projects\PredictIQ\.venv\Scripts\python.exe" -m pytest -v

# 2. Run Ruff linter
& "D:\Projects\PredictIQ\.venv\Scripts\ruff.exe" check .

# 3. Test metrics endpoint
curl http://localhost:8000/metrics
```
