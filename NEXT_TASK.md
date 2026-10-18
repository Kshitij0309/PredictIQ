# PredictIQ — Next Task: Phase 5 (Production REST API Service & Deployment)

## 1. Task Definition
**Task Name:** Phase 5 — FastAPI Production Inference Service, Schema Validation & Containerized Deployment  
**Objective:** Package the trained and calibrated churn prediction champion model into a production-grade FastAPI REST service supporting single-record scoring, batch inference, health checks, metadata introspection, and containerized execution via Docker Compose.

---

## 2. Scope & Boundaries
- **In Scope:**
  1. **FastAPI Application (`src/predictiq/api/`)**:
     - Modern asynchronous FastAPI application with CORS middleware, request ID tracking, and structured logging.
     - Pydantic v2 data models for input customer records (`CustomerInputSchema`) and prediction responses (`PredictionResponseSchema`, `BatchPredictionResponseSchema`).
     - Real-time schema validation ensuring strict compatibility with `DomainFeatureEngineer`.
  2. **API Endpoints**:
     - `GET /health`: Readiness and liveness probes reporting model version, uptime, and memory status.
     - `GET /model/metadata`: Exposes model configuration, calibration score, optimal threshold ($p^* = 0.3400$), and top SHAP features.
     - `POST /predict`: Single customer churn scoring returning predicted churn label, calibrated probability, decision threshold, and risk tier.
     - `POST /predict/batch`: High-throughput batch prediction endpoint.
  3. **Inference Pipeline Integration**:
     - Seamless integration with `ChurnPredictor` loading `models/churn_champion_model.joblib`.
     - Zero-downtime model loading on application startup.
  4. **Containerization & Docker Compose**:
     - Multi-stage Dockerfile for the API service.
     - Docker Compose integration connecting FastAPI, MLflow, and PostgreSQL.
  5. **Deterministic Unit & API Tests**:
     - Pytest test suite using `httpx.AsyncClient` or `starlette.testclient.TestClient`.
     - Edge-case testing: missing features, invalid types, extreme values, batch payload limits.
- **Out of Scope (Deferred to Phase 6):**
  - Drift detection and model performance monitoring (Evidently AI / Prometheus).
  - CI/CD automated deployment pipelines.

---

## 3. Prerequisites
- [x] Phase 1 Project Foundation verified.
- [x] Phase 2 Dataset Acquisition, Validation & EDA verified.
- [x] Phase 3 Feature Engineering & MLflow Tracking verified.
- [x] Phase 4 Evaluation Protocol Correction, Tuning, Calibration, Thresholding & SHAP verified (97 passing tests, champion model serialized with $p^* = 0.3400$).
- [ ] Add `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `httpx>=0.27.0` to `pyproject.toml`.

---

## 4. Acceptance Criteria
1. FastAPI service initializes and passes automated test client requests for `/health`, `/model/metadata`, and `/predict`.
2. Prediction latency is under 50ms for single-record scoring.
3. Input validation rejects invalid types or missing required features with descriptive HTTP 422 errors.
4. Threshold $p^* = 0.3400$ is consistently applied during API scoring.
5. All existing 97 tests continue to pass alongside new API tests.

---

## 5. Verification Commands
```powershell
# 1. Run all unit and API tests
& "D:\Projects\PredictIQ\.venv\Scripts\python.exe" -m pytest -v

# 2. Run Ruff linter
& "D:\Projects\PredictIQ\.venv\Scripts\ruff.exe" check .

# 3. Launch local FastAPI server for manual testing
& "D:\Projects\PredictIQ\.venv\Scripts\uvicorn.exe" predictiq.api.main:app --host 0.0.0.0 --port 8000 --reload
```
