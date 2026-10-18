# PredictIQ — Predictive Intelligence Platform

PredictIQ is a modular, production-grade predictive intelligence platform built to demonstrate advanced competencies across the complete machine learning lifecycle—from exploratory data analysis and feature engineering to experiment tracking, model registry, low-latency API serving, containerized deployment, and drift monitoring.

---

## 1. Project Purpose & Target Users

- **Data Scientists:** Reproducible exploratory data analysis, hypothesis-driven feature transformations, statistical validation, calibration, and model explainability.
- **Machine Learning Engineers:** Production-ready code structure, 12-factor configuration, containerized microservices, FastAPI serving, asynchronous batch processing, automated testing, and CI/CD readiness.
- **Hiring Managers & Technical Evaluators:** A transparent, end-to-end reference implementation demonstrating rigorous engineering practices, architecture documentation, latency benchmarking, and clean system design.

---

## 2. Core Objectives

1. **Reproducibility:** Deterministic environments, versioned dependencies, pinned Docker images, and isolated local development workflows.
2. **Modular Architecture:** Clear boundary separation between data layers, analysis notebooks, core application packages, serving APIs, and infrastructure.
3. **Operational Rigor:** Fast automated testing (113 passing tests), typed configuration with input validation, zero committed secrets, non-root container security, and health-monitored infrastructure.
4. **Phased Delivery:** Disciplined incremental execution where each phase is verified against explicit acceptance criteria before proceeding.

---

## 3. Technology Stack

| Domain | Technology | Version / Specification | Rationale |
|---|---|---|---|
| **Language** | Python | `>=3.10` (Tested on `3.13.14`) | Modern type hints, dataclasses, wide ML library ecosystem |
| **API Framework** | FastAPI + Uvicorn | `fastapi>=0.110.0`, `uvicorn>=0.28.0` | Asynchronous ASGI framework, OpenAPI docs, threadpool offloading |
| **Data Validation** | Pydantic | `>=2.0.0` (v2) | Strict schema validation, automatic serialization, clean errors |
| **Packaging** | Setuptools / PEP 517 | Standard `pyproject.toml` | Standardized Python packaging with editable install support |
| **Testing & Linting** | Pytest & Ruff | `pytest>=8.0.0`, `ruff>=0.4.0` | Deterministic unit testing, zero-config linter and formatter |
| **Machine Learning** | Scikit-Learn & LightGBM | `scikit-learn>=1.4.0`, `lightgbm>=4.0.0` | Calibrated classifier ensembles, hyperparameter tuning |
| **Explainability** | SHAP | `shap>=0.45.0` | TreeSHAP local and global feature attribution |
| **Containerization** | Docker & Compose | Compose Spec v2 / `python:3.13-slim` | Non-root security (UID 10001), healthcheck probe, minimal image |
| **Database** | PostgreSQL | `16.8-alpine` | High-performance relational backend store for MLflow tracking |
| **Experiment Tracking** | MLflow | `ghcr.io/mlflow/mlflow:v2.19.0` | Experiment metrics, parameter logging, and artifact registry |

---

## 4. Repository Structure

```
PredictIQ/
├── .env.example                     # Environment configuration template
├── .gitignore                        # Comprehensive ignore rules (data, venv, secrets, logs)
├── CURRENT_STATE.md                  # Verified milestone status & test logs
├── Dockerfile                        # Multi-layer production container (non-root, healthchecked)
├── docker-compose.yml                # API + PostgreSQL 16 & MLflow 2.19 orchestration
├── data/                             # Data layer with immutable isolation
│   ├── metadata/                     # Provenance metadata (Telco-Customer-Churn_metadata.json)
│   ├── raw/                          # Original immutable datasets (.gitkeep, Telco-Customer-Churn.csv)
│   ├── interim/                      # Cleaned intermediate transformations (.gitkeep, Telco-Customer-Churn-cleaned.csv)
│   ├── processed/                    # Model-ready features & splits (.gitkeep)
│   └── external/                     # Third-party benchmark data (.gitkeep)
├── docs/                             # Architecture & operational guides
│   ├── adr/
│   │   └── ADR-001-initial-architecture.md  # Accepted architecture decision record
│   ├── data/
│   │   └── DATASET_CARD.md           # Dataset provenance, schema, audit, and ethics
│   ├── ARCHITECTURE.md               # End-to-end system design & layer boundaries
│   ├── CHANGELOG.md                  # Versioned change history & validation notes
│   ├── DEVELOPMENT_SETUP.md          # Exact Windows PowerShell setup instructions
│   ├── PHASE_3_IMPLEMENTATION_REPORT.md # Phase 3 report
│   ├── PHASE_4_IMPLEMENTATION_REPORT.md # Phase 4 report (protocol correction, tuning, calibration)
│   └── PHASE_5_IMPLEMENTATION_REPORT.md # Phase 5 report (production API, latency, containerization)
├── models/                           # Persisted model artifacts & metadata
│   ├── churn_champion_model.joblib   # Champion pipeline (RandomForest + Platt CalibratedClassifierCV)
│   └── churn_champion_model_metadata.json # Optimal threshold (p*=0.3400), metrics, and features
├── NEXT_TASK.md                      # Phase 6 scope (Drift Monitoring & Observability)
├── notebooks/                        # Exploratory analysis & prototyping
│   ├── README.md                     # Notebook reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # End-to-end EDA & hypothesis testing notebook
├── pyproject.toml                    # Package build metadata & pytest configuration
├── README.md                         # Project overview and quickstart guide
├── reports/                          # Visual figures & evaluation tables
│   ├── figures/                      # Calibration curves, ROC/PR curves, SHAP waterfalls
│   └── phase4_final_test_summary.csv # Evaluated model comparison metrics
├── scripts/                          # Operational CLI scripts
│   ├── benchmark_latency.py          # Latency benchmark measuring p50 and p95
│   ├── train_baseline.py             # Baseline model training CLI
│   └── tune_and_train.py             # Hyperparameter tuning & calibration CLI
├── src/                              # Application source tree (src/ layout)
│   └── predictiq/
│       ├── __init__.py               # Package exports & version metadata
│       ├── config.py                 # Typed configuration, validation, & .env loader
│       ├── api/                      # Production REST API layer (FastAPI)
│       │   ├── __init__.py           # Exports app & create_app
│       │   ├── app.py                # App factory, lifespan loader, PII-safe logging middleware
│       │   ├── deps.py               # Dependency injection & readiness guard
│       │   ├── main.py               # Production Uvicorn ASGI entrypoint
│       │   ├── routes.py             # Route handlers (/health, /ready, /model-info, /predict)
│       │   └── schemas.py            # Pydantic v2 schemas for requests & responses
│       ├── data/                     # Data acquisition, validation, & cleaning
│       ├── features/                 # Domain feature engineering & preprocessing pipeline
│       ├── inference/                # ChurnPredictor engine with risk tiering
│       ├── models/                   # Training, tuning, calibration, thresholds, SHAP explainability
│       └── tracking/                 # MLflow experiment tracking manager
└── tests/                            # Automated test suite (113 passed)
    ├── test_api.py                   # 16 unit & integration tests for FastAPI service
    ├── test_calibration.py           # 2 unit tests for Platt scaling & Brier score
    ├── test_config.py                # 42 unit tests for config, ports, paths, & env
    ├── test_data_ingestion.py        # 9 unit tests for checksums, provenance, & acquisition
    ├── test_data_loader.py           # 7 unit tests for stratified splitting & leakage prevention
    ├── test_data_validation.py       # 12 unit tests for schema, constraints, & domain rules
    ├── test_evaluation_protocol.py   # 2 unit tests for protocol independence & fold stability
    ├── test_explainability.py        # 1 unit test for TreeSHAP attributions & waterfall plots
    ├── test_features.py              # 6 unit tests for feature engineering & transformations
    ├── test_inference.py             # 4 unit tests for serialization, loading, & risk tiers
    ├── test_model_evaluation.py      # 4 unit tests for evaluation metrics & curves
    ├── test_thresholds.py            # 3 unit tests for OOF threshold search & cost matrices
    ├── test_tracking.py              # 2 unit tests for MLflow tracking & fallback
    └── test_tuning.py                # 3 unit tests for hyperparameter search spaces
```

---

## 5. Prerequisites

Before running the platform, verify that your machine has:
1. **Python 3.10+** (verified on `3.13.14`)
2. **Git for Windows** (verified on `2.55.0.windows.3`)
3. **Docker Desktop** (optional for pure Python tests, recommended for containerized API and MLflow)

---

## 6. Quickstart & Local Execution

### Step 1: Clone Repository & Create Virtual Environment
```powershell
Set-Location -Path "D:\Projects\PredictIQ"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
```

### Step 2: Install Package with API & Development Dependencies
```powershell
& ".\.venv\Scripts\pip.exe" install -e ".[dev,api,analysis]"
```

### Step 3: Run Automated Test Suite & Linter
```powershell
# Run full 113-item test suite
& ".\.venv\Scripts\pytest.exe" -v

# Run Ruff linter
& ".\.venv\Scripts\ruff.exe" check .
```
*Expected: 113 passed in ~23s with zero warnings or errors.*

### Step 4: Launch Production FastAPI Service Locally
```powershell
# Start Uvicorn ASGI server
& ".\.venv\Scripts\python.exe" -m predictiq.api.main
```
The service initializes on `http://127.0.0.1:8000`.

---

## 7. REST API Endpoints & Usage Examples

### 7.1 Liveness Probe (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "environment": "production",
  "uptime_seconds": 12.45,
  "timestamp": "2026-10-18T16:00:00.000000Z"
}
```

### 7.2 Readiness Probe (`GET /ready`)
```bash
curl -X GET http://localhost:8000/ready
```
```json
{
  "ready": true,
  "model_loaded": true,
  "timestamp": "2026-10-18T16:00:00.000000Z",
  "model_name": "churn_champion_model",
  "model_version": "1.0.0"
}
```

### 7.3 Model Information (`GET /model-info`)
```bash
curl -X GET http://localhost:8000/model-info
```
```json
{
  "model_name": "churn_champion_model",
  "model_version": "1.0.0",
  "algorithm": "CalibratedClassifierCV(RandomForestClassifier)",
  "calibrated": true,
  "calibration_method": "sigmoid",
  "decision_threshold": 0.3400,
  "features_count": 19,
  "metrics": {
    "cv_average_precision": 0.6607,
    "brier_score": 0.1354,
    "held_out_pr_auc": 0.6684,
    "held_out_roc_auc": 0.8428,
    "held_out_f1": 0.6231,
    "held_out_recall": 0.7139
  }
}
```

### 7.4 Single Customer Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```
```json
{
  "predicted_churn": 1,
  "churn_probability": 0.5842,
  "decision_threshold": 0.3400,
  "risk_tier": "High"
}
```

### 7.5 Bounded Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
      },
      {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 34,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.5
      }
    ]
  }'
```
```json
{
  "predictions": [
    {
      "predicted_churn": 1,
      "churn_probability": 0.5842,
      "decision_threshold": 0.3400,
      "risk_tier": "High"
    },
    {
      "predicted_churn": 0,
      "churn_probability": 0.0411,
      "decision_threshold": 0.3400,
      "risk_tier": "Low"
    }
  ],
  "batch_size": 2,
  "processing_time_ms": 20.85
}
```

---

## 8. Empirical Latency Benchmarks

Measured locally using `scripts/benchmark_latency.py`:
```powershell
python scripts/benchmark_latency.py
```

| Request Type | Sample Count | Median (p50) | 95th Percentile (p95) | Mean Latency | Per-Record Time |
|---|---|---|---|---|---|
| **Single Record (`POST /predict`)** | 50 requests | **208.24 ms** | **258.77 ms** | 213.26 ms | 208.24 ms |
| **Batch ($N = 10$)** | 20 batches | **207.60 ms** | **253.22 ms** | 211.58 ms | **20.76 ms** |
| **Batch ($N = 50$)** | 10 batches | **213.30 ms** | **250.43 ms** | 217.40 ms | **4.27 ms** |

*Note: The 5-fold ensemble consisting of 600 decision trees is amortized across batches, dropping per-record inference time to 4.27 ms.*

---

## 9. Containerization & Docker Operations

### 9.1 Build Production Docker Image
```powershell
docker build -t predictiq-api:latest .
```

### 9.2 Run Container Independently
```powershell
docker run -d --name predictiq-api -p 8000:8000 predictiq-api:latest
```

Verify security & healthcheck:
```powershell
# Confirm running as non-root user (appuser, UID 10001)
docker exec predictiq-api id

# Inspect healthcheck probe status
docker inspect --format='{{json .State.Health}}' predictiq-api
```

### 9.3 Multi-Service Orchestration with Docker Compose
```powershell
# Validate compose syntax
docker compose config

# Spin up API service
docker compose up -d api
```

---

## 10. Implementation Status & Roadmap

### Phase 1: Foundation & Architecture — COMPLETE
- [x] Standard `src/` modular layout and directory hierarchy.
- [x] Typed configuration system with validation.
- [x] Pytest foundation suite.

### Phase 2: Ingestion & EDA — COMPLETE
- [x] Automated IBM Telco dataset acquisition with SHA-256 verification.
- [x] Schema & boundary constraint validation.
- [x] 25-cell exploratory analysis notebook with statistical hypothesis testing.

### Phase 3: Modeling & Tracking — COMPLETE
- [x] Stratified train/test splitting (`src/predictiq/data/loader.py`).
- [x] Domain feature engineering (`DomainFeatureEngineer`).
- [x] Baseline modeling with MLflow experiment tracking.

### Phase 4: Protocol Correction, Calibration & Explainability — COMPLETE
- [x] **Evaluation Protocol Correction:** Training-only 5-fold CV for model selection; isolated evaluation split (Seed 2026, $N=1,409$).
- [x] **Hyperparameter Optimization:** LightGBM and Random Forest tuning via `RandomizedSearchCV`.
- [x] **Probability Calibration:** Platt sigmoid scaling reducing Brier score loss to 0.1354.
- [x] **Decision Threshold Optimization:** Selected $p^* = 0.3400$ on training OOF, raising recall to 71.39%.
- [x] **Explainability:** TreeSHAP global beeswarm and local archetype waterfalls.

### Phase 5: Production API & Deployment — COMPLETE
- [x] **FastAPI Inference Service:** Lifespan model loading with zero retraining on startup.
- [x] **5 Core Endpoints:** `/health`, `/ready`, `/model-info`, `/predict`, `/predict/batch`.
- [x] **Strict Input Validation & Bounded Batches:** Pydantic v2 schemas; batch limit $\le 500$.
- [x] **Security & Privacy:** PII-safe logging middleware, zero secret leakage, production doc hiding.
- [x] **Non-Root Container:** `python:3.13-slim`, UID 10001, built-in healthchecks.
- [x] **Empirical Latency:** Benchmarked at 208.24 ms median, 258.77 ms p95.
- [x] **113 Passing Tests:** 16 new API unit and integration tests added.

### Phase 6: Drift Monitoring & Observability — NEXT
- [ ] Asynchronous prediction and input feature logging.
- [ ] Statistical data drift and concept drift detection (Evidently AI / PSI).
- [ ] Prometheus metrics endpoint (`/metrics`) and Grafana monitoring dashboard.
