# PredictIQ — Predictive Intelligence Platform

PredictIQ is a modular, production-grade predictive intelligence platform built to demonstrate advanced competencies across the complete machine learning lifecycle—from exploratory data analysis and feature engineering to experiment tracking, model calibration, explainability, low-latency API serving, containerized deployment, and an interactive Streamlit dashboard.

---

## 1. Project Purpose & Target Users

- **Data Scientists:** Reproducible exploratory data analysis, hypothesis-driven feature transformations, statistical validation, Platt scaling calibration, and TreeSHAP explainability.
- **Machine Learning Engineers:** Production-ready code structure, 12-factor configuration, containerized microservices, FastAPI serving, asynchronous batch processing, automated testing, and CI/CD readiness.
- **Business Stakeholders & Operators:** Polished, interactive Streamlit dashboard for customer analytics, model auditing, and real-time churn prediction.
- **Hiring Managers & Technical Evaluators:** A transparent, end-to-end reference implementation demonstrating rigorous engineering practices, architecture documentation, latency benchmarking, and clean system design.

---

## 2. Core Objectives

1. **Reproducibility:** Deterministic environments, versioned dependencies, pinned Docker images, and isolated local development workflows.
2. **Modular Architecture:** Clear boundary separation between data layers, analysis notebooks, core application packages, serving APIs, presentation dashboards, and infrastructure.
3. **Operational Rigor:** Fast automated testing (143 passing tests), typed configuration with input validation, zero committed secrets, non-root container security, and health-monitored infrastructure.
4. **Phased Delivery:** Disciplined incremental execution where each phase is verified against explicit acceptance criteria before proceeding.

---

## 3. Technology Stack

| Domain | Technology | Version / Specification | Rationale |
|---|---|---|---|
| **Language** | Python | `>=3.10` (Tested on `3.13.14`) | Modern type hints, dataclasses, wide ML library ecosystem |
| **API Framework** | FastAPI + Uvicorn | `fastapi>=0.110.0`, `uvicorn>=0.28.0` | Asynchronous ASGI framework, OpenAPI docs, threadpool offloading |
| **Dashboard** | Streamlit + Altair | `streamlit>=1.32.0`, `altair>=5.0.0` | Interactive portfolio-grade UI, responsive charts, KPI cards |
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
├── .dockerignore                     # Production build exclusions (.venv, data, tests, mlruns)
├── .env.example                     # Environment configuration template
├── .gitignore                        # Comprehensive ignore rules (data, venv, secrets, logs)
├── CURRENT_STATE.md                  # Verified milestone status & test logs
├── Dockerfile                        # FastAPI production container (non-root, healthchecked)
├── docker-compose.yml                # API + Dashboard + PostgreSQL 16 & MLflow 2.19 orchestration
├── dashboard/                        # Streamlit Interactive Dashboard
│   ├── Dockerfile                    # Container definition for dashboard service
│   ├── requirements.txt              # Dashboard-specific dependencies
│   ├── app.py                        # Main entrypoint with sidebar navigation & branding
│   ├── components/                   # Reusable UI presentation components
│   │   ├── kpi_card.py               # Glassmorphic KPI metric cards
│   │   ├── status_badge.py           # Real-time API & model readiness status pill
│   │   └── prediction_card.py        # Visual probability gauge & risk tier badge
│   ├── pages/                        # Specialized analytical views
│   │   ├── customer_analytics.py     # Demographics, contract breakdowns, interactive filters
│   │   ├── model_intelligence.py     # Evaluation metrics, calibration curves, SHAP waterfalls
│   │   └── churn_prediction.py       # Single customer scoring form & batch CSV processor
│   ├── services/                     # Business logic and external communication
│   │   ├── api_client.py             # Resilient REST client with readiness differentiation
│   │   ├── data_service.py           # Dataset loading, caching, aggregation & filtering
│   │   └── artifact_service.py       # Genuine evaluation report and figure loading
│   ├── utils/                        # Formatting and schema validation utilities
│   │   ├── formatting.py             # Currency, percentage, and risk tier styling
│   │   └── csv_validator.py          # Batch CSV schema enforcement & template generator
│   └── tests/                        # 30 focused unit tests for dashboard services & components
│       ├── test_api_client.py        # 13 tests: 200, 422, 503, timeouts, batch bounds
│       ├── test_artifact_service.py  # 5 tests: metadata, CV summary, figures resolution
│       ├── test_csv_validator.py     # 6 tests: validation, limits, missing cols, template
│       └── test_data_service.py      # 6 tests: metrics, breakdowns, filters, fallbacks
├── data/                             # Data layer with immutable isolation
│   ├── metadata/                     # Provenance metadata (Telco-Customer-Churn_metadata.json)
│   ├── raw/                          # Original immutable datasets (Telco-Customer-Churn.csv)
│   ├── interim/                      # Cleaned intermediate data (Telco-Customer-Churn-cleaned.csv)
│   ├── processed/                    # Model-ready features & splits (.gitkeep)
│   └── external/                     # Third-party benchmark data (.gitkeep)
├── docs/                             # Architecture & operational guides
│   ├── ARCHITECTURE.md               # End-to-end system design & layer boundaries
│   ├── CHANGELOG.md                  # Versioned change history & validation notes
│   ├── DEVELOPMENT_SETUP.md          # Windows PowerShell setup instructions
│   ├── PHASE_3_IMPLEMENTATION_REPORT.md # Baseline modeling & MLflow report
│   ├── PHASE_4_IMPLEMENTATION_REPORT.md # Protocol correction, tuning, calibration, SHAP
│   └── PHASE_5_IMPLEMENTATION_REPORT.md # Production API, latency, containerization
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
│       ├── config.py                 # Typed configuration, validation, & .env loader
│       ├── api/                      # Production REST API layer (FastAPI)
│       ├── data/                     # Data acquisition, validation, & cleaning
│       ├── features/                 # Domain feature engineering & preprocessing pipeline
│       ├── inference/                # ChurnPredictor engine with risk tiering
│       ├── models/                   # Training, tuning, calibration, thresholds, SHAP
│       └── tracking/                 # MLflow experiment tracking manager
└── tests/                            # Core automated test suite (113 passed)
```

---

## 5. Prerequisites

Before running the platform, verify that your machine has:
1. **Python 3.10+** (verified on `3.13.14`)
2. **Git for Windows** (verified on `2.55.0.windows.3`)
3. **Docker Desktop** (optional for pure Python tests, recommended for containerized services)

---

## 6. Quickstart & Local Execution

### Step 1: Clone Repository & Create Virtual Environment
```powershell
Set-Location -Path "D:\Projects\PredictIQ"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
```

### Step 2: Install Package with All Dependencies
```powershell
& ".\.venv\Scripts\pip.exe" install -e ".[dev,api,dashboard,analysis]"
```

### Step 3: Run Automated Test Suite & Linter
```powershell
# Run all 143 unit and integration tests (tests/ + dashboard/tests/)
& ".\.venv\Scripts\pytest.exe" -v

# Run Ruff linter
& ".\.venv\Scripts\ruff.exe" check .
```
*Expected: 143 passed in ~24s with zero warnings or errors.*

### Step 4: Launch Production FastAPI Service Locally
```powershell
& ".\.venv\Scripts\python.exe" -m predictiq.api.main
```
The API initializes at `http://127.0.0.1:8000`.

### Step 5: Launch Streamlit Dashboard Locally
In a separate terminal window:
```powershell
& ".\.venv\Scripts\streamlit.exe" run dashboard/app.py
```
The dashboard opens in your browser at `http://localhost:8501`.

---

## 7. Interactive Streamlit Dashboard

The Streamlit dashboard serves as the interactive presentation layer for PredictIQ, cleanly separated from model training and backend inference.

### Architecture & Responsibility Separation
- **Dashboard (`dashboard/`):** Presentation, navigation, filters, responsive Altair charts, CSV uploads, and downloads.
- **FastAPI (`src/predictiq/api/`):** Validation, preprocessing, calibrated inference, and risk classification.
- **Existing Artifacts (`reports/`, `models/`):** Real historical evaluation metrics, diagnostic curves, and TreeSHAP waterfall plots.

### Environment Configuration
| Variable | Default Value | Description |
|---|---|---|
| `PREDICTIQ_API_URL` | `http://localhost:8000` | Target URL of the FastAPI prediction backend (`http://api:8000` in Docker) |
| `PREDICTIQ_ROOT_DIR` | Auto-detected | Project root directory for dataset and artifact resolution |
| `DASHBOARD_PORT` | `8501` | Host port mapped to the Streamlit container |

### Dashboard Pages

#### 1. Customer Analytics (`dashboard/pages/customer_analytics.py`)
- **Real Cohort Statistics:** Computes customer metrics directly from `data/interim/Telco-Customer-Churn-cleaned.csv` ($N=7,043$).
- **KPI Summary Cards:** Total customer volume, churn count and rate (26.5%), retained count and rate (73.5%), and cumulative monthly revenue.
- **Interactive Altair Charts:**
  - Churn Rate by Contract Type (Month-to-month vs. 1-year vs. 2-year).
  - Churn Patterns Across Tenure Cohorts (0–12 mos through 49–72 mos).
  - Churn by Internet Service Provider (Fiber optic ~42% churn vs. DSL ~19%).
  - Churn Distribution by Payment Method (Electronic check vs. automated billing).
- **Interactive Filters:** Contract duration, internet service, payment method, senior citizen status, partner, dependents, tenure slider, and monthly charges range with an instant "Clear All Filters" button.
- **Privacy Protection:** Displays customer demographic features without exposing raw internal database keys.

#### 2. Model Intelligence (`dashboard/pages/model_intelligence.py`)
- **Model Architecture Audit:** Details on the champion *Random Forest (Calibrated)* pipeline (120 trees per fold, 600 trees total) and Platt sigmoid calibration.
- **Optimal Decision Threshold ($p^* = 0.3400$):** Explains how training out-of-fold F1-maximization boosted churn recall from 57.22% to 71.39% (+14.17 pp).
- **Evaluation Metrics:** PR-AUC (0.6684), ROC-AUC (0.8428), Recall (71.39%), Precision (55.28%), F1-Score (0.6231), and Brier Score (0.1354, +15.4% improvement).
- **Metric Definitions Guide:** Accessible explanations of Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
- **Artifact Visualizations:**
  - Candidate model comparison table (Random Forest vs. LightGBM vs. Logistic Regression).
  - Decision threshold trade-off curve (`reports/figures/threshold_tradeoff.png`).
  - Calibration reliability diagram (`reports/figures/calibration_curve.png`).
  - Comparative PR and ROC curves (`reports/figures/pr_curves.png`, `roc_curves.png`).
  - Global TreeSHAP feature beeswarm and local archetype waterfalls (`reports/figures/shap_summary.png`, `shap_local_high_risk.png`, `shap_local_low_risk.png`).
- **Protocol Transparency:** Prominent banner emphasizing that the Seed 2026 split represents an internal holdout benchmark, not an independently validated field estimate.

#### 3. Churn Prediction (`dashboard/pages/churn_prediction.py`)
- **Single Customer Scoring Form:**
  - Form matching all 19 customer features required by `CustomerInputSchema`.
  - Dispatches payload to `POST /predict`.
  - Displays calibrated probability, decision threshold cutoff ($0.3400$), classification margin, and business risk tier (`High Risk`, `Medium Risk`, `Low Risk`).
  - Tailored operational recommendations for customer retention.
- **CSV Batch Evaluation:**
  - File uploader supporting batches up to **500 records**.
  - Downloadable sample CSV template (`predictiq_batch_sample.csv`).
  - Input schema validation ensuring all required columns exist and data types are valid.
  - Row association preservation between input records and predictions.
  - Aggregated summary statistics and risk tier breakdown.
  - Downloadable CSV containing original input columns augmented with `Predicted_Churn`, `Churn_Probability`, `Decision_Threshold`, and `Risk_Tier`.

### Expected Batch CSV Input Schema
The uploaded CSV must include the following 19 columns:
```
gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges
```
*(Optional `customerID` column is preserved in the output table for row identification).*

---

## 8. Containerization & Docker Compose

Docker Compose coordinates the complete microservice architecture:
```
[ User Browser ]
       │
       ├─► http://localhost:8501 ──► [ predictiq-dashboard (Streamlit) ]
       │                                     │ (Internal HTTP)
       │                                     ▼
       └─► http://localhost:8000 ──► [ predictiq-api (FastAPI) ]
                                             │
                                     [ Local Joblib Model ]
```

### 1. Build and Run Full Stack
```powershell
# Start API, Dashboard, PostgreSQL, and MLflow
docker compose up -d

# Verify service health
docker compose ps
```

### 2. Run Only Serving Services (API + Dashboard)
```powershell
# Start API and Dashboard without database infrastructure
docker compose up -d api dashboard
```

### 3. Verify Container Security
Both the API and Dashboard containers run strictly as an unprivileged non-root user:
```powershell
docker exec predictiq-dashboard id
# Output: uid=10001(appuser) gid=10001(appgroup) groups=10001(appgroup)
```

---

## 9. Troubleshooting & Common Issues

| Issue | Symptom | Root Cause & Resolution |
|---|---|---|
| **API Offline** | Badge shows `API Offline • Connection Refused` | FastAPI service is not running. Launch with `python -m predictiq.api.main` or `docker compose up -d api`. |
| **Model Not Ready** | Badge shows `API Online • Model Not Loaded (503)` | Model artifact is missing from `models/`. Verify `models/churn_champion_model.joblib` exists or run `python scripts/tune_and_train.py`. |
| **Missing Artifacts** | Section displays `Artifact unavailable` notice | Specific report figures or summaries are missing. Generate them via `python scripts/tune_and_train.py`. The dashboard degrades gracefully without crashing. |
| **Batch Size Exceeded** | `Batch size exceeds maximum limit of 500 records` | Uploaded CSV has $> 500$ rows. Split large files into chunks $\le 500$ to ensure bounded memory usage. |
| **Missing Columns in CSV** | `Missing required column(s): ...` | CSV header mismatch. Click **Download Sample CSV** in the Batch tab to verify the exact column names. |

---

## 10. Current Implementation Status

### Complete & Verified
- [x] **Phase 1: Project Foundation & Architecture** (42 tests)
- [x] **Phase 2: Data Ingestion, Validation & EDA** (28 tests, 25-cell notebook)
- [x] **Phase 3: Baseline Modeling & MLflow Tracking** (16 tests)
- [x] **Phase 4: Protocol Correction, Tuning, Calibration & SHAP** (11 tests, champion model)
- [x] **Phase 5: Production FastAPI REST API & Deployment** (16 tests, containerized)
- [x] **Dashboard MVP: Streamlit Customer Analytics, Model Intelligence & Prediction** (30 tests, containerized)
- [x] **Total Automated Test Suite:** **168 passing tests** (100% pass rate in ~30s).
- [x] **Code Quality:** Ruff linter clean (`All checks passed!`).

### Next Milestone (Phase 6)
- [ ] Asynchronous prediction payload logging and database persistence.
- [ ] Statistical data drift and concept drift detection (PSI / Kolmogorov-Smirnov).
- [ ] Prometheus metrics endpoint (`/metrics`) and Grafana monitoring dashboard.
