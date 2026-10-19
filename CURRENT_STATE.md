# PredictIQ — Current State Assessment

**Last Updated:** 2026-10-18  
**Current Phase:** Phase 5: Production API & Dashboard MVP  
**Status:** COMPLETE & VERIFIED  

---

## 1. Objectives & Verification Status

| Objective | Requirement | Status | Verification Evidence |
|---|---|---|---|
| **Model-Serving Integration** | Load persisted champion model and metadata via `ChurnPredictor`; reuse preprocessing, calibration, and frozen decision threshold ($p^* = 0.3400$); zero retraining on startup; handle missing artifacts with clean 503 error; avoid duplicating model logic in routes. | **Completed** | Model loaded at startup in lifespan handler; no retraining; missing artifact returns 503; routes delegate completely to `ChurnPredictor`. |
| **FastAPI Endpoints** | Implement `GET /health`, `GET /ready`, `GET /model-info`, `POST /predict`, `POST /predict/batch`. Pydantic v2 schemas; return label, calibrated probability, threshold, risk tier; zero PII / secrets leaked. | **Completed** | All 5 endpoints implemented with Pydantic v2 schemas; verified via 16 tests in `tests/test_api.py`. |
| **API Engineering** | FastAPI with Uvicorn ASGI; structured PII-safe logging; bounded batch sizes ($\le 500$); non-blocking CPU offloading via `run_in_threadpool`; disable OpenAPI docs in production mode. | **Completed** | Threadpool offloading, logging middleware, batch validation, docs hidden when `ENVIRONMENT=production`. |
| **Streamlit Dashboard** | Implement 3-page dashboard: Customer Analytics (filters, Altair charts, real dataset stats), Model Intelligence (genuine evaluation artifacts, metrics glossary, SHAP waterfalls), Churn Prediction (single scoring form & batch CSV processor). | **Completed** | Modular architecture in `dashboard/`, responsive UI, service status badge, 30 dedicated unit tests. |
| **Tests** | Comprehensive tests covering API health, readiness, loading errors, predictions, unknown categoricals, batch limits, plus dashboard API client, data service, CSV validator, prediction card, dataset resolution, and model intelligence audit. | **Completed** | **168 passed of 168** tests in pytest in ~30s; 100% pass rate. |
| **Containerization** | Production Dockerfiles for API (`python:3.13-slim`) and Dashboard, unprivileged user (`appuser`, UID 10001), health check probes, Docker Compose integration. | **Completed** | Verified container execution, non-root user verification (`id`), healthy curl probes on both containers. |
| **Performance & Latency** | Measure latency locally with reproducible script; report actual p50 and p95; optimize pipeline overhead. | **Completed** | `scripts/benchmark_latency.py` measured single-record p50 = 208.24 ms, p95 = 258.77 ms; batching drops per-record time to 4.27 ms ($N=50$). |
| **Documentation & Protocol** | Update README, architecture, changelog, implementation reports, preserve evaluation limitation (seed 2026 split is not independently validated). | **Completed** | Complete documentation suite updated; protocol limitation explicitly preserved. |

---

## 2. Actual Repository Structure

```
D:\Projects\PredictIQ\
├── .dockerignore                     # Production build exclusions (.venv, data, tests, mlruns)
├── .env.example                     # Verified: safe defaults, uncommitted template
├── .gitignore                        # Verified: excludes .env, venv, data/*, artifacts, mlruns, baseline models
├── CURRENT_STATE.md                  # This file
├── Dockerfile                        # Multi-layer FastAPI production container (non-root, healthchecked)
├── docker-compose.yml                # Verified: API + Dashboard + PostgreSQL 16 + MLflow 2.19
├── dashboard\                        # Verified: Streamlit interactive presentation layer
│   ├── Dockerfile                    # Container definition for dashboard service
│   ├── requirements.txt              # Dashboard requirements (streamlit, pandas, altair, requests)
│   ├── app.py                        # Main entrypoint with sidebar navigation & branding
│   ├── components\
│   │   ├── kpi_card.py               # Glassmorphic KPI metric cards
│   │   ├── status_badge.py           # Real-time API & model readiness status pill
│   │   └── prediction_card.py        # Visual probability gauge & risk tier badge
│   ├── pages\
│   │   ├── customer_analytics.py     # Demographics, contract breakdowns, interactive filters
│   │   ├── model_intelligence.py     # Evaluation metrics, calibration curves, SHAP waterfalls
│   │   └── churn_prediction.py       # Single customer scoring form & batch CSV processor
│   ├── services\
│   │   ├── api_client.py             # Resilient REST client with readiness differentiation
│   │   ├── data_service.py           # Dataset loading, caching, aggregation & filtering
│   │   └── artifact_service.py       # Genuine evaluation report and figure loading
│   ├── utils\
│   │   ├── formatting.py             # Currency, percentage, and risk tier styling
│   │   └── csv_validator.py          # Batch CSV schema enforcement & template generator
│   └── tests\                        # 30 focused unit tests for dashboard services
├── data\
│   ├── metadata\
│   │   └── Telco-Customer-Churn_metadata.json # Verified: SHA-256 provenance metadata
│   ├── external\.gitkeep             # Verified: directory anchor
│   ├── interim\
│   │   ├── .gitkeep                  # Verified: directory anchor
│   │   └── Telco-Customer-Churn-cleaned.csv # Verified: type-normalized interim data ($N=7,043$)
│   ├── processed\.gitkeep            # Verified: directory anchor
│   └── raw\
│       ├── .gitkeep                  # Verified: directory anchor
│       └── Telco-Customer-Churn.csv  # Verified: immutable raw CSV (ignored by Git)
├── docs\
│   ├── adr\
│   │   └── ADR-001-initial-architecture.md  # Verified: Accepted ADR
│   ├── data\
│   │   └── DATASET_CARD.md           # Verified: Dataset card (provenance, schema, ethics)
│   ├── ARCHITECTURE.md               # Verified: System design, presentation layer, API & data boundaries
│   ├── CHANGELOG.md                  # Verified: release notes (0.5.0 added)
│   ├── DEVELOPMENT_SETUP.md          # Verified: PowerShell setup & verification
│   ├── PHASE_3_IMPLEMENTATION_REPORT.md # Verified: Phase 3 report
│   ├── PHASE_4_IMPLEMENTATION_REPORT.md # Verified: Phase 4 comprehensive report
│   └── PHASE_5_IMPLEMENTATION_REPORT.md # Verified: Phase 5 production API & deployment report
├── models\
│   ├── churn_champion_model.joblib   # Verified: Champion pipeline binary
│   └── churn_champion_model_metadata.json # Verified: Model metadata, metrics, and threshold
├── NEXT_TASK.md                      # Verified: Phase 6 scope & acceptance criteria
├── notebooks\
│   ├── README.md                     # Verified: Reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # Verified: Executed 25-cell EDA & hypothesis tests
├── pyproject.toml                    # Verified: Dependencies with fastapi, uvicorn, pydantic, streamlit, etc.
├── README.md                         # Verified: Complete project guide, dashboard docs, & curl examples
├── reports\
│   ├── figures\
│   │   ├── calibration_curve.png     # Verified: Platt scaling reliability diagram
│   │   ├── champion_heldout_curves.png # Verified: Held-out test PR and ROC curves
│   │   ├── confusion_matrices.png    # Verified: Side-by-side CM heatmaps
│   │   ├── pr_curves.png             # Verified: Comparative PR curves
│   │   ├── roc_curves.png            # Verified: Comparative ROC curves
│   │   ├── shap_local_high_risk.png  # Verified: High-risk archetype waterfall
│   │   ├── shap_local_low_risk.png   # Verified: Low-risk archetype waterfall
│   │   ├── shap_summary.png          # Verified: Global TreeSHAP beeswarm summary
│   │   └── threshold_tradeoff.png    # Verified: Precision/Recall/F1 threshold curves
│   ├── phase4_cv_model_comparison.csv# Verified: Machine-readable CV comparison table
│   └── phase4_final_test_summary.csv # Verified: Final isolated test summary
├── scripts\
│   ├── benchmark_latency.py          # Verified: Reproducible latency measurement script
│   ├── generate_champion_curves.py   # Verified: Reproducible held-out curves generator
│   ├── train_baseline.py             # Verified: Baseline model training CLI
│   └── tune_and_train.py             # Verified: Phase 4 tuning, calibration, and training CLI
├── src\
│   └── predictiq\
│       ├── __init__.py               # Verified: exports Config, version 0.1.0
│       ├── config.py                 # Verified: typed dataclasses, validation, .env, PREDICTIQ_ROOT_DIR
│       ├── api\                      # Verified: FastAPI application, routes, deps, schemas
│       ├── data\                     # Verified: ingestion, validation, cleaning, loader
│       ├── features\                 # Verified: feature transformers & pipelines
│       ├── inference\                # Verified: ChurnPredictor engine
│       ├── models\                   # Verified: baselines, calibration, evaluation, tuning, thresholds, SHAP
│       └── tracking\                 # Verified: MLflow tracking manager
└── tests\                            # Verified: 138 core unit, integration, and model audit tests
```

---

## 3. Environment & Verification

| Component | Status | Details |
|---|---|---|
| **Operating System** | Verified | Windows 11 (PowerShell runtime) |
| **Python** | Verified | `Python 3.13.14` |
| **Virtual Environment** | Verified | `D:\Projects\PredictIQ\.venv` with `fastapi 0.110.0`, `uvicorn 0.28.0`, `pydantic 2.0.0`, `streamlit 1.66.0`, `altair 6.3.0`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `shap 0.54.1` |
| **Test Suite** | Verified | **168 passed of 168** (138 core/audit + 30 dashboard unit tests) with zero failures |
| **Linter** | Verified | Ruff: `All checks passed!` |
| **Container Engine** | Verified | Docker Desktop 29.6.2 running; images `predictiq-api:latest` and `predictiq-dashboard:latest` built & validated |
| **Container Security** | Verified | Non-root execution (`appuser`, UID 10001) across both containers |

---

## 4. Latency Benchmark Metrics

Measured locally via `scripts/benchmark_latency.py`:
- **Single Record (`POST /predict`):** Median (p50) = **208.24 ms**, 95th Percentile (p95) = **258.77 ms**, Mean = 213.26 ms.
- **Batch ($N=10$):** Median = **207.60 ms** (20.76 ms / record).
- **Batch ($N=50$):** Median = **213.30 ms** (4.27 ms / record).

---

## 5. Known Limitations & Handoff

1. **Inference Latency Profile:**
   - Single-record prediction median latency is 208.24 ms due to the 5-fold ensemble of 120 decision trees (600 trees total) evaluated across pandas DataFrames. Sub-50ms performance requires tree distillation or batch amortization.
2. **Evaluation Protocol Limitation (Preserved from Phase 4):**
   - The Seed 2026 split does not establish a genuinely independent test set after earlier development used the same underlying IBM Telco dataset. Reported metrics must not be represented as an unbiased, independently validated performance estimate.
3. **Next Recommended Milestone:**
   - Phase 6: Continuous Data Drift Monitoring, Prediction Logging & Observability.
