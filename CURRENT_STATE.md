# PredictIQ — Current State Assessment

**Last Updated:** 2026-10-18  
**Current Phase:** Phase 5: Production API and Deployment  
**Status:** COMPLETE & VERIFIED  

---

## 1. Phase 5 Objectives & Verification

| Objective | Requirement | Status | Verification Evidence |
|---|---|---|---|
| **Model-serving integration** | Load persisted champion model and metadata via `ChurnPredictor`; reuse preprocessing, calibration, and frozen decision threshold ($p^* = 0.3400$); zero retraining on startup; handle missing artifacts with clean 503 error; avoid duplicating model logic in routes. | **Completed** | Model loaded at startup in lifespan handler; no retraining; missing artifact returns 503; routes delegate completely to `ChurnPredictor`. |
| **API endpoints** | Implement `GET /health`, `GET /ready`, `GET /model-info`, `POST /predict`, `POST /predict/batch`. Pydantic v2 schemas; return label, calibrated probability, threshold, risk tier; zero PII / secrets leaked. | **Completed** | All 5 endpoints implemented with Pydantic v2 schemas; verified via 16 tests in `tests/test_api.py`. |
| **API engineering** | FastAPI with Uvicorn ASGI; structured PII-safe logging; bounded batch sizes ($\le 500$); non-blocking CPU offloading via `run_in_threadpool`; disable OpenAPI docs in production mode. | **Completed** | Threadpool offloading, logging middleware, batch validation, docs hidden when `ENVIRONMENT=production`. |
| **Tests** | Comprehensive tests covering health, readiness, loading errors, valid/invalid predictions, unknown categoricals, batch limits, schema boundaries, threshold usage. | **Completed** | **113 passed of 113** tests in pytest in ~23s; 16 dedicated API tests. |
| **Containerization** | Production Dockerfile (`python:3.13-slim`), unprivileged user (`appuser`, UID 10001), health check probe, independent of PostgreSQL/MLflow at runtime, `.dockerignore`, `docker-compose.yml` integration. | **Completed** | Verified container execution, non-root user verification (`id`), healthy curl probe. |
| **Performance & Latency** | Measure latency locally with reproducible script; report actual p50 and p95; do not claim sub-50ms unless measured; optimize pipeline overhead. | **Completed** | `scripts/benchmark_latency.py` measured single-record p50 = 208.24 ms, p95 = 258.77 ms; batching drops per-record time to 4.27 ms ($N=50$). |
| **Documentation & Protocol** | Update README, architecture, changelog, implementation report, preserve evaluation limitation (seed 2026 split is not independently validated). | **Completed** | Complete documentation suite updated; protocol limitation explicitly preserved. |

---

## 2. Actual Repository Structure

```
D:\Projects\PredictIQ\
├── .dockerignore                     # Production build exclusions (.venv, data, tests, mlruns)
├── .env.example                     # Verified: safe defaults, uncommitted template
├── .gitignore                        # Verified: excludes .env, venv, data/*, artifacts, mlruns, models/*.joblib
├── CURRENT_STATE.md                  # This file
├── Dockerfile                        # Multi-layer production container (non-root, healthchecked)
├── docker-compose.yml                # Verified: API + PostgreSQL 16 + MLflow 2.19
├── data\
│   ├── metadata\
│   │   └── Telco-Customer-Churn_metadata.json # Verified: SHA-256 provenance metadata
│   ├── external\.gitkeep             # Verified: directory anchor
│   ├── interim\
│   │   ├── .gitkeep                  # Verified: directory anchor
│   │   └── Telco-Customer-Churn-cleaned.csv # Verified: type-normalized interim data
│   ├── processed\.gitkeep            # Verified: directory anchor
│   └── raw\
│       ├── .gitkeep                  # Verified: directory anchor
│       └── Telco-Customer-Churn.csv  # Verified: immutable raw CSV (ignored by Git)
├── docs\
│   ├── adr\
│   │   └── ADR-001-initial-architecture.md  # Verified: Accepted ADR
│   ├── data\
│   │   └── DATASET_CARD.md           # Verified: Dataset card (provenance, schema, ethics)
│   ├── ARCHITECTURE.md               # Verified: System design, API serving & data layer boundaries
│   ├── CHANGELOG.md                  # Verified: release notes (0.5.0 added)
│   ├── DEVELOPMENT_SETUP.md          # Verified: PowerShell setup & verification
│   ├── PHASE_3_IMPLEMENTATION_REPORT.md # Verified: Phase 3 report
│   ├── PHASE_4_IMPLEMENTATION_REPORT.md # Verified: Phase 4 comprehensive report
│   └── PHASE_5_IMPLEMENTATION_REPORT.md # Verified: Phase 5 production API & deployment report
├── models\
│   ├── churn_champion_model.joblib   # Verified: Champion pipeline binary (ignored by Git)
│   └── churn_champion_model_metadata.json # Verified: Model metadata, metrics, and threshold
├── NEXT_TASK.md                      # Verified: Phase 6 scope & acceptance criteria
├── notebooks\
│   ├── README.md                     # Verified: Reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # Verified: Executed 25-cell EDA & hypothesis tests
├── pyproject.toml                    # Verified: Dependencies with fastapi, uvicorn, pydantic, etc.
├── README.md                         # Verified: Complete project guide, API docs, & curl examples
├── reports\
│   ├── figures\
│   │   ├── calibration_curve.png     # Verified: Platt scaling reliability diagram
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
│   ├── train_baseline.py             # Verified: Baseline model training CLI
│   └── tune_and_train.py             # Verified: Phase 4 tuning, calibration, and training CLI
├── src\
│   └── predictiq\
│       ├── __init__.py               # Verified: exports Config, version 0.1.0
│       ├── config.py                 # Verified: typed dataclasses, validation, .env, PREDICTIQ_ROOT_DIR
│       ├── api\
│       │   ├── __init__.py           # Verified: exports app and create_app
│       │   ├── app.py                # Verified: FastAPI factory, lifespan loader, logging middleware
│       │   ├── deps.py               # Verified: Dependency injection & 503 readiness guard
│       │   ├── main.py               # Verified: Uvicorn ASGI production entrypoint
│       │   ├── routes.py             # Verified: /health, /ready, /model-info, /predict, /predict/batch
│       │   └── schemas.py            # Verified: Pydantic v2 schemas with bounded batch validation
│       ├── data\
│       ├── features\
│       ├── inference\
│       │   ├── __init__.py           # Verified: exports ChurnPredictor
│       │   └── predictor.py          # Verified: Optimized inference engine & risk tiers
│       ├── models\
│       │   ├── persistence.py        # Verified: Robust model & metadata path resolution
│       │   └── ...                   # Verified: baselines, calibration, evaluation, tuning, thresholds, SHAP
│       └── tracking\
└── tests\
    ├── test_api.py                   # Verified: 16 unit & integration tests for FastAPI service
    ├── test_calibration.py           # Verified: 2 unit tests for calibration & Brier score
    ├── test_config.py                # Verified: 42 pytest foundation unit tests
    ├── test_data_ingestion.py        # Verified: 9 unit tests for acquisition & hashing
    ├── test_data_loader.py           # Verified: 7 unit tests for splitting & leakage prevention
    ├── test_data_validation.py       # Verified: 12 unit tests for schema & domain rules
    ├── test_evaluation_protocol.py   # Verified: 2 unit tests for protocol isolation & CV folds
    ├── test_explainability.py        # Verified: 1 unit test for TreeSHAP attributions & plots
    ├── test_features.py              # Verified: 6 unit tests for feature engineering
    ├── test_inference.py             # Verified: 4 unit tests for persistence & prediction
    ├── test_model_evaluation.py      # Verified: 4 unit tests for evaluation & metrics
    ├── test_thresholds.py            # Verified: 3 unit tests for OOF threshold search & cost matrix
    ├── test_tracking.py              # Verified: 2 unit tests for MLflow local logging
    └── test_tuning.py                # Verified: 3 unit tests for hyperparameter search spaces
```

---

## 3. Environment & Verification

| Component | Status | Details |
|---|---|---|
| **Operating System** | Verified | Windows 11 (PowerShell runtime) |
| **Python** | Verified | `Python 3.13.14` |
| **Virtual Environment** | Verified | `D:\Projects\PredictIQ\.venv` with `fastapi 0.110.0`, `uvicorn 0.28.0`, `pydantic 2.0.0`, `httpx 0.28.1`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `shap 0.54.1` |
| **Test Suite** | Verified | **113 passed** in 23.38s with zero failures |
| **Linter** | Verified | Ruff: `All checks passed!` |
| **Container Engine** | Verified | Docker Desktop 29.6.2 running; image `predictiq-api:latest` built & validated |
| **Container Security** | Verified | Non-root execution (`appuser`, UID 10001) |

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
