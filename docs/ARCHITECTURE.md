# PredictIQ — System Architecture & Design Specification

## 1. System Vision & Purpose
PredictIQ is an end-to-end predictive intelligence platform designed to showcase dual competencies in **Data Science** (statistical exploratory analysis, feature engineering hypotheses, reproducible modeling, explainability) and **Machine Learning Engineering** (modular package design, containerized tracking, deterministic configuration, automated testing, and CI/CD readiness).

The system automates the lifecycle from ingestion of multi-dimensional raw business/tabular data to low-latency model inference and continuous performance/drift monitoring.

---

## 2. Layered Component Boundaries

```mermaid
graph TD
    subgraph Data Layer ["Data Layer (`data/`)"]
        Raw[Raw Ingestion]
        Interim[Interim Transformations]
        Processed[Processed Features]
        External[External Benchmarks]
    end

    subgraph Core Package ["Core Application (`src/predictiq/`)"]
        Config[Config & Validation]
        Features[Feature Transformers]
        Models[Model Training & Evaluators]
        Serving[Prediction API / Serving]
        Monitoring[Drift & Health Monitors]
    end

    subgraph Infrastructure ["Infrastructure Services (Docker Compose)"]
        Postgres[(PostgreSQL 16\nMetadata Store)]
        MLflow[MLflow 2.19\nTracking & Registry]
        ArtifactStore[(Persistent Volume\nArtifact Store)]
    end

    subgraph Exploration ["Analysis Layer (`notebooks/`)"]
        EDA[Interactive EDA & Hypotheses]
    end

    subgraph Quality Assurance ["Test & Foundation Suite (`tests/`)"]
        Pytest[Pytest Test Suite]
    end

    Raw --> Interim --> Processed
    Processed --> Features --> Models
    Models --> MLflow
    MLflow --> Postgres
    MLflow --> ArtifactStore
    Models --> Serving
    Serving --> Monitoring
    Config -.-> Core Package
    Pytest -.-> Core Package
```

### Component Responsibilities

| Layer / Component | Location | Responsibility | Phase 1 Status |
|---|---|---|---|
| **Data Layer** | `data/{raw,interim,processed,external}` | Immutable storage hierarchy with gitignore segregation for raw downloads, transformed tables, model-ready matrices, and external benchmarks. | **Implemented** (directory tree & gitkeep anchors) |
| **Analysis Layer** | `notebooks/` | Interactive prototyping, exploratory analysis, distribution tests, hypothesis verification. | **Implemented** (reproducibility contract & guidelines) |
| **Configuration** | `src/predictiq/config.py` | Typed dataclass models, port/mode validation, standard library `.env` parsing, dynamic path resolution, connection URL generation. | **Implemented & Verified** (42 passing unit tests) |
| **Feature Layer** | `src/predictiq/features/` | Scalable scikit-learn compatible transformers, feature stores, categorical encoding, imputers. | *Planned for Phase 3* |
| **ML Layer** | `src/predictiq/models/` | Model estimators, cross-validation runners, hyperparameter optimization, MLflow metric logging. | *Planned for Phase 4* |
| **Serving Layer** | `src/predictiq/serving/` | FastAPI asynchronous inference endpoints, input schema validation, model artifact loading. | *Planned for Phase 5* |
| **Monitoring Layer**| `src/predictiq/monitoring/`| Prediction data logging, PSI/Wasserstein drift calculation, accuracy tracking. | *Planned for Phase 6* |
| **Infrastructure** | `docker-compose.yml` | Containerized PostgreSQL 16 metadata store and MLflow 2.19 tracking server bound to localhost. | **Implemented & Verified** (validated via `docker compose config`) |
| **Test Suite** | `tests/` | Deterministic unit, integration, and configuration regression tests executed via `pytest`. | **Implemented & Verified** (42 passing tests) |

---

## 3. End-to-End Data-to-Model-to-Serving Workflow

```
[External / Raw Data]
         │
         ▼
[1. Ingestion & Validation] ────────► data/raw/ (checksum verification)
         │
         ▼
[2. Cleaning & Typing] ─────────────► data/interim/ (null handling, anomaly filtering)
         │
         ▼
[3. Feature Engineering] ───────────► data/processed/ (scaling, encodings, interaction terms)
         │
         ▼
[4. Training & Experimentation] ───► MLflow Tracking (parameters, metrics, loss curves)
         │
         ▼
[5. Model Registry & Evaluation] ───► MLflow Models (versioned artifacts, SHAP plots)
         │
         ▼
[6. Real-time Prediction Service] ──► FastAPI (input validation, inference, latency logs)
         │
         ▼
[7. Drift & Monitoring] ────────────► Continuous comparison against baseline distributions
```

1. **Ingestion:** Raw datasets are downloaded or received, verified with cryptographic hash checksums, and stored immutably in `data/raw/`.
2. **Cleaning & Profiling:** Data cleaning pipelines handle missing values, out-of-range dates, and invalid types, writing intermediate tables to `data/interim/`.
3. **Feature Pipelines:** Reusable scikit-learn transformers compute numerical transformations and categorical encodings, producing train/validation/test splits in `data/processed/`.
4. **Model Training:** Training loops execute multiple model candidates (e.g. baseline logistic/ridge, gradient boosted trees, neural baselines). Metrics (AUC, F1, LogLoss, RMSE, MAE) and parameters are logged to the MLflow tracking server.
5. **Model Registry:** The best-performing model candidate meeting validation acceptance criteria is registered into the MLflow model registry and serialized to persistent storage.
6. **Inference Serving:** A lightweight FastAPI service loads the registered model artifact, validates incoming payloads using typed schemas, and returns predictions with sub-second latency.
7. **Production Monitoring:** Real-time predictions and inputs are sampled to detect covariate shift (Population Stability Index / Kolmogorov-Smirnov tests) and trigger retraining alerts.

---

## 4. Configuration & Secrets Policy

PredictIQ strictly adheres to **The Twelve-Factor App** methodology for configuration:

- **Strict Environment Segregation:** All operational settings (database hosts, ports, credentials, tracking endpoints, logging levels) are decoupled from codebase logic and driven by environment variables.
- **Zero Secrets in Version Control:**
  - `.env` files, certificates, private keys, and passwords are permanently ignored in `.gitignore`.
  - `.env.example` provides committed template placeholders with safe local development defaults only.
  - The configuration subsystem (`src/predictiq/config.py`) parses environment variables with strict type validation and explicit range checks.
- **Fail-Fast Validation:** Applications or services start only if configuration passes validation rules (e.g., valid integer port numbers between 1 and 65535, valid operational environments like `development`, `testing`, `staging`, `production`).

---

## 5. Infrastructure Architecture

The local development infrastructure is containerized using Docker Compose to ensure zero drift between contributor workstations:

1. **PostgreSQL 16 (`postgres:16.8-alpine`):**
   - Serves as the relational backend tracking store for MLflow.
   - Utilizes named persistent Docker volume `predictiq_postgres_data` so metadata survives container restarts.
   - Configured with a health check running `pg_isready` at 10-second intervals.
   - Bound explicitly to `127.0.0.1:5432` to eliminate external network exposure.
2. **MLflow Tracking Server (`ghcr.io/mlflow/mlflow:v2.19.0`):**
   - Connects to PostgreSQL via standard SQLAlchemy connection URI.
   - Stores serialized model artifacts in named persistent Docker volume `predictiq_mlflow_artifacts`.
   - Starts only after PostgreSQL passes its health check (`condition: service_healthy`).
   - Exposes web UI and REST tracking API on `127.0.0.1:5000`.

---

## 6. Testing Strategy

Quality assurance is built into the foundation from day one:

- **Deterministic Unit Testing:** Fast, pure Python unit tests verifying configuration parsing, error boundaries, path resolution, and string construction without external network or service requirements.
- **Environment Isolation:** Tests leverage pytest fixtures and `monkeypatch` to isolate environment variables without polluting the runtime host process.
- **Infrastructure Validation:** Declarative Docker Compose validation ensures syntax, volume bindings, network isolation, and health check definitions are continuously verified.
- **Pre-Commit Readiness:** Test runs take under 0.1 seconds, enabling friction-free pre-commit and CI execution.

---

## 7. Scope Boundaries: Phase 1 vs. Future Phases

| Capability | Phase 1 (Current Scope) | Future Phases |
|---|---|---|
| Project Structure & Layout | Established (`src/`, `tests/`, `data/`, `notebooks/`, `docs/`) | Maintained |
| Dependency Management | Minimal standard library + pytest in `pyproject.toml` | ML & serving dependencies added per phase |
| Configuration System | Typed dataclasses, validation, `.env` loader | Extended for serving and drift thresholds |
| Infrastructure Definition | Docker Compose for PostgreSQL 16 + MLflow 2.19 | Added FastAPI service container |
| Unit Tests | Comprehensive foundation & config tests (42 passing) | Dataset, feature, and model test suites |
| Documentation Suite | Full Phase 1 architecture, ADR-001, setup, status | Feature catalogs and model cards |
| Dataset Acquisition & EDA | Deferred to Phase 2 | Phase 2 implementation |
| Feature Engineering Pipeline| Deferred to Phase 3 | Phase 3 implementation |
| Model Training & MLflow Logging | Deferred to Phase 4 | Phase 4 implementation |
| Prediction API Service | Deferred to Phase 5 | Phase 5 implementation |
| Drift & Monitoring System | Deferred to Phase 6 | Phase 6 implementation |
