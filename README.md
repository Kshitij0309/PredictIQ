# PredictIQ — Predictive Intelligence Platform

PredictIQ is a modular, production-grade predictive intelligence platform built to demonstrate advanced competencies across the complete machine learning lifecycle—from exploratory data analysis and feature engineering to experiment tracking, model registry, low-latency API serving, and drift monitoring.

---

## 1. Project Purpose & Target Users

- **Data Scientists:** Reproducible exploratory data analysis, hypothesis-driven feature transformations, statistical validation, and model explainability.
- **Machine Learning Engineers:** Production-ready code structure, 12-factor configuration, containerized infrastructure, automated testing, and CI/CD readiness.
- **Hiring Managers & Technical Evaluators:** A transparent, end-to-end reference implementation demonstrating rigorous engineering practices, architecture documentation, and clean system design.

---

## 2. Core Objectives

1. **Reproducibility:** Deterministic environments, versioned dependencies, pinned Docker images, and isolated local development workflows.
2. **Modular Architecture:** Clear boundary separation between data layers, analysis notebooks, core application packages, and infrastructure.
3. **Operational Rigor:** Fast automated testing, typed configuration with input validation, zero committed secrets, and health-monitored infrastructure.
4. **Phased Delivery:** Disciplined incremental execution where each phase is verified against explicit acceptance criteria before proceeding.

---

## 3. Technology Stack

| Domain | Technology | Version / Specification | Rationale |
|---|---|---|---|
| **Language** | Python | `>=3.10` (Tested on `3.13.14`) | Modern type hints, dataclasses, wide ML library ecosystem |
| **Packaging** | Setuptools / PEP 517 | Standard `pyproject.toml` | Standardized Python packaging with editable install support |
| **Testing** | Pytest | `>=8.0.0` (Installed `9.1.1`) | Deterministic unit testing and fixture-based environment isolation |
| **Configuration** | Python Standard Library | `dataclasses`, `os`, `pathlib` | Zero-dependency, ultra-fast initialization, strict type validation |
| **Containerization** | Docker Compose | Compose Specification v2/v5 | Local service orchestration with persistent volumes and healthchecks |
| **Database** | PostgreSQL | `16.8-alpine` | High-performance relational backend store for MLflow tracking |
| **Experiment Tracking** | MLflow | `ghcr.io/mlflow/mlflow:v2.19.0` | Experiment metrics, parameter logging, and artifact registry |

---

## 4. Repository Structure

```
PredictIQ/
├── .env.example                     # Environment configuration template
├── .gitignore                        # Comprehensive ignore rules (data, venv, secrets, logs)
├── artifacts/                        # Model artifacts & serializations (.gitkeep)
├── CURRENT_STATE.md                  # Verified milestone status & test logs
├── data/                             # Data layer with immutable isolation
│   ├── metadata/                     # Provenance metadata (Telco-Customer-Churn_metadata.json)
│   ├── raw/                          # Original immutable datasets (.gitkeep, Telco-Customer-Churn.csv)
│   ├── interim/                      # Cleaned intermediate transformations (.gitkeep, Telco-Customer-Churn-cleaned.csv)
│   ├── processed/                    # Model-ready features & splits (.gitkeep)
│   └── external/                     # Third-party benchmark data (.gitkeep)
├── docker-compose.yml                # PostgreSQL 16 & MLflow 2.19 orchestration
├── docs/                             # Architecture & operational guides
│   ├── adr/
│   │   └── ADR-001-initial-architecture.md  # Accepted architecture decision record
│   ├── data/
│   │   └── DATASET_CARD.md           # Dataset provenance, schema, audit, and ethics
│   ├── ARCHITECTURE.md               # End-to-end system design & layer boundaries
│   ├── CHANGELOG.md                  # Versioned change history & validation notes
│   └── DEVELOPMENT_SETUP.md          # Exact Windows PowerShell setup instructions
├── NEXT_TASK.md                      # Phase 3 scope, prerequisites, and handoff plan
├── notebooks/                        # Exploratory analysis & prototyping
│   ├── README.md                     # Notebook reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # End-to-end EDA & hypothesis testing notebook
├── pyproject.toml                    # Package build metadata & pytest configuration
├── README.md                         # Project overview and quickstart guide
├── src/                              # Application source tree (src/ layout)
│   └── predictiq/
│       ├── __init__.py               # Package exports & version metadata
│       ├── config.py                 # Typed configuration, validation, & .env loader
│       └── data/                     # Data acquisition, validation, & cleaning
│           ├── __init__.py           # Data module public exports
│           ├── ingestion.py          # Verified download & SHA-256 provenance tracking
│           ├── validation.py         # Non-destructive schema & domain constraint validation
│           └── cleaning.py          # Type normalization & interim data preparation
└── tests/                            # Automated test suite (63 passed)
    ├── test_config.py                # 42 unit tests for config, ports, paths, & env
    ├── test_data_ingestion.py        # 9 unit tests for checksums, provenance, & acquisition
    └── test_data_validation.py       # 12 unit tests for schema, constraints, & domain rules
```

---

## 5. Prerequisites

Before running the platform, verify that your machine has:
1. **Python 3.10+** (verified on `3.13.14`)
2. **Git for Windows** (verified on `2.55.0.windows.3`)
3. **Docker Desktop** (optional for pure unit tests, required for MLflow & PostgreSQL services)

---

## 6. Quickstart (Windows PowerShell)

### Step 1: Open PowerShell & Navigate to Directory
```powershell
Set-Location -Path "D:\Projects\PredictIQ"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
```

### Step 3: Install Package & Dependencies
```powershell
& ".\.venv\Scripts\pip.exe" install -e ".[dev,analysis]"
```

### Step 4: Configure Local Environment
```powershell
Copy-Item -Path ".env.example" -Destination ".env"
```

### Step 5: Acquire & Validate IBM Telco Dataset (Phase 2)
```powershell
# 1. Download and verify provenance (SHA-256 hash checking)
& ".\.venv\Scripts\python.exe" -m predictiq.data.ingestion

# 2. Run schema and domain validation audit
& ".\.venv\Scripts\python.exe" -c "from predictiq.data.validation import validate_dataset; print(validate_dataset('data/raw/Telco-Customer-Churn.csv').summary())"

# 3. Clean and generate interim dataset
& ".\.venv\Scripts\python.exe" -c "from predictiq.data.cleaning import process_raw_to_interim; process_raw_to_interim()"
```

### Step 6: Run Automated Tests
```powershell
& ".\.venv\Scripts\pytest.exe" tests -v
```
*Expected: 63 passed in < 0.8s.*

### Step 7: Execute Exploratory Data Analysis Notebook
```powershell
& ".\.venv\Scripts\python.exe" -m nbconvert --to notebook --execute "notebooks/01_eda_telco_churn.ipynb"
```

---

## 7. Docker Infrastructure Operations

Docker Compose orchestrates PostgreSQL and the MLflow Tracking Server.

### Validate Configuration Syntax
```powershell
docker compose config
```

### Start Services (When Docker Desktop is Running)
```powershell
docker compose up -d
docker compose ps
```

### Access UIs
- **MLflow Tracking UI:** [http://localhost:5000](http://localhost:5000)
- **PostgreSQL Database:** `127.0.0.1:5432`

### Shutdown Services
```powershell
# Stop services and keep persistent data
docker compose down

# Stop services and wipe volumes (destructive)
docker compose down -v
```

---

## 8. Troubleshooting & Common Issues

| Issue | Symptom | Solution |
|---|---|---|
| **Script Execution Policy** | `cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell. |
| **Docker Daemon Inactive** | `failed to connect to the docker API at npipe://...` | Launch Docker Desktop from Windows Start Menu and wait until running. |
| **Port Conflicts** | `port 5432 or 5000 is already allocated` | Update `POSTGRES_PORT` or `MLFLOW_PORT` in your local `.env` file. |
| **Missing Module in Tests** | `ModuleNotFoundError: No module named 'predictiq'` | Run `pip install -e .` from within your `.venv`. |

---

## 9. Current Implementation Status & Limitations

### Phase 1 Implemented
- [x] Standard `src/` modular layout and directory hierarchy.
- [x] Typed, zero-dependency configuration system with strict input validation.
- [x] Docker Compose definition for PostgreSQL 16 + MLflow 2.19 with health checks and persistent volumes.
- [x] Comprehensive pytest test suite (42 tests passing).
- [x] Complete architecture documentation, ADR-001, changelog, and setup guide.

### Phase 3 Implemented
- [x] Leakage-safe stratified train/test split with zero index overlap and class preservation (`src/predictiq/data/loader.py`).
- [x] Reusable, testable domain feature engineering pipeline (`src/predictiq/features/`).
- [x] Naive baseline (`DummyClassifier`), linear baseline (`LogisticRegression`), and tree-based baseline (`RandomForestClassifier`).
- [x] 5-Fold Stratified Cross-Validation strictly restricted to training data.
- [x] Full classification evaluation suite (PR-AUC, ROC-AUC, Recall, Precision, F1, Brier score, Confusion Matrices).
- [x] Local and remote MLflow experiment tracking integration (`src/predictiq/tracking/`).
- [x] Joblib model persistence paired with rich metadata (`src/predictiq/models/persistence.py`).
- [x] Production inference engine with schema validation and risk tiers (`src/predictiq/inference/predictor.py`).
- [x] Standalone CLI execution script (`scripts/train_baseline.py`).
- [x] 23 new deterministic unit tests added (total **86 passing tests**).

### Execution Commands (Phase 3 Modeling & Inference)
```powershell
# Run baseline model training, cross-validation, and MLflow logging
python scripts/train_baseline.py

# Run all 86 unit and integration tests
pytest -v

# Run Ruff linter
ruff check .

# Test inference on saved model
python -c "
from predictiq.inference import ChurnPredictor
predictor = ChurnPredictor.from_saved_model()
print('Predictor successfully loaded and operational!')
"
```

### Current Limitations & Exclusions
- **Hyperparameter search & threshold calibration:** Model baselines use sensible defaults. Comprehensive hyperparameter optimization and profit-maximizing threshold calibration belong to Phase 4.
- **Docker runtime requires active Docker Desktop:** `docker compose config` validates deterministically; local file tracking (`./mlruns`) acts as the verified fallback when the Docker daemon is offline.
