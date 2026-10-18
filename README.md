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
│   ├── raw/                          # Original immutable datasets (.gitkeep)
│   ├── interim/                      # Cleaned intermediate transformations (.gitkeep)
│   ├── processed/                    # Model-ready features & splits (.gitkeep)
│   └── external/                     # Third-party benchmark data (.gitkeep)
├── docker-compose.yml                # PostgreSQL 16 & MLflow 2.19 orchestration
├── docs/                             # Architecture & operational guides
│   ├── adr/
│   │   └── ADR-001-initial-architecture.md  # Accepted architecture decision record
│   ├── ARCHITECTURE.md               # End-to-end system design & layer boundaries
│   ├── CHANGELOG.md                  # Versioned change history & validation notes
│   └── DEVELOPMENT_SETUP.md          # Exact Windows PowerShell setup instructions
├── NEXT_TASK.md                      # Phase 2 scope, prerequisites, and handoff plan
├── notebooks/                        # Exploratory analysis & prototyping
│   └── README.md                     # Notebook reproducibility guidelines
├── pyproject.toml                    # Package build metadata & pytest configuration
├── README.md                         # Project overview and quickstart guide
├── src/                              # Application source tree (src/ layout)
│   └── predictiq/
│       ├── __init__.py               # Package exports & version metadata
│       └── config.py                 # Typed configuration, validation, & .env loader
└── tests/                            # Automated test suite
    └── test_config.py                # 42 unit tests for config, ports, paths, & env
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
& ".\.venv\Scripts\pip.exe" install -e ".[dev]"
```

### Step 4: Configure Local Environment
```powershell
Copy-Item -Path ".env.example" -Destination ".env"
```

### Step 5: Run Automated Tests
```powershell
& ".\.venv\Scripts\pytest.exe" tests -v
```
*Expected: 42 passed in < 0.15s.*

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

### Current Limitations & Exclusions
- **No ML models or training pipelines yet:** By design, Phase 1 establishes infrastructure and configuration. Modeling begins in Phase 4.
- **No data ingestion or EDA executed yet:** Scheduled for Phase 2.
- **Docker runtime requires active Docker Desktop:** `docker compose config` validates deterministically; actual container startup depends on Docker Desktop running in the user session.
