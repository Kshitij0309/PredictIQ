# PredictIQ — Changelog

All notable changes, architectural decisions, and verification results for the PredictIQ platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-18

### Phase 2: Dataset Acquisition, Validation & Exploratory Data Analysis

#### Added
- **Data Ingestion Engine (`src/predictiq/data/ingestion.py`):**
  - Automated downloading from official IBM repository (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`).
  - Cryptographic identity verification via SHA-256 (`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, size: 970,457 bytes).
  - Provenance metadata generator writing to `data/metadata/Telco-Customer-Churn_metadata.json`.
  - Non-overwriting acquisition logic and discrimination between automated downloads and manual imports.
  - CLI runner for acquisition and verification: `python -m predictiq.data.ingestion`.
- **Data Validation Engine (`src/predictiq/data/validation.py`):**
  - Reusable, non-destructive validation engine checking canonical schema, 0 duplicate identifiers, target distribution, numerical boundaries, and domain constraints.
  - Generates comprehensive `ValidationReport` with Markdown formatting and actionable recommendations.
  - Audited 11 blank `TotalCharges` values to brand-new accounts (`tenure == 0`, `Churn == 'No'`) with recommendation to impute `TotalCharges = 0.0`.
- **Data Cleaning & Normalization (`src/predictiq/data/cleaning.py`):**
  - Type normalization pipeline coercing `TotalCharges` to `float64`, `tenure` to `int64`, and adding `tenure_cohort` buckets.
  - Generates cleaned interim datasets in `data/interim/Telco-Customer-Churn-cleaned.csv`.
- **Exploratory Data Analysis (`notebooks/01_eda_telco_churn.ipynb`):**
  - Executed 25-cell reproducible notebook with full tables and visual charts.
  - Formal Hypothesis 1: Contract duration vs. Churn ($\chi^2 = 1184.60, p = 5.86 \times 10^{-258}$, Cramér's $V = 0.410$).
  - Formal Hypothesis 2: Monthly charges vs. Churn (Welch's $t = 18.41, p = 8.59 \times 10^{-73}$, Cohen's $d = 0.446$, 95% CI: [+$11.77, +$14.58]).
  - Comprehensive analytical summary with Q&A, key findings, and next steps.
- **Dataset Documentation (`docs/data/DATASET_CARD.md`):**
  - Comprehensive dataset card detailing provenance, schema, audit observations, intended use, and ethical guardrails.
- **Automated Tests:**
  - `tests/test_data_ingestion.py`: 9 deterministic unit tests for hashing, metadata, and acquisition.
  - `tests/test_data_validation.py`: 12 deterministic unit tests for schemas, missing columns, bounds, and domain consistency.

#### Validation Outcomes
- **Test Suite:** 63 of 63 tests passed in 0.64s.
- **Raw Data Isolation:** Raw CSV verified excluded from Git via `.gitignore`.
- **Notebook Execution:** Headless execution via `nbconvert` completed with exit code 0.

---

## [0.1.0] - 2026-10-18

### Phase 1: Project Foundation & Architecture

#### Added
- **Repository Structure:**
  - Standard `src/` layout establishing modular layer separation: Data Layer (`data/{raw,interim,processed,external}`), Analysis Layer (`notebooks/`), Core Application (`src/predictiq/`), Infrastructure (`docker-compose.yml`), Documentation (`docs/`), and Tests (`tests/`).
  - `.gitkeep` anchors across data and artifact directories to retain directory tree in Git without tracking generated binaries or data.
- **Dependency & Package Configuration:**
  - `pyproject.toml` adhering to PEP 517 / PEP 621 using `setuptools` build backend.
  - Zero heavy third-party framework dependencies in Phase 1; configured `pytest>=8.0.0` as primary development requirement.
  - Python version requirement pinned to `>=3.10` (verified against active runtime Python 3.13.14).
- **Configuration Subsystem:**
  - `src/predictiq/config.py`: High-performance, zero-dependency typed dataclasses (`PathConfig`, `DatabaseConfig`, `MLflowConfig`, `Config`).
  - Strict input validation for network ports (1..65535, rejects floats and non-integers), operational environment modes (`development`, `testing`, `staging`, `production`), and log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
  - Custom `load_env_file()` standard-library parser supporting quoted strings, comments, and non-destructive `setdefault` injection.
  - Dynamic path resolution anchored to project root.
  - Custom `ConfigurationError` exception hierarchy for fail-fast error reporting.
  - Package exports in `src/predictiq/__init__.py`.
- **Infrastructure:**
  - `docker-compose.yml` declaring PostgreSQL 16 (`postgres:16.8-alpine`) and MLflow Tracking Server (`ghcr.io/mlflow/mlflow:v2.19.0`).
  - Named persistent volumes: `predictiq_postgres_data` and `predictiq_mlflow_artifacts`.
  - Health checks: `pg_isready` for PostgreSQL and HTTP health probing for MLflow.
  - Strict dependency ordering (`depends_on: postgres: condition: service_healthy`).
  - Localhost binding on `127.0.0.1` to prevent inadvertent public exposure.
- **Environment & Git Safety:**
  - `.gitignore` covering Python cache, compiled artifacts, virtual environments, environment secrets (`.env*`), data layers (`data/*`), and local Docker mounts.
  - `.env.example` detailing all configurable variables with development defaults.
- **Test Automation:**
  - `tests/test_config.py`: 42 comprehensive unit tests covering package exports, port parsing, path computation, environment overrides, error boundaries, and `.env` parsing.
- **Architecture & Operational Documentation:**
  - `README.md`: Central project landing page, stack overview, architecture map, setup commands, and limitations.
  - `docs/ARCHITECTURE.md`: Comprehensive system specification, component boundaries, data-to-model workflow, secrets policy, and scope matrix.
  - `docs/DEVELOPMENT_SETUP.md`: Windows PowerShell setup, exact commands, expected outputs, and recovery instructions.
  - `docs/adr/ADR-001-initial-architecture.md`: Accepted Architecture Decision Record for Phase 1.
  - `notebooks/README.md`: Notebook standards, reproducibility rules, and hygiene guidelines.
  - `CURRENT_STATE.md`: Verified status of environment, tests, and deliverables.
  - `NEXT_TASK.md`: Detailed plan and acceptance criteria for Phase 2 (Data Ingestion & EDA).

#### Validation Outcomes
- **Package Installation:** `pip install -e .` succeeded in `.venv` (Python 3.13.14).
- **Test Suite:** 42 of 42 tests passed in 0.08 seconds (re-verified in 0.07s during audit).
- **Docker Compose Validation:** `docker compose -f docker-compose.yml config` validated with zero syntax or schema warnings.
- **Environmental Note:** Docker daemon execution deferred to user runtime (environmental dependency on Docker Desktop running in Windows user session).

#### Verification Audit
- **Audit Date:** 2026-10-18
- **Auditor:** Senior Python/ML Platform Engineer
- **Verdict:** All Phase 1 criteria verified. 42/42 tests passing. Zero defects found. Phase 2 execution authorized.
