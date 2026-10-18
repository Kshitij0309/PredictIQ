# ADR-001: Initial Architecture and Foundation Environment

- **Status:** Accepted
- **Date:** 2026-10-18
- **Deciders:** PredictIQ Architecture & Engineering Team
- **Technical Context:** Phase 1 Project Foundation & Architecture

---

## 1. Context and Problem Statement
PredictIQ is envisioned as a production-grade predictive intelligence platform showcasing both Data Science and Machine Learning Engineering capabilities. Prior to starting data ingestion, feature engineering, or model training, the project requires a clean, reproducible, and verifiable foundation.

Key challenges to address:
1. Ensuring strict boundary separation between data, analysis notebooks, core library code, containerized infrastructure, and automated tests.
2. Preventing dependency bloat and configuration drift across developer workstations.
3. Establishing reproducible experiment tracking with MLflow and PostgreSQL without leaking credentials or hardcoding host configurations.
4. Supporting standard Windows PowerShell developer workflows without requiring esoteric shell emulators.

---

## 2. Decision Drivers
- **Reproducibility:** The environment and tests must run identically across fresh clones.
- **Minimal Initial Footprint:** Avoid prematurely introducing heavy ML frameworks (scikit-learn, XGBoost, PyTorch, FastAPI) until their respective implementation phases.
- **Robust Configuration:** Configuration must be strictly typed, validated at runtime, and decoupled from code via environment variables (12-Factor methodology).
- **Auditability & Documentation:** Architectural decisions, setup steps, and state changes must be recorded in durable Markdown files.

---

## 3. Considered Alternatives

### Alternative 1: Flat Project Structure vs. `src/` Layout
- **Flat Layout:** Placing modules directly in repository root (`predictiq/` side-by-side with `tests/` and root files).
  - *Cons:* Prone to import parity bugs where tests accidentally import from local directory rather than installed package, masking packaging defects.
- **Chosen: `src/` Layout (`src/predictiq/`):**
  - *Pros:* Enforces proper packaging, forces tests to test against the installed package in editable mode (`pip install -e .`), preventing subtle packaging errors.

### Alternative 2: Pydantic / Dynaconf vs. Standard Library Dataclasses for Phase 1
- **Pydantic / Dynaconf:** Feature-rich configuration libraries.
  - *Cons:* Adds heavy external dependencies and potential version pinning conflicts before any business logic is written.
- **Chosen: Standard Library `dataclasses` + Custom Type Checking:**
  - *Pros:* Zero external dependencies, ultra-fast test execution (0.08s), zero friction across Python versions (3.10 through 3.13), custom `ConfigurationError` providing transparent error messages.

### Alternative 3: Local SQLite Tracking vs. Containerized PostgreSQL + MLflow
- **Local SQLite File:** Running MLflow directly against `sqlite:///mlruns.db`.
  - *Cons:* SQLite suffers from concurrency locking during parallel hyperparameter searches and does not mirror production cloud topologies.
- **Chosen: Docker Compose with PostgreSQL 16 + MLflow 2.19:**
  - *Pros:* Production-aligned relational metadata store, containerized process isolation, explicit persistent volumes (`predictiq_postgres_data` and `predictiq_mlflow_artifacts`), deterministic container health checks.

---

## 4. Decision Outcome

We decided to implement:
1. **Packaging:** A standard `src/` layout managed by `pyproject.toml` (PEP 517/621) with `setuptools`.
2. **Configuration:** A typed, immutable `dataclass` configuration hierarchy in `src/predictiq/config.py` with port range validation, environment mode validation, and zero third-party dependencies.
3. **Infrastructure:** A minimal Docker Compose file declaring `postgres:16.8-alpine` (healthy check via `pg_isready`) and `ghcr.io/mlflow/mlflow:v2.19.0` (healthy check via HTTP probe) bound to `127.0.0.1`.
4. **Data Hierarchy:** Strict folder separation under `data/` (`raw`, `interim`, `processed`, `external`) with `.gitkeep` placeholders and `.gitignore` exclusion rules.
5. **Testing Framework:** `pytest` test suite covering package exports, configuration parsing, error handling, and directory resolution.

---

## 5. Consequences & Trade-offs

### Positive
- **Instant Testing Feedback:** 42 unit tests execute in ~0.08 seconds without requiring Docker to be running for pure logic validation.
- **Portability:** Works seamlessly across Windows PowerShell, Linux, and macOS.
- **Security:** Services are bound strictly to `127.0.0.1`; `.gitignore` prevents inadvertent leakage of `.env` or data binaries.
- **Clear Roadmap:** Modular separation prepares clean landing zones for Phase 2 (Data Ingestion & EDA) and Phase 3 (Feature Engineering).

### Negative / Risks
- Running MLflow and PostgreSQL requires Docker Desktop to be active on the host machine. If Docker daemon is stopped, container services cannot be brought up (environmental constraint documented in troubleshooting).
- Windows PowerShell users must ensure script execution policy allows virtual environment activation script (`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`).
