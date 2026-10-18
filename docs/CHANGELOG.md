# PredictIQ — Changelog

All notable changes, architectural decisions, and verification results for the PredictIQ platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
