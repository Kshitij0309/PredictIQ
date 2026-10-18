# PredictIQ — Next Task: Phase 2 (Data Ingestion & Exploratory Analysis)

## 1. Task Definition
**Task Name:** Phase 2 — Dataset Acquisition, Ingestion Pipeline & Exploratory Data Analysis (EDA)  
**Objective:** Ingest an industry-standard predictive intelligence dataset into `data/raw/`, implement an automated data ingestion and validation module, and conduct statistical exploratory data analysis within a reproducible notebook.

---

## 2. Scope & Boundaries
- **In Scope:**
  1. Selection and automated downloading of a multi-feature tabular dataset (e.g., Telco Customer Churn or Financial Credit Default).
  2. Implement `src/predictiq/data/ingestion.py` with checksum validation (SHA-256) and directory routing via `config.paths`.
  3. Implement `src/predictiq/data/cleaning.py` for type coercion, missing value audits, and writing intermediate tables to `data/interim/`.
  4. Develop `notebooks/01_eda_initial.ipynb` demonstrating distribution plots, categorical cardinality, target imbalance, and outlier detection.
  5. Add `tests/test_data_ingestion.py` testing download resilience, hash integrity, and schema parsing.
- **Out of Scope (Deferred to Later Phases):**
  - Feature engineering and scaling (Phase 3).
  - Model training, MLflow tracking, and hyperparameter tuning (Phase 4).
  - FastAPI serving endpoints (Phase 5).

---

## 3. Prerequisites
- [x] Phase 1 Project Foundation & Architecture complete, verified, and audited.
- [x] `.venv` virtual environment active (Python 3.13.14).
- [ ] Add data science dependencies (`pandas>=2.2.0`, `numpy>=1.26.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`) to `pyproject.toml`.

---

## 4. Files Likely to Change / Be Created
- `pyproject.toml`: Add data processing and analysis dependencies.
- `src/predictiq/data/__init__.py`: Data layer package exports.
- `src/predictiq/data/ingestion.py`: Ingestion logic and checksum verification.
- `src/predictiq/data/cleaning.py`: Data type normalization and cleaning pipeline.
- `notebooks/01_eda_initial.ipynb`: End-to-end exploratory data analysis notebook.
- `tests/test_data_ingestion.py`: Pytest suite for ingestion and validation functions.
- `docs/CHANGELOG.md`: Log Phase 2 implementation.
- `CURRENT_STATE.md`: Update milestone progress and verification logs.
- `NEXT_TASK.md`: Update handoff for Phase 3 (Feature Engineering).

---

## 5. Acceptance Criteria
1. The ingestion pipeline saves raw immutable files to `data/raw/` with verified cryptographic hashes.
2. The cleaning pipeline handles missing values without dropping critical records and outputs cleaned frames to `data/interim/`.
3. The exploratory notebook `notebooks/01_eda_initial.ipynb` runs top-to-bottom without manual intervention or errors.
4. Unit tests in `tests/test_data_ingestion.py` pass cleanly with `pytest`.
5. Raw datasets and intermediate data remain excluded from Git as enforced by `.gitignore`.

---

## 6. Verification Commands
```powershell
# 1. Run full test suite including new ingestion tests
& ".\.venv\Scripts\pytest.exe" tests -v

# 2. Run data ingestion CLI
& ".\.venv\Scripts\python.exe" -m predictiq.data.ingestion --verify

# 3. Execute and validate EDA notebook headlessly
& ".\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute notebooks/01_eda_initial.ipynb
```
