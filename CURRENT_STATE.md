# PredictIQ — Current State Assessment

**Last Updated:** 2026-10-18  
**Current Phase:** Phase 2: Dataset Acquisition, Validation & Exploratory Data Analysis  
**Status:** COMPLETE & VERIFIED  

---

## 1. Phase Objective
Acquire the public IBM Telco Customer Churn dataset from the verified upstream IBM repository, implement a non-destructive data validation and cleaning pipeline under `src/predictiq/data/`, produce an end-to-end reproducible exploratory data analysis notebook with formal statistical hypothesis testing, and add deterministic unit tests.

---

## 2. Actual Repository Structure

```
D:\Projects\PredictIQ\
├── .env.example                     # Verified: safe defaults, uncommitted template
├── .gitignore                        # Verified: excludes .env, venv, data/*, artifacts
├── artifacts\
│   └── .gitkeep                      # Verified: directory anchor
├── CURRENT_STATE.md                  # This file
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
├── docker-compose.yml                # Verified: PostgreSQL 16 + MLflow 2.19
├── docs\
│   ├── adr\
│   │   └── ADR-001-initial-architecture.md  # Verified: Accepted ADR
│   ├── data\
│   │   └── DATASET_CARD.md           # Verified: Dataset card (provenance, schema, ethics)
│   ├── ARCHITECTURE.md               # Verified: System design & data layer boundaries
│   ├── CHANGELOG.md                  # Verified: v0.2.0 release notes
│   └── DEVELOPMENT_SETUP.md          # Verified: PowerShell setup & verification
├── NEXT_TASK.md                      # Verified: Phase 3 scope & acceptance criteria
├── notebooks\
│   ├── README.md                     # Verified: Reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # Verified: Executed 25-cell EDA & hypothesis tests
├── pyproject.toml                    # Verified: PEP 517 package config with analysis dependencies
├── README.md                         # Verified: Complete project guide & Phase 2 quickstart
├── src\
│   └── predictiq\
│       ├── __init__.py               # Verified: exports Config, version 0.1.0
│       ├── config.py                 # Verified: typed dataclasses, validation, .env
│       └── data\
│           ├── __init__.py           # Verified: exports ingestion, validation, cleaning
│           ├── ingestion.py          # Verified: SHA-256 provenance tracking & acquisition
│           ├── validation.py         # Verified: schema, constraint, & domain validation
│           └── cleaning.py           # Verified: type normalization & interim export
└── tests\
    ├── test_config.py                # Verified: 42 pytest foundation unit tests
    ├── test_data_ingestion.py        # Verified: 9 unit tests for acquisition & hashing
    └── test_data_validation.py       # Verified: 12 unit tests for schema & domain rules
```

---

## 3. Implemented vs. Planned Functionality

| Capability | Planned in Phase 2 | Status | Verification Evidence |
|---|---|---|---|
| Dataset Acquisition & Hashing | Yes | **Completed** | SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Dataset Provenance Tracking | Yes | **Completed** | Metadata JSON in `data/metadata/Telco-Customer-Churn_metadata.json` |
| Schema & Domain Validation | Yes | **Completed** | `DatasetValidator` with non-destructive `ValidationReport` |
| Data Cleaning & Normalization | Yes | **Completed** | `clean_telco_churn_data` imputes 11 blank TotalCharges (tenure=0) |
| Interim Data Export | Yes | **Completed** | Cleaned dataset in `data/interim/Telco-Customer-Churn-cleaned.csv` |
| Exploratory Analysis Notebook | Yes | **Completed** | `notebooks/01_eda_telco_churn.ipynb` executed headlessly (exit code 0) |
| Hypothesis Testing (Contract) | Yes | **Completed** | Chi-Square $\chi^2 = 1184.60, p = 5.86 \times 10^{-258}$, Cramér's $V = 0.410$ |
| Hypothesis Testing (Charges) | Yes | **Completed** | Welch's $t = 18.41, p = 8.59 \times 10^{-73}$, Cohen's $d = 0.446$ |
| Dataset Card Documentation | Yes | **Completed** | `docs/data/DATASET_CARD.md` with schema, ethics, and limitations |
| Deterministic Unit Tests | Yes | **Completed** | 21 new tests added; total 63 passing tests |
| Feature Engineering Pipeline | No (Phase 3) | Deferred | Scheduled for Phase 3 |
| Model Training & MLflow Logging | No (Phase 4) | Deferred | Scheduled for Phase 4 |
| FastAPI inference service | No (Phase 5) | Deferred | Scheduled for Phase 5 |
| Monitoring & drift detection | No (Phase 6) | Deferred | Scheduled for Phase 6 |

---

## 4. Environment & Infrastructure Verification

| Component | Status | Details |
|---|---|---|
| **Operating System** | Verified | Windows 11 (PowerShell 5.1/7.x runtime) |
| **Python** | Verified | `Python 3.13.14` at `C:\Users\mkshi\AppData\Local\Programs\Python\Python313\python.exe` |
| **Virtual Environment** | Verified | `D:\Projects\PredictIQ\.venv` with `pandas 3.0.6`, `numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `ipykernel 7.4.0`, `nbconvert 7.17.2` |
| **Git** | Verified | `git version 2.55.0.windows.3` installed; raw data excluded |
| **Docker CLI** | Verified | `Docker version 29.6.2`, `Docker Compose version v5.3.1` |
| **Docker Compose Config** | Verified | Validated with zero syntax errors via `docker compose config` |
| **Docker Daemon Runtime** | Environmental Blocker | Docker Desktop CLI is present, but the background daemon process is not running in the Windows session (`npipe:////./pipe/dockerDesktopLinuxEngine` connection refused). Container startup requires user to start Docker Desktop. |

---

## 5. Automated Tests Executed & Actual Results

- **Command:** `& 'D:\Projects\PredictIQ\.venv\Scripts\pytest.exe' 'D:\Projects\PredictIQ\tests' -v`
- **Result:**
  ```
  collected 63 items
  63 passed in 0.64s
  ```
- **Coverage Breakdown:**
  - `tests/test_config.py`: 42 tests passed (version string, ports, paths, database, MLflow, environment modes, caching).
  - `tests/test_data_ingestion.py`: 9 tests passed (SHA-256 calculation, metadata serialization, local import, mismatch errors).
  - `tests/test_data_validation.py`: 12 tests passed (valid dataset, missing columns, unexpected columns, duplicate IDs, invalid target labels, missing target, whitespace detection, out-of-bounds charges/tenure, domain consistency rules, markdown formatting).

---

## 6. Known Blockers & Limitations
1. **Docker Daemon Execution:**
   - Docker Compose configuration is validated and ready for startup.
   - However, the host Docker Desktop engine is currently stopped. Launching Docker Desktop is required before executing `docker compose up -d`.

---

## 7. Last Verified Milestone
**Milestone:** Phase 2: Dataset Acquisition, Validation & Exploratory Data Analysis.  
**Exit Criteria Met:**
- IBM Telco Customer Churn dataset acquired and verified via SHA-256 digest.
- Raw CSV strictly excluded from Git via `.gitignore`.
- Reusable, non-destructive validation engine operational with domain rules.
- Type normalization and interim dataset generation completed.
- Full exploratory analysis notebook executed with 2 formal hypothesis tests.
- Dataset card completed and accepted.
- 63 of 63 automated tests passing.
- Ready for handoff to Phase 3 (Feature Engineering).
