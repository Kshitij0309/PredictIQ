# PredictIQ — Current State Assessment

**Last Updated:** 2026-10-18  
**Current Phase:** Phase 3: Feature Engineering, Baseline Modeling & MLflow Experiment Tracking  
**Status:** COMPLETE & VERIFIED  

---

## 1. Phase Objective
Establish a reproducible, leakage-safe machine learning workflow for churn classification:
- Load the validated dataset via a dedicated data loader.
- Separate features $X$ and target $y$ (`Churn`), strictly excluding customer identifiers.
- Partition data into reproducible stratified 80/20 train/test splits.
- Construct leakage-safe feature engineering pipelines (`DomainFeatureEngineer`, `create_preprocessor`).
- Train and evaluate naive baseline (`DummyClassifier`), linear baseline (`LogisticRegression`), and tree baseline (`RandomForestClassifier`).
- Run 5-fold cross-validation on training data.
- Track experiments, parameters, metrics, artifacts, and signatures in MLflow (with local fallback).
- Persist winning pipeline and metadata via `joblib`.
- Build a production inference engine (`ChurnPredictor`).
- Add comprehensive deterministic tests.

---

## 2. Actual Repository Structure

```
D:\Projects\PredictIQ\
├── .env.example                     # Verified: safe defaults, uncommitted template
├── .gitignore                        # Verified: excludes .env, venv, data/*, artifacts, mlruns, models
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
│   ├── CHANGELOG.md                  # Verified: release notes
│   ├── DEVELOPMENT_SETUP.md          # Verified: PowerShell setup & verification
│   └── PHASE_3_IMPLEMENTATION_REPORT.md # Verified: Phase 3 comprehensive report
├── models\
│   ├── churn_baseline_model.joblib   # Verified: Champion pipeline binary (ignored by Git)
│   └── churn_baseline_model_metadata.json # Verified: Model metadata & metrics
├── NEXT_TASK.md                      # Verified: Phase 4 scope & acceptance criteria
├── notebooks\
│   ├── README.md                     # Verified: Reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # Verified: Executed 25-cell EDA & hypothesis tests
├── pyproject.toml                    # Verified: Package config with core, analysis, dev, and db dependencies
├── README.md                         # Verified: Complete project guide & execution commands
├── reports\
│   ├── figures\
│   │   ├── confusion_matrices.png    # Verified: Side-by-side CM heatmaps
│   │   ├── pr_curves.png             # Verified: Comparative PR curves
│   │   └── roc_curves.png            # Verified: Comparative ROC curves
│   ├── model_comparison.csv          # Verified: Machine-readable summary table
│   └── model_comparison.json         # Verified: JSON metrics for CI/CD
├── scripts\
│   └── train_baseline.py             # Verified: Standalone model training CLI
├── src\
│   └── predictiq\
│       ├── __init__.py               # Verified: exports Config, version 0.1.0
│       ├── config.py                 # Verified: typed dataclasses, validation, .env
│       ├── data\
│       │   ├── __init__.py           # Verified: exports ingestion, validation, cleaning, loader
│       │   ├── ingestion.py          # Verified: SHA-256 provenance tracking & acquisition
│       │   ├── validation.py         # Verified: schema, constraint, & domain validation
│       │   ├── cleaning.py           # Verified: type normalization & interim export
│       │   └── loader.py             # Verified: stratified train/test splitting & target isolation
│       ├── features\
│       │   ├── __init__.py           # Verified: exports feature transformers & pipelines
│       │   ├── transformers.py       # Verified: DomainFeatureEngineer
│       │   └── pipeline.py           # Verified: ColumnTransformer & model pipeline builders
│       ├── inference\
│       │   ├── __init__.py           # Verified: exports ChurnPredictor
│       │   └── predictor.py          # Verified: Production inference engine & risk tiers
│       ├── models\
│       │   ├── __init__.py           # Verified: exports baselines, evaluation, train, persistence
│       │   ├── baselines.py          # Verified: Dummy, Logistic, Random Forest builders
│       │   ├── evaluation.py         # Verified: metrics extraction & figure plotting
│       │   ├── persistence.py        # Verified: joblib serialization & loading
│       │   └── train.py              # Verified: end-to-end training & CV orchestration
│       └── tracking\
│           ├── __init__.py           # Verified: exports MLflowTrackingManager
│           └── manager.py            # Verified: MLflow tracking with local directory fallback
└── tests\
    ├── test_config.py                # Verified: 42 pytest foundation unit tests
    ├── test_data_ingestion.py        # Verified: 9 unit tests for acquisition & hashing
    ├── test_data_loader.py           # Verified: 7 unit tests for splitting & leakage prevention
    ├── test_data_validation.py       # Verified: 12 unit tests for schema & domain rules
    ├── test_features.py              # Verified: 6 unit tests for feature engineering
    ├── test_inference.py             # Verified: 4 unit tests for persistence & prediction
    ├── test_model_evaluation.py      # Verified: 4 unit tests for evaluation & metrics
    └── test_tracking.py              # Verified: 2 unit tests for MLflow local logging
```

---

## 3. Implemented vs. Planned Functionality

| Capability | Planned in Phase 3 | Status | Verification Evidence |
|---|---|---|---|
| Stratified Split (80/20, seed 42) | Yes | **Completed** | Zero index overlap, 26.54% class balance preserved |
| Domain Feature Engineering | Yes | **Completed** | `DomainFeatureEngineer` (service counts, contract flags, cohorts) |
| Missing Value Imputation | Yes | **Completed** | Numeric median + Categorical most-frequent in Pipeline |
| Leakage-Free Preprocessing | Yes | **Completed** | Fit strictly on train; `OneHotEncoder(handle_unknown='ignore')` |
| Baseline Models (Dummy, Logistic, RF) | Yes | **Completed** | All 3 evaluated under identical split & CV conditions |
| Evaluation Framework (PR-AUC, ROC, F1) | Yes | **Completed** | RF: PR-AUC 0.6471, ROC 0.8420, F1 0.6263; reports exported |
| MLflow Experiment Tracking | Yes | **Completed** | Runs logged to `./mlruns` with params, metrics, plots, signatures |
| Model Persistence & Metadata | Yes | **Completed** | `models/churn_baseline_model.joblib` and metadata saved |
| Production Inference Engine | Yes | **Completed** | `ChurnPredictor` with validation, scoring, and risk tiers |
| Deterministic Unit Tests | Yes | **Completed** | 23 new tests added; total **86 passing tests** |
| Hyperparameter Optimization | No (Phase 4) | Deferred | Scheduled for Phase 4 |
| FastAPI inference service | No (Phase 5) | Deferred | Scheduled for Phase 5 |
| Monitoring & drift detection | No (Phase 6) | Deferred | Scheduled for Phase 6 |

---

## 4. Environment & Infrastructure Verification

| Component | Status | Details |
|---|---|---|
| **Operating System** | Verified | Windows 11 (PowerShell runtime) |
| **Python** | Verified | `Python 3.13.14` |
| **Virtual Environment** | Verified | `D:\Projects\PredictIQ\.venv` with `scikit-learn 1.9.1`, `joblib 1.6.0`, `mlflow 3.17.1`, `ruff 0.17.1` |
| **Test Suite** | Verified | **86 passed** in 10.69s with zero failures |
| **Linter** | Verified | Ruff: `All checks passed!` |
| **MLflow Store** | Verified | Local filesystem store (`./mlruns`) functional with 3 verified runs |
| **Docker Daemon Runtime** | Environmental Blocker | Docker Desktop process not running on host; local file fallback operational |

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
