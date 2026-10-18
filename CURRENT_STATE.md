# PredictIQ — Current State Assessment

**Last Updated:** 2026-10-18  
**Current Phase:** Phase 4: Evaluation Protocol Correction, Advanced Modeling, Calibration, Threshold Optimization & Explainability  
**Status:** COMPLETE & VERIFIED  

---

## 1. Phase Objective
1. **Part A (Evaluation Protocol Correction)**:
   - Audit Phase 3 evaluation workflow and classify historical Seed 42 test metrics as exploratory.
   - Establish model selection strictly driven by training-only 5-fold Stratified Cross-Validation on Average Precision (PR-AUC).
   - Create and document a fresh, unobserved held-out evaluation split (Seed 2026, $N=1,409$).
   - Freeze candidate model, calibration parameters, and decision threshold before evaluating on held-out test data once.
2. **Part B (Advanced Modeling & Optimization)**:
   - Verify compatibility of advanced models on Python 3.13 (LightGBM 4.7.0).
   - Implement reproducible hyperparameter optimization on training CV (`src/predictiq/models/tuning.py`).
   - Implement probability calibration via Platt scaling (`src/predictiq/models/calibration.py`).
   - Optimize business decision threshold using training out-of-fold predictions with cost matrix support (`src/predictiq/models/thresholds.py`).
   - Implement global and local archetype explainability using TreeSHAP (`src/predictiq/models/explainability.py`).
   - Persist champion pipeline with metadata (`models/churn_champion_model.joblib`).
   - Add comprehensive tests and maintain 100% test pass rate.

---

## 2. Actual Repository Structure

```
D:\Projects\PredictIQ\
├── .env.example                     # Verified: safe defaults, uncommitted template
├── .gitignore                        # Verified: excludes .env, venv, data/*, artifacts, mlruns, models/*.joblib
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
│   ├── PHASE_3_IMPLEMENTATION_REPORT.md # Verified: Phase 3 report
│   └── PHASE_4_IMPLEMENTATION_REPORT.md # Verified: Phase 4 comprehensive report
├── models\
│   ├── churn_champion_model.joblib   # Verified: Champion pipeline binary (ignored by Git)
│   └── churn_champion_model_metadata.json # Verified: Model metadata, metrics, and threshold
├── NEXT_TASK.md                      # Verified: Phase 5 scope & acceptance criteria
├── notebooks\
│   ├── README.md                     # Verified: Reproducibility guidelines
│   └── 01_eda_telco_churn.ipynb      # Verified: Executed 25-cell EDA & hypothesis tests
├── pyproject.toml                    # Verified: Dependencies with lightgbm, shap, scikit-learn, etc.
├── README.md                         # Verified: Complete project guide & execution commands
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
│   ├── train_baseline.py             # Verified: Baseline model training CLI
│   └── tune_and_train.py             # Verified: Phase 4 tuning, calibration, and training CLI
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
│       │   ├── __init__.py           # Verified: exports baselines, evaluation, train, persistence, tuning, calibration, thresholds, explainability
│       │   ├── baselines.py          # Verified: Dummy, Logistic, Random Forest, LightGBM builders
│       │   ├── calibration.py        # Verified: Platt scaling & reliability curves
│       │   ├── evaluation.py         # Verified: metrics extraction & figure plotting
│       │   ├── explainability.py     # Verified: TreeSHAP attribution & archetype waterfall plots
│       │   ├── persistence.py        # Verified: joblib serialization & loading
│       │   ├── thresholds.py         # Verified: OOF decision threshold optimization
│       │   ├── train.py              # Verified: end-to-end training & CV orchestration
│       │   └── tuning.py             # Verified: RandomizedSearchCV on training CV folds
│       └── tracking\
│           ├── __init__.py           # Verified: exports MLflowTrackingManager
│           └── manager.py            # Verified: MLflow tracking with local directory fallback
└── tests\
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

## 3. Implemented vs. Planned Functionality

| Capability | Planned in Phase 4 | Status | Verification Evidence |
|---|---|---|---|
| Evaluation Protocol Independence | Yes | **Completed** | Training-only CV selection; isolated evaluation split (seed 2026) |
| Modern Gradient Boosting | Yes | **Completed** | LightGBM integrated and verified on Python 3.13.14 |
| Hyperparameter Optimization | Yes | **Completed** | `RandomizedSearchCV` on training 5-fold CV (Random Forest: 0.6607 PR-AUC) |
| Probability Calibration | Yes | **Completed** | Platt sigmoid scaling; Brier score reduced from 0.1602 to 0.1354 (+15.42%) |
| Decision Threshold Optimization | Yes | **Completed** | Training OOF F1-maximization selected $p^* = 0.3400$; cost matrix supported |
| Explainability (TreeSHAP) | Yes | **Completed** | Global beeswarm summary + High/Low risk archetype local waterfalls |
| Single Held-Out Evaluation | Yes | **Completed** | Evaluated once: PR-AUC 0.6684, ROC-AUC 0.8428, F1 0.6231 (Recall 71.39%) |
| Model Persistence & Metadata | Yes | **Completed** | `models/churn_champion_model.joblib` and metadata saved with $p^* = 0.3400$ |
| Inference Dynamic Threshold | Yes | **Completed** | `ChurnPredictor` loads threshold and outputs detailed predictions |
| Deterministic Test Suite | Yes | **Completed** | 11 new tests added; total **97 passing tests** |
| FastAPI inference service | No (Phase 5) | Deferred | Scheduled for Phase 5 |
| Monitoring & drift detection | No (Phase 6) | Deferred | Scheduled for Phase 6 |

---

## 4. Environment & Infrastructure Verification

| Component | Status | Details |
|---|---|---|
| **Operating System** | Verified | Windows 11 (PowerShell runtime) |
| **Python** | Verified | `Python 3.13.14` |
| **Virtual Environment** | Verified | `D:\Projects\PredictIQ\.venv` with `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `shap 0.54.1`, `joblib 1.6.0`, `mlflow 3.17.1`, `ruff 0.17.1` |
| **Test Suite** | Verified | **97 passed** in 20.40s with zero failures |
| **Linter** | Verified | Ruff: `All checks passed!` |
| **MLflow Store** | Verified | Local filesystem store (`./mlruns`) functional with verified run `champion_random_forest_1783` |
| **Docker Daemon Runtime** | Environmental Blocker | Docker Desktop process not running on host; local file fallback operational |

---

## 5. Automated Tests Executed & Actual Results

- **Command:** `& "D:\Projects\PredictIQ\.venv\Scripts\python.exe" -m pytest -v`
- **Result:**
  ```
  collected 97 items
  97 passed, 1 warning in 20.40s
  ```
- **Coverage Breakdown:**
  - `tests/test_calibration.py`: 2 tests passed (sigmoid calibration, Brier score reduction, curve plotting).
  - `tests/test_config.py`: 42 tests passed (version string, ports, paths, database, MLflow, environment modes, caching).
  - `tests/test_data_ingestion.py`: 9 tests passed (SHA-256 calculation, metadata serialization, local import, mismatch errors).
  - `tests/test_data_loader.py`: 7 tests passed (splitting reproducibility, stratification balance, zero index overlap, target validation).
  - `tests/test_data_validation.py`: 12 tests passed (valid dataset, missing columns, unexpected columns, duplicate IDs, invalid target labels, domain rules).
  - `tests/test_evaluation_protocol.py`: 2 tests passed (split seed independence, identical fold comparison).
  - `tests/test_explainability.py`: 1 test passed (TreeSHAP attributions, feature ranking, and archetype plots).
  - `tests/test_features.py`: 6 tests passed (feature engineering, service count logic, unseen category handling).
  - `tests/test_inference.py`: 4 tests passed (model serialization, schema validation, risk tiers).
  - `tests/test_model_evaluation.py`: 4 tests passed (metrics extraction, probability checks, artifact export).
  - `tests/test_thresholds.py`: 3 tests passed (OOF threshold search, cost matrix trade-offs, plot generation).
  - `tests/test_tracking.py`: 2 tests passed (MLflow tracking initialization, local directory fallback).
  - `tests/test_tuning.py`: 3 tests passed (search space definitions, CV execution, candidate tuning).

---

## 6. Known Blockers & Limitations
1. **Docker Daemon Execution:**
   - Docker Compose configuration is validated and ready for startup.
   - Host Docker Desktop engine is currently stopped. Local file fallback (`./mlruns`) operates reliably.
2. **Explainability Causal Limitation:**
   - SHAP attributions represent model predictive associations, not causal interventions.

---

## 7. Last Verified Milestone
**Milestone:** Phase 4: Evaluation Protocol Correction, Advanced Modeling, Calibration, Threshold Optimization & Explainability.  
**Exit Criteria Met:**
- Evaluation protocol audited and corrected; model selection governed strictly by training-only 5-fold CV.
- Seed 2026 isolated evaluation split created, locked, and evaluated exactly once.
- Hyperparameter tuning across Logistic Regression, Random Forest, and LightGBM complete.
- Platt sigmoid calibration reduced Brier score from 0.1602 to 0.1354 (+15.42%).
- Decision threshold optimized on training OOF to $p^* = 0.3400$, boosting churn recall to 71.39%.
- TreeSHAP explainability generated global summary and local archetype waterfalls.
- 97 of 97 automated tests passing.
- Ready for handoff to Phase 5 (Production REST API Service).
