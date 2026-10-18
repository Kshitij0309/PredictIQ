# PredictIQ — Changelog

All notable changes, architectural decisions, and verification results for the PredictIQ platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-18

### Phase 4: Evaluation Protocol Correction, Advanced Modeling, Calibration, Threshold Optimization & Explainability

#### Added
- **Part A — Evaluation Protocol Correction (`src/predictiq/data/loader.py`, `tests/test_evaluation_protocol.py`):**
  - Audited and corrected Phase 3 evaluation protocol; classified historical Seed 42 test scores as exploratory benchmarks.
  - Enforced model selection governed exclusively by training-only 5-fold Stratified Cross-Validation on Average Precision (PR-AUC) across identical folds.
  - Created isolated evaluation split seed (`ISOLATED_EVALUATION_SPLIT_SEED = 2026`) preserving class balance (26.54% churn) with zero index leakage.
  - Frozen champion pipeline, calibration parameters, and decision threshold before single held-out evaluation.
- **Part B — Advanced Gradient Boosted Trees (`src/predictiq/models/baselines.py`):**
  - Verified LightGBM compatibility on Python 3.13.14.
  - Added `build_lightgbm_baseline` with bounded depth and leaf-wise histogram growth.
- **Hyperparameter Optimization (`src/predictiq/models/tuning.py`, `tests/test_tuning.py`):**
  - `RandomizedSearchCV` on training 5-fold CV optimizing PR-AUC.
  - Random Forest selected as champion with CV PR-AUC $0.6607 \pm 0.0220$, outperforming LightGBM ($0.6598 \pm 0.0287$) and Logistic Regression ($0.6559 \pm 0.0221$).
- **Probability Calibration (`src/predictiq/models/calibration.py`, `tests/test_calibration.py`):**
  - Implemented Platt sigmoid scaling via `CalibratedClassifierCV(method='sigmoid', cv=5)` on training folds.
  - Reduced Brier score from 0.1602 to 0.1354 (+15.42% improvement).
  - Diagnostic reliability diagram saved to `reports/figures/calibration_curve.png`.
- **Decision Threshold Optimization (`src/predictiq/models/thresholds.py`, `tests/test_thresholds.py`):**
  - Implemented OOF threshold sweep optimizing empirical F1-score without speculative cost assumptions.
  - Supported configurable `BusinessCostMatrix`.
  - Selected and froze $p^* = 0.3400$, increasing final test churn catch rate from 57.22% to 71.39% (+14.17 pp).
  - Diagnostic trade-off curve saved to `reports/figures/threshold_tradeoff.png`.
- **Model Explainability (`src/predictiq/models/explainability.py`, `tests/test_explainability.py`):**
  - Integrated TreeSHAP feature attributions on fitted pipeline.
  - Identified top churn predictive drivers: `is_month_to_month`, `Contract_Month-to-month`, `tenure`, `InternetService_Fiber optic`, `is_fiber_optic`.
  - Exported `reports/figures/shap_summary.png`, `shap_local_high_risk.png`, and `shap_local_low_risk.png`.
- **Production Persistence & Inference Integration (`src/predictiq/inference/predictor.py`):**
  - Serialized champion model to `models/churn_champion_model.joblib` and runtime metadata to `models/churn_champion_model_metadata.json`.
  - Updated `ChurnPredictor` to dynamically apply saved threshold during inference.
- **CLI Runner (`scripts/tune_and_train.py`):**
  - Standalone CLI for Phase 4 end-to-end execution.

#### Changed
- Pytest suite expanded from 86 to 97 passing tests (100% pass rate in 20.40s).
- Matplotlib backend explicitly set to `Agg` across all modules to eliminate headless Tkinter dependency.

## [0.3.0] - 2026-10-18

### Phase 3: Feature Engineering, Baseline Modeling & MLflow Experiment Tracking

#### Added
- **Data Splitting & Loading (`src/predictiq/data/loader.py`):**
  - Stratified 80/20 train/test splitting (`random_state=42`) with strict target isolation and zero index overlap.
  - Stripped identifier column `customerID` from model features to prevent key memorization leakage.
  - Validated target distribution (26.54% positive churn prevalence across train and test sets).
- **Domain Feature Engineering (`src/predictiq/features/transformers.py`):**
  - Custom scikit-learn transformer `DomainFeatureEngineer`:
    - `service_count`: integer count of active subscriptions (0 to 9).
    - `streaming_services_count`: count of active entertainment services.
    - Retention flags: `has_tech_support` and `has_online_security`.
    - Risk indicators: `is_month_to_month`, `is_fiber_optic`, and `is_electronic_check`.
    - Categorical `tenure_group` cohorts.
  - Evaluated and documented explicit omission of `TotalCharges / MonthlyCharges` (proxy for tenure with $r > 0.98$ correlation).
- **Leakage-Safe Preprocessing (`src/predictiq/features/pipeline.py`):**
  - Modular `ColumnTransformer` with `SimpleImputer` (median for numericals, most-frequent for categoricals).
  - Continuous feature scaling via `StandardScaler`.
  - Nominal encoding via `OneHotEncoder(handle_unknown='ignore')` for safe unseen categorical handling.
  - End-to-end `build_model_pipeline()` assembler.
- **Baseline Models (`src/predictiq/models/baselines.py`):**
  - Naive baseline: `DummyClassifier(strategy='most_frequent')`.
  - Linear baseline: `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)`.
  - Non-linear tree baseline: `RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, class_weight='balanced')`.
- **Evaluation & Reporting (`src/predictiq/models/evaluation.py`):**
  - Full evaluation framework: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision (PR-AUC), Brier Score, and Confusion Matrix.
  - Automated plot generation: `reports/figures/roc_curves.png`, `reports/figures/pr_curves.png`, `reports/figures/confusion_matrices.png`.
  - Machine-readable exports: `reports/model_comparison.csv` and `reports/model_comparison.json`.
- **MLflow Experiment Tracking (`src/predictiq/tracking/manager.py`):**
  - Resilient tracking manager supporting remote server connections with automatic local filesystem fallback (`./mlruns`).
  - Logged 3 baseline model runs with parameters, metrics, plots, signatures, and input examples.
- **Model Persistence & Inference (`src/predictiq/models/persistence.py`, `src/predictiq/inference/predictor.py`):**
  - Serialized champion `RandomForestClassifier` pipeline and metadata to `models/churn_baseline_model.joblib`.
  - Production inference engine `ChurnPredictor` supporting input schema validation, single/batch prediction, and risk tier stratification (`Low Risk`, `Medium Risk`, `High Risk`).
- **CLI Workflow Runner (`scripts/train_baseline.py`):**
  - Reproducible script for training, cross-validation, and metric reporting.
- **Automated Tests:**
  - `tests/test_data_loader.py`: 7 tests for splitting, stratification, and target validation.
  - `tests/test_features.py`: 6 tests for feature engineering, service counts, and unseen categories.
  - `tests/test_model_evaluation.py`: 4 tests for metric extraction, probability checks, and report generation.
  - `tests/test_inference.py`: 4 tests for pipeline persistence, bit-level reload consistency, and inference schema validation.
  - `tests/test_tracking.py`: 2 tests for MLflow local logging and fallback resolution.

#### Validation Outcomes
- **Test Suite:** **86 of 86 tests passed** in 10.69s.
- **Linter:** Ruff checks completed with 0 errors across all source and test modules.
- **Model Comparison:** Random Forest selected as champion with Average Precision **0.6471** (+0.38 over 0.2654 naive baseline), ROC-AUC **0.8420**, Recall **0.7888**, and F1 **0.6263**.
- **Inference Verification:** Reloaded pipeline passed smoke test with correct risk tier classifications.

---

## [0.2.0] - 2026-10-18

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
