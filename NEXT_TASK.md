# PredictIQ — Next Task: Phase 3 (Feature Engineering & Preprocessing Pipelines)

## 1. Task Definition
**Task Name:** Phase 3 — Feature Engineering & Preprocessing Pipelines  
**Objective:** Design and implement modular, scikit-learn compatible feature transformations, domain feature extractors, and stratified data splitting pipelines under `src/predictiq/features/` to transform cleaned interim data into model-ready matrices without target leakage.

---

## 2. Scope & Boundaries
- **In Scope:**
  1. **Stratified Splitting:** Implement `src/predictiq/features/split.py` with reproducible, stratified train/val/test partitioning (e.g. 70/15/15) preserving the 26.54% churn class ratio across folds.
  2. **Domain Feature Transformers:** Implement `src/predictiq/features/transformers.py`:
     - Ratio features: `charges_to_tenure_ratio = MonthlyCharges / (tenure + 1)`.
     - Bundle depth: `service_depth` (sum of active internet add-ons: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies).
     - Risk indicators: `fiber_without_support = (InternetService == 'Fiber optic') & (TechSupport == 'No')`.
     - Household stability: `household_single = (Partner == 'No') & (Dependents == 'No')`.
  3. **Preprocessing Pipeline:** Implement `src/predictiq/features/pipeline.py`:
     - `ColumnTransformer` with `StandardScaler` for continuous numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`).
     - One-hot encoding for nominal variables (`gender`, `PaymentMethod`, `InternetService`).
     - Ordinal encoding for contract duration (`Month-to-month` -> 0, `One year` -> 1, `Two year` -> 2).
     - Strict fit-on-train-only discipline to eliminate data leakage.
     - Export processed matrices (`train.parquet`, `val.parquet`, `test.parquet`) to `data/processed/`.
  4. **Deterministic Unit Tests:** Implement `tests/test_features.py` validating split proportions, transformer repeatability, dimension preservation, and lack of leakage.
- **Out of Scope (Deferred to Later Phases):**
  - Estimator training, cross-validation, and MLflow run logging (Phase 4).
  - Model serialization and registry packaging (Phase 4).
  - FastAPI real-time prediction endpoints (Phase 5).

---

## 3. Prerequisites
- [x] Phase 1 Project Foundation & Architecture complete, verified, and audited.
- [x] Phase 2 Dataset Acquisition, Validation & EDA complete and verified.
- [x] Clean interim dataset available at `data/interim/Telco-Customer-Churn-cleaned.csv`.
- [ ] Add `scikit-learn>=1.4.0` dependency to `pyproject.toml`.

---

## 4. Files Likely to Change / Be Created
- `pyproject.toml`: Add `scikit-learn>=1.4.0`.
- `src/predictiq/features/__init__.py`: Public exports for feature pipelines.
- `src/predictiq/features/split.py`: Stratified data splitting utilities.
- `src/predictiq/features/transformers.py`: Custom scikit-learn compatible transformers.
- `src/predictiq/features/pipeline.py`: Composable end-to-end preprocessing pipeline.
- `tests/test_features.py`: Unit tests for feature extraction and leak-free transformations.
- `docs/CHANGELOG.md`: Log Phase 3 implementation.
- `CURRENT_STATE.md`: Update milestone progress and verification logs.
- `NEXT_TASK.md`: Update handoff for Phase 4 (Model Training & MLflow Tracking).

---

## 5. Acceptance Criteria
1. Stratified splitting preserves class balance ($26.54\% \pm 0.5\%$) across train, validation, and test sets.
2. Preprocessing pipeline strictly fits on the training set only; validation and test sets are transformed without re-fitting.
3. Feature transformers correctly compute domain features (`service_depth`, `charges_to_tenure_ratio`, `fiber_without_support`).
4. Processed datasets are output to `data/processed/` and remain excluded from Git via `.gitignore`.
5. All new feature tests pass deterministically alongside existing Phase 1 & 2 tests (zero regressions).

---

## 6. Verification Commands
```powershell
# 1. Run full test suite including new feature engineering tests
& ".\.venv\Scripts\pytest.exe" tests -v

# 2. Run feature generation pipeline CLI
& ".\.venv\Scripts\python.exe" -m predictiq.features.pipeline --split

# 3. Verify processed output files exist and are valid
& ".\.venv\Scripts\python.exe" -c "import pandas as pd; df = pd.read_parquet('data/processed/train.parquet'); print(df.shape, df['Churn'].value_counts(normalize=True))"
```
