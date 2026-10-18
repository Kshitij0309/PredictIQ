# PredictIQ — Next Task: Phase 4 (Hyperparameter Optimization, Calibration & Explainability)

## 1. Task Definition
**Task Name:** Phase 4 — Advanced Modeling, Hyperparameter Optimization, Threshold Calibration & Explainability  
**Objective:** Systematically search hyperparameter spaces for advanced gradient boosted decision trees (LightGBM / XGBoost / CatBoost), optimize classification decision thresholds to minimize business churn financial loss, and integrate SHAP model explainability.

---

## 2. Scope & Boundaries
- **In Scope:**
  1. **Gradient Boosting Estimators:**
     - Integrate LightGBM / XGBoost / CatBoost classifiers into PredictIQ pipeline framework.
     - Compare performance against Phase 3 Random Forest and Logistic Regression baselines.
  2. **Systematic Hyperparameter Search:**
     - Implement Optuna or RandomizedSearchCV with cross-validation on training data.
     - Optimize primary metric: Average Precision / PR-AUC.
  3. **Decision Threshold Optimization:**
     - Calibrate probabilities using isotonic regression or Platt scaling.
     - Formulate business financial cost-benefit matrix:
       - Cost of retention outreach campaign ($C_{\text{action}}$)
       - Customer Lifetime Value preserved ($V_{\text{salvage}}$)
       - Cost of false negative churn loss
     - Determine profit-maximizing decision threshold $p^*$ on training cross-validation folds.
  4. **Model Explainability & Transparency:**
     - Compute TreeSHAP values for global feature importances and local decision waterfall plots.
     - Export explainability summary figures into `reports/figures/shap_summary.png`.
  5. **Deterministic Unit Tests:**
     - Tests for hyperparameter search pipeline, threshold selection logic, and SHAP computation.
- **Out of Scope (Deferred to Later Phases):**
  - FastAPI serving endpoints and Docker containerization (Phase 5).
  - Data drift and model monitoring service (Phase 6).

---

## 3. Prerequisites
- [x] Phase 1 Project Foundation verified.
- [x] Phase 2 Dataset Acquisition, Validation & EDA verified.
- [x] Phase 3 Feature Engineering, Baseline Modeling & MLflow Tracking verified (86 passing tests, champion Random Forest model saved).
- [ ] Add `lightgbm` and `shap` to `pyproject.toml`.

---

## 4. Acceptance Criteria
1. Gradient boosting models evaluate against the baseline suite under identical 80/20 train/test splits.
2. Cross-validation hyperparameter search yields a champion model with documented improvements in PR-AUC.
3. Decision threshold $p^*$ is derived strictly from training CV folds to prevent test set data leakage.
4. SHAP feature attribution artifacts are automatically logged to MLflow and saved under `reports/figures/`.
5. All new tests pass deterministically alongside existing 86 tests (zero regressions).

---

## 5. Verification Commands
```powershell
# 1. Run all unit and integration tests
pytest -v

# 2. Run advanced model tuning workflow
python scripts/tune_models.py

# 3. Inspect champion model comparison
cat reports/model_comparison.csv
```
